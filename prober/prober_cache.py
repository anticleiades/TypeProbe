from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from prober_activations import _fim_token_indices, _pool_activation
from prober_data import collate_batch, get_dataset_path


def _hash_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _collect_run_fingerprint() -> Dict[str, object]:
    import platform

    fingerprint = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }
    if torch.cuda.is_available():
        fingerprint["gpu_name"] = torch.cuda.get_device_name(0)
    else:
        fingerprint["gpu_name"] = None
    try:
        import transformers

        fingerprint["transformers"] = transformers.__version__
    except Exception:
        fingerprint["transformers"] = None
    try:
        import transformer_lens

        fingerprint["transformer_lens"] = transformer_lens.__version__
    except Exception:
        fingerprint["transformer_lens"] = None
    return fingerprint


def _write_run_fingerprint(save_dir: Path, fingerprint: Dict[str, object]) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / "run_fingerprint.json"
    txt_path = save_dir / "run_fingerprint.txt"

    json_path.write_text(json.dumps(fingerprint, indent=2, sort_keys=True))
    lines = [f"{k}: {v}" for k, v in sorted(fingerprint.items())]
    txt_path.write_text("\n".join(lines) + "\n")


def _build_cache_meta(
    *,
    layer: int,
    act_name: str,
    dataset: Dataset,
    model_name: str,
    use_java: bool,
    pool: str,
    seed: int,
    dtype: Optional[str] = None,
    fingerprint: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    dataset_path = get_dataset_path(dataset)
    if dataset_path is None:
        raise ValueError(
            "Dataset path unavailable; reproducible cache metadata requires "
            "a V2PromptDataset or Subset of one."
        )
    meta = {
        "layer": int(layer),
        "act_name": act_name,
        "n_samples": int(len(dataset)),
        "model_name": model_name,
        "use_java": bool(use_java),
        "pool": pool,
        "dataset_path": str(dataset_path.resolve()),
        "seed": int(seed),
        "dataset_sha256": _hash_file(dataset_path),
    }
    if dtype is not None:
        meta["dtype"] = str(dtype)
    if fingerprint is not None:
        meta["fingerprint"] = fingerprint
    return meta


def _validate_cached_meta(*, cache_dir: Path, layer: int, expected: Dict[str, Any]) -> None:
    meta_path = cache_dir / f"meta_layer{layer}.json"
    if not meta_path.exists():
        raise ValueError(f"Missing cache metadata: {meta_path}")

    cached = json.loads(meta_path.read_text())

    def normalize(d: Dict[str, Any]) -> Dict[str, Any]:
        # remove datasetpath
        res = {k: v for k, v in d.items() if k != "dataset_path"}

        sha = res.pop("dataset_sha256", None) or res.get("fingerprint", {}).get("dataset_sha256")

        fp = res.get("fingerprint", {})
        if isinstance(fp, dict):
            fp = {k: v for k, v in fp.items() if k != "dataset_path"}
        else:
            fp = {}

        if sha:
            fp["dataset_sha256"] = sha

        res["fingerprint"] = fp
        return res

    exp_norm = normalize(expected)
    cache_norm = normalize(cached)

    # (same layer, activation, model, pooling strategy, dtype and fingerprint)
    keys_to_check = [
        "layer",
        "act_name",
        "model_name",
        "pool",
        "dtype",
        "fingerprint"  # SW+HW env + datasethash
    ]

    mismatch = {}
    for k in keys_to_check:
        v_exp = exp_norm.get(k)
        v_cache = cache_norm.get(k)

        if v_exp != v_cache:
            mismatch[k] = {"expected": v_exp, "cached": v_cache}

    if mismatch:
        raise ValueError(
            "[Reproducibility] Cache metadata mismatch! Configuration or environment has changed:\n"
            f"{json.dumps(mismatch, indent=2, sort_keys=True)}"
        )


def _cache_layer_activations(
    *,
    dataset,
    model,
    act_name: str,
    layer: int,
    batch_size: int,
    cache_dir: Path,
    overwrite: bool,
    use_java: bool,
    pool: str,
    model_name: str,
    expected_meta: Optional[Dict[str, object]] = None,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    acts_path = cache_dir / f"acts_layer{layer}.npy"
    labels_path = cache_dir / "labels.npy"
    meta_path = cache_dir / f"meta_layer{layer}.json"

    if acts_path.exists() and not overwrite:
        if expected_meta is not None:
            if not meta_path.exists():
                raise ValueError(
                    f"Missing cache metadata for reproducible run: {meta_path}"
                )
            cached_meta = json.loads(meta_path.read_text())
            mismatch = {
                k: {"expected": expected_meta.get(k), "cached": cached_meta.get(k)}
                for k in expected_meta
                if cached_meta.get(k) != expected_meta.get(k)
            }
            if mismatch:
                raise ValueError(
                    "Cached activations do not match current configuration: "
                    f"{json.dumps(mismatch, indent=2)}"
                )
        if not labels_path.exists():
            raise ValueError(
                f"Cached activations exist without labels: {labels_path}"
            )
        existing_labels = np.load(labels_path, mmap_mode="r")
        if existing_labels.shape[0] != len(dataset):
            raise ValueError(
                f"Cached labels rows ({existing_labels.shape[0]}) "
                f"do not match dataset rows ({len(dataset)})."
            )
        print(f"[cache] exists -> {acts_path}")
        return

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    iterator = iter(loader)
    try:
        first_prompts, first_labels = next(iterator)
    except StopIteration:
        raise ValueError("Dataset is empty; cannot cache activations.")

    label_dim = int(first_labels.shape[1]) if first_labels.ndim > 1 else 1
    need_write_labels = True
    if labels_path.exists() and not overwrite:
        existing_labels = np.load(labels_path, mmap_mode="r")
        if (
            existing_labels.shape[0] == len(dataset)
            and existing_labels.shape[1] == label_dim
        ):
            need_write_labels = False
        else:
            raise ValueError(
                f"Cached labels shape {existing_labels.shape} does not match dataset."
            )

    device = model.cfg.device
    toks = model.to_tokens(first_prompts).to(device)
    token_indices = None
    if pool == "fim":
        token_indices = _fim_token_indices(
            toks=toks, tokenizer=model.tokenizer, model_name=model_name
        )
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=lambda name: name == act_name)
        act = cache[act_name]
    feats = _pool_activation(act, pool=pool, token_indices=token_indices).detach().cpu()
    in_features = feats.shape[-1]
    feats_dtype = feats.numpy().dtype

    acts_memmap = np.lib.format.open_memmap(
        acts_path,
        mode="w+",
        dtype=feats_dtype,
        shape=(len(dataset), in_features),
    )
    labels_memmap = None
    if need_write_labels:
        labels_memmap = np.lib.format.open_memmap(
            labels_path,
            mode="w+",
            dtype=np.int64,
            shape=(len(dataset), label_dim),
        )

    offset = 0
    from tqdm import tqdm

    with tqdm(total=len(dataset), desc=f"cache layer {layer}", leave=False) as pbar:
        batch_size_actual = feats.shape[0]
        acts_memmap[offset : offset + batch_size_actual] = feats.numpy()
        if labels_memmap is not None:
            labels_memmap[offset : offset + batch_size_actual] = first_labels.numpy()
        offset += batch_size_actual
        pbar.update(batch_size_actual)

        for prompts, labels in iterator:
            toks = model.to_tokens(prompts).to(device)
            token_indices = None
            if pool == "fim":
                token_indices = _fim_token_indices(
                    toks=toks, tokenizer=model.tokenizer, model_name=model_name
                )
            with torch.no_grad():
                _, cache = model.run_with_cache(toks, names_filter=lambda name: name == act_name)
                act = cache[act_name]
            feats = _pool_activation(act, pool=pool, token_indices=token_indices).detach().cpu()
            batch_size_actual = feats.shape[0]
            acts_memmap[offset : offset + batch_size_actual] = feats.numpy()
            if labels_memmap is not None:
                labels_memmap[offset : offset + batch_size_actual] = labels.numpy()
            offset += batch_size_actual
            pbar.update(batch_size_actual)

    meta = {
        "layer": layer,
        "act_name": act_name,
        "n_samples": len(dataset),
        "in_features": int(in_features),
        "dtype": str(feats_dtype),
        "model_name": model_name,
        "use_java": bool(use_java),
        "pool": pool,
    }
    if expected_meta is not None:
        for key in ("dataset_path", "seed", "fingerprint"):
            if key in expected_meta:
                meta[key] = expected_meta[key]
    meta_path.write_text(json.dumps(meta, indent=2))
