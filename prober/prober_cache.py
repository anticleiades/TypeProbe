from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from prober_activations import _fim_token_indices, _pool_activation, _resolve_act_name, _fim_token_indicesLegacy
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

def tokenize_fixup(
    prompts: list[tuple[str, str]], tokenizer, cfg, model_key: str
) -> torch.Tensor:
    FIM_TOKENS_BY_MODEL = {
        "bigcode/santacoder": {
            "prefix": "<fim-prefix>",
            "suffix": "<fim-suffix>",
            "middle": "<fim-middle>",
        },
        "codellama/CodeLlama-7b-hf": {
            "prefix": "<PRE>",
            "suffix": "<SUF>",
            "middle": "<MID>",
        },
    }

    fim = FIM_TOKENS_BY_MODEL.get(model_key)
    if fim is None:
        raise RuntimeError(f"FIM not found for {model_key}")

    pre_ids = tokenizer.encode(fim["prefix"], add_special_tokens=False)
    suf_ids = tokenizer.encode(fim["suffix"], add_special_tokens=False)
    mid_ids = tokenizer.encode(fim["middle"], add_special_tokens=False)

    if len(pre_ids) != 1:
        raise ValueError(f"CRITICAL: Prefix FIM token was encoded into {len(pre_ids)} tokens.")
    if len(suf_ids) != 1:
        raise ValueError(f"CRITICAL: Suffix FIM token was encoded into {len(suf_ids)} tokens.")
    if len(mid_ids) != 1:
        raise ValueError(f"CRITICAL: Middle FIM token was encoded into {len(mid_ids)} tokens.")

    true_pre_id = pre_ids[0]
    true_suf_id = suf_ids[0]
    true_mid_id = mid_ids[0]

    batch_ids = []

    for prefix_code, suffix_code in prompts:
        prefix_ids = tokenizer.encode(prefix_code, add_special_tokens=False)
        suffix_ids = tokenizer.encode(suffix_code, add_special_tokens=False)

        final_seq = (
            [true_pre_id]
            + prefix_ids
            + [true_suf_id]
            + suffix_ids
            + [true_mid_id]
        )

        if cfg.default_prepend_bos and tokenizer.bos_token_id is not None:
            final_seq = [tokenizer.bos_token_id] + final_seq

        batch_ids.append(final_seq)

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    max_len = max(len(seq) for seq in batch_ids)
    toks = torch.full((len(batch_ids), max_len), pad_id, dtype=torch.long)

    for row, seq in enumerate(batch_ids):
        toks[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)

    return toks

def _cache_layer_activationsV2(
        *,
        dataset,
        model,
        arg_act: str,
        batch_size: int,
        cache_dir: Path,
        overwrite: bool,
        use_java: bool,
        pool: str,
        model_name: str,
        codellama_fixup: bool = True,
        seed,
        run_fingerprint: Optional[Dict[str, object]] = None,
        dtype
) -> None:
    if dataset is None:
        raise RuntimeError("null dataset")
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_layers = model.cfg.n_layers
    act_names = {l: _resolve_act_name(arg_act, l) for l in range(n_layers)}

    meta_per_layer = {
        l: _build_cache_meta(
            layer=l,
            act_name=act_names[l],
            dataset=dataset,
            model_name=model_name,
            use_java=use_java,
            pool=pool,
            seed=seed,
            dtype=str(dtype).replace("torch.", "") if dtype is not None else None,
            fingerprint=run_fingerprint,
        ) for l in range(n_layers)
    }

    names_filter = lambda name: name in act_names.values()
    labels_path = cache_dir / "labels.npy"
    all_exist = all((cache_dir / f"acts_layer{l}.npy").exists() for l in range(n_layers))

    if all_exist and labels_path.exists() and not overwrite:
        print("[cache] All layers already exist. Skipping.")
        return

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    device = model.cfg.device

    acts_memmaps = {}
    labels_memmap = None
    offset = 0
    in_features = None
    feats_dtype = None
    label_dim = None

    from tqdm import tqdm
    with tqdm(total=len(dataset), desc=f"caching {n_layers} layers") as pbar:

        for prompts, labels, _, raw_prompts in loader:
            if codellama_fixup:
                toks = tokenize_fixup(prompts, model.tokenizer, model.cfg, model_key=model_name).to(device)
            else:
                toks = model.to_tokens(raw_prompts).to(device)
            token_indices = None
            if pool == "fim":
                if codellama_fixup:
                    __fim_tkn_impl = _fim_token_indices
                else:
                    __fim_tkn_impl = _fim_token_indicesLegacy
                token_indices = __fim_tkn_impl(
                    toks=toks, tokenizer=model.tokenizer, model_name=model_name
                )

            with torch.no_grad():
                _, cache = model.run_with_cache(toks, names_filter=names_filter)

            batch_size_actual = toks.shape[0]

            if offset == 0:
                label_dim = int(labels.shape[1]) if labels.ndim > 1 else 1

                act_0 = cache[act_names[0]]
                feats_0 = _pool_activation(act_0, pool=pool, token_indices=token_indices).detach().cpu()
                in_features = feats_0.shape[-1]
                feats_dtype = feats_0.numpy().dtype

                for l in range(n_layers):
                    acts_path = cache_dir / f"acts_layer{l}.npy"
                    acts_memmaps[l] = np.lib.format.open_memmap(
                        acts_path,
                        mode="w+",
                        dtype=feats_dtype,
                        shape=(len(dataset), in_features),
                    )

                labels_memmap = np.lib.format.open_memmap(
                    labels_path,
                    mode="w+",
                    dtype=np.int64,
                    shape=(len(dataset), label_dim),
                )

            for l, act_name in act_names.items():
                act = cache[act_name]
                feats = _pool_activation(act, pool=pool, token_indices=token_indices).detach().cpu().numpy()
                acts_memmaps[l][offset: offset + batch_size_actual] = feats

            labels_memmap[offset: offset + batch_size_actual] = labels.numpy()

            offset += batch_size_actual
            pbar.update(batch_size_actual)

            del cache
            del _

    for l, act_name in act_names.items():
        meta = {
            "layer": l,
            "act_name": act_name,
            "n_samples": len(dataset),
            "in_features": int(in_features),
            "dtype": str(feats_dtype),
            "model_name": model_name,
            "use_java": bool(use_java),
            "pool": pool,
        }
        if meta_per_layer[l] is not None:
            for key in ("dataset_path", "seed", "fingerprint"):
                if key in meta_per_layer[l]:
                    meta[key] = meta_per_layer[l][key]
        (cache_dir / f"meta_layer{l}.json").write_text(json.dumps(meta, indent=2))

    for l in acts_memmaps:
        acts_memmaps[l].flush()
    labels_memmap.flush()