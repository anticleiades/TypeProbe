import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformer_lens import HookedTransformer

from prober_cli import parse_args
from prober_activations import _resolve_act_name
from prober_cache import (
    _build_cache_meta,
    _cache_layer_activations,
    _collect_run_fingerprint,
    _hash_file,
    _validate_cached_meta,
    _write_run_fingerprint,
)
from prober_adapters import get_adapter
from prober_data import (
    CachedActivationDataset,
    RandomTargetsDataset,
    _make_random_labels,
    collate_batch,
    collate_batch_cached,
    get_dataset_path,
)
from prober_train import (
    BatchedLinear,
    _aggregate_causality_metrics,
    _causality_metrics,
    _eval_predictions,
    _eval_predictions_cached,
    _run_kfold,
    _run_kfold_cached,
    _run_single_split,
    _run_single_split_cached,
    _write_causality_report,
)


def _suffix_dir(path: Path, suffix: str) -> Path:
    return path.parent / f"{path.name}{suffix}"


def _suffix_file(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def main() -> None:
    args = parse_args()
    if args.save_plot is not None:
        args.plot_path = args.save_plot
    if args.cache_acts and args.use_cache:
        raise ValueError("Use either --cache-acts or --use-cache, not both.")
    if args.dump_acts and args.use_cache:
        raise ValueError("--dump-acts requires a live model; do not use with --use-cache.")
    if args.eval_only:
        if args.probe_path is None and args.eval_dir is None:
            raise ValueError("--eval-only requires --probe-path or --eval-dir.")
        if args.cache_acts:
            raise ValueError("--eval-only cannot be combined with --cache-acts.")
        if args.eval_dir is not None and args.probe_path is not None:
            raise ValueError("--eval-only: use either --eval-dir or --probe-path, not both.")
    if args.cache_acts and args.control_task:
        raise ValueError("--control-task is not supported with --cache-acts.")
    if args.disable_kfold and args.k_folds < 2:
        raise ValueError("--disable-kfold requires --k-folds >= 2 to set test split size.")

    if args.repro:
        if not torch.cuda.is_available():
            raise ValueError("--repro requires CUDA; none detected.")
        if not args.model_revision:
            raise ValueError("--repro requires --model-revision to pin weights.")
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as exc:
            raise RuntimeError("--repro was requested, but deterministic algorithms could not be enabled.") from exc
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    adapter = get_adapter(args.adapter)
    meta = adapter.load_meta(args.metadata, args)
    class_counts = meta.class_counts

    dataset = adapter.build_dataset(args, meta)
    dataset = adapter.apply_java_filter(dataset, args)
    if args.causality and len(class_counts) < 3:
        raise ValueError("--causality requires at least 3 tasks.")
    if args.act is None and not args.eval_only:
        print("Specify the activation name with the --act param.")
        exit(0)
    print(f"[dataset] rows={len(dataset)}")
    if args.java:
        print(f"[dataset] rows_java_tagged={len(dataset)}")
    label_override = None
    if args.control_task:
        label_override = _make_random_labels(
            n_samples=len(dataset),
            class_counts=class_counts,
            seed=args.seed,
        )
    if label_override is not None and not args.use_cache:
        dataset = RandomTargetsDataset(dataset, label_override)

    cache_dir = args.cache_dir or (args.save_dir / "cache")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    if args.use_cache:
        model = None
        # device = None
        # dtype = None
    else:
        model_kwargs = {
            "device": device,
            "dtype": dtype,
            "trust_remote_code": True,
        }
        if args.model_revision:
            model_kwargs["revision"] = args.model_revision
        model = HookedTransformer.from_pretrained_no_processing(args.model, **model_kwargs)
        model.eval()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    run_fingerprint = None
    if args.repro:
        run_fingerprint = _collect_run_fingerprint()
        dataset_path = get_dataset_path(dataset)
        if dataset_path is not None:
            run_fingerprint["dataset_path"] = str(dataset_path.resolve())
            run_fingerprint["dataset_sha256"] = _hash_file(dataset_path)
        _write_run_fingerprint(args.save_dir, run_fingerprint)
    if args.dump_acts:
        if len(dataset) == 0:
            raise ValueError("Dataset is empty; cannot dump activations.")
        sample_prompt = dataset[0]["prompt"]
        toks = model.to_tokens([sample_prompt]).to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(toks)
        for name in sorted(cache.keys()):
            print(name)
        return

    probe_payload = None
    eval_payloads = None
    if args.eval_only and args.eval_dir is not None:
        if not args.eval_dir.exists():
            raise FileNotFoundError(f"Missing eval dir: {args.eval_dir}")
        pattern = re.compile(r"prober_layer(\d+)_best\.pt$")
        eval_payloads = {}
        for path in sorted(args.eval_dir.glob("prober_layer*_best.pt")):
            match = pattern.match(path.name)
            if not match:
                continue
            layer = int(match.group(1))
            eval_payloads[layer] = torch.load(path, map_location="cpu", weights_only=False)
        if not eval_payloads:
            raise ValueError(f"No prober_layer*_best.pt files found in {args.eval_dir}.")
        if args.act is None and not args.use_cache:
            for payload in eval_payloads.values():
                if "act_name" not in payload:
                    raise ValueError("--act is required when evaluating without cached activations.")
        layers = sorted(eval_payloads.keys())
    elif args.eval_only:
        probe_payload = torch.load(args.probe_path, map_location="cpu", weights_only=False)
        probe_layer = int(probe_payload.get("layer", args.layer))
        if args.act is None and "act_name" not in probe_payload and not args.use_cache:
            raise ValueError("--act is required when evaluating without cached activations.")
        layers = [probe_layer]
    else:
        layer_end = args.layer_end
        if layer_end is None:
            if model is None:
                raise ValueError("--layer-end must be provided when no live model is loaded.")
            layer_end = model.cfg.n_layers - 1
        layers = list(range(args.layer_start, layer_end + 1))
    print(args)
    layer_accs: List[List[float]] = []
    for layer in tqdm(layers, desc="layers"):
        if eval_payloads is not None:
            probe_payload = eval_payloads[layer]
        if args.act is None and probe_payload is not None and "act_name" in probe_payload:
            act_name = probe_payload["act_name"]
        elif args.use_cache and args.eval_only and args.act is None:
            act_name = None
        else:
            act_name = _resolve_act_name(args.act, layer)
        expected_meta = None
        if args.repro and act_name is not None:
            expected_meta = _build_cache_meta(
                layer=layer,
                act_name=act_name,
                dataset=dataset,
                model_name=args.model,
                use_java=args.java,
                pool=args.pool,
                seed=args.seed,
                dtype=str(dtype).replace("torch.", "") if dtype is not None else None,
                fingerprint=run_fingerprint,
            )
        if args.cache_acts:
            _cache_layer_activations(
                dataset=dataset,
                model=model,
                act_name=act_name,
                layer=layer,
                batch_size=args.batch_size,
                cache_dir=cache_dir,
                overwrite=args.overwrite_cache,
                use_java=args.java,
                pool=args.pool,
                model_name=args.model,
                expected_meta=expected_meta,
            )
            continue
        cache_fingerprint = None
        if args.use_cache:
            acts_path = cache_dir / f"acts_layer{layer}.npy"
            labels_path = cache_dir / "labels.npy"
            if not acts_path.exists():
                raise FileNotFoundError(f"Missing cached activations: {acts_path}")
            if not labels_path.exists():
                raise FileNotFoundError(f"Missing cached labels: {labels_path}")
            meta_path = cache_dir / f"meta_layer{layer}.json"
            if meta_path.exists():
                cached_meta = json.loads(meta_path.read_text())
                cache_fingerprint = cached_meta.get("fingerprint")
            if args.repro and expected_meta is not None:
                _validate_cached_meta(cache_dir=cache_dir, layer=layer, expected=expected_meta)
            cached_dataset = CachedActivationDataset(acts_path, labels_path)
            if label_override is not None:
                cached_dataset = RandomTargetsDataset(cached_dataset, label_override)
            if args.eval_only:
                state_dict = probe_payload.get("state_dict", probe_payload)
                weight = state_dict.get("weight")
                if weight is None:
                    raise ValueError("Probe payload is missing 'weight' in state_dict.")
                out_features = int(weight.shape[1])
                in_features = int(weight.shape[2])
                n_tasks = int(weight.shape[0])
                prober = BatchedLinear(n_tasks, in_features, out_features)
                prober.load_state_dict(state_dict)
                class_counts_eval = probe_payload.get("class_counts", class_counts)
                if len(class_counts_eval) != n_tasks:
                    raise ValueError("class_counts does not match probe task count.")
                accs, labels_np, preds_np = _eval_predictions_cached(
                    prober=prober,
                    loader=DataLoader(
                        cached_dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        collate_fn=collate_batch_cached,
                    ),
                    class_counts=class_counts_eval,
                )
                acc_str = ", ".join(f"{a:.3f}" for a in accs)
                print(f"[eval-only] layer={layer} acc=[{acc_str}]")
                if args.causality:
                    metrics = _causality_metrics(labels_np, preds_np)
                    _write_causality_report(
                        save_dir=args.save_dir, layer=layer, fold=0, metrics=metrics
                    )
                if eval_payloads is not None:
                    layer_accs.append(accs)
                continue
            if args.disable_kfold:
                results = _run_single_split_cached(
                    dataset=cached_dataset,
                    class_counts=class_counts,
                    val_frac=args.val_frac,
                    test_frac=1.0 / args.k_folds,
                    seed=args.seed,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    layer=layer,
                    save_dir=args.save_dir,
                    causality=args.causality,
                    cache_fingerprint=cache_fingerprint,
                    train_fingerprint=run_fingerprint,
                )
            else:
                results = _run_kfold_cached(
                    dataset=cached_dataset,
                    class_counts=class_counts,
                    k_folds=args.k_folds,
                    val_frac=args.val_frac,
                    seed=args.seed,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    layer=layer,
                    save_dir=args.save_dir,
                    causality=args.causality,
                    cache_fingerprint=cache_fingerprint,
                    train_fingerprint=run_fingerprint,
                )
        else:
            if args.eval_only:
                state_dict = probe_payload.get("state_dict", probe_payload)
                weight = state_dict.get("weight")
                if weight is None:
                    raise ValueError("Probe payload is missing 'weight' in state_dict.")
                out_features = int(weight.shape[1])
                in_features = int(weight.shape[2])
                n_tasks = int(weight.shape[0])
                prober = BatchedLinear(n_tasks, in_features, out_features).to(device)
                prober.load_state_dict(state_dict)
                loader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    collate_fn=collate_batch,
                )
                class_counts_eval = probe_payload.get("class_counts", class_counts)
                if len(class_counts_eval) != n_tasks:
                    raise ValueError("class_counts does not match probe task count.")
                accs, labels_np, preds_np = _eval_predictions(
                    model=model,
                    prober=prober,
                    loader=loader,
                    act_name=act_name,
                    class_counts=class_counts_eval,
                    pool=args.pool,
                    model_name=args.model,
                )
                acc_str = ", ".join(f"{a:.3f}" for a in accs)
                print(f"[eval-only] layer={layer} acc=[{acc_str}]")
                if args.causality:
                    metrics = _causality_metrics(labels_np, preds_np)
                    _write_causality_report(
                        save_dir=args.save_dir, layer=layer, fold=0, metrics=metrics
                    )
                if eval_payloads is not None:
                    layer_accs.append(accs)
                continue
            if args.disable_kfold:
                results = _run_single_split(
                    dataset=dataset,
                    model=model,
                    act_name=act_name,
                    class_counts=class_counts,
                    val_frac=args.val_frac,
                    test_frac=1.0 / args.k_folds,
                    seed=args.seed,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    layer=layer,
                    save_dir=args.save_dir,
                    causality=args.causality,
                    pool=args.pool,
                    model_name=args.model,
                    cache_fingerprint=cache_fingerprint,
                    train_fingerprint=run_fingerprint,
                )
            else:
                results = _run_kfold(
                    dataset=dataset,
                    model=model,
                    act_name=act_name,
                    class_counts=class_counts,
                    k_folds=args.k_folds,
                    val_frac=args.val_frac,
                    seed=args.seed,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    layer=layer,
                    save_dir=args.save_dir,
                    causality=args.causality,
                    pool=args.pool,
                    model_name=args.model,
                    cache_fingerprint=cache_fingerprint,
                    train_fingerprint=run_fingerprint,
                )
        if args.cache_acts:
            continue
        mean_acc = np.mean([r.test_acc for r in results], axis=0).tolist()
        layer_accs.append(mean_acc)
        acc_str = ", ".join(f"{a:.3f}" for a in mean_acc)
        print(f"[layer {layer}] mean_test_acc=[{acc_str}]")

    if args.cache_acts:
        print(f"[cache] saved to {cache_dir}")
        return
    if args.eval_only and eval_payloads is None:
        return

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to plot accuracy per layer."
        ) from exc

    layer_accs_arr = np.array(layer_accs)
    plt.figure(figsize=(10, 6))
    for task_idx in range(layer_accs_arr.shape[1]):
        plt.plot(layers, layer_accs_arr[:, task_idx], marker="o", label=f"task{task_idx}")
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title("Prober Accuracy per Layer")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    args.plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.plot_path)
    print(f"[plot] saved -> {args.plot_path}")

    if args.causality:
        series = _aggregate_causality_metrics(
            save_dir=args.save_dir,
            layers=layers,
            k_folds=args.k_folds,
            disable_kfold=args.disable_kfold,
        )
        labels = {
            "task2_acc_task01_correct": "task0+task1 correct",
            "task2_acc_task01_wrong": "task0+task1 wrong",
            "task2_acc_task0_correct_task1_wrong": "task0 correct, task1 wrong",
            "task2_acc_task0_wrong_task1_correct": "task0 wrong, task1 correct",
        }
        plt.figure(figsize=(10, 6))
        for key, label in labels.items():
            plt.plot(layers, series[key], marker="o", label=label)
        plt.xlabel("Layer")
        plt.ylabel("Task2 accuracy")
        plt.title("Causality: Task2 Accuracy by Task0/1 Outcome")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plot_path = args.plot_path
        causality_path = plot_path.with_name(f"{plot_path.stem}_causal{plot_path.suffix}")
        plt.savefig(causality_path)
        print(f"[plot] saved -> {causality_path}")

    if args.acc_json is not None:
        acc_report: Dict[str, Dict[str, float]] = {}
        for layer, accs in zip(layers, layer_accs):
            acc_report[f"layer_{int(layer)}"] = {
                f"task{task_idx}": float(accs[task_idx])
                for task_idx in range(len(accs))
            }

        out_path = args.acc_json
        if not out_path.is_absolute():
            out_path = args.save_dir / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(acc_report, indent=2, sort_keys=True))
        print(f"[json] saved -> {out_path}")


if __name__ == "__main__":
    main()
