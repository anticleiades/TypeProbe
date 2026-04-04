from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformer_lens import HookedTransformer

from prober_activations import _fim_token_indices, _pool_activation
from prober_data import (
    CachedActivationDataset,
    V2PromptDataset,
    collate_batch,
    collate_batch_cached,
)


class BatchedLinear(nn.Module):
    def __init__(self, n_tasks: int, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_tasks, out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(n_tasks, out_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bi,toi->tbo", x, self.weight) + self.bias[:, None, :]


def _task_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_counts: List[int],
    criterion: nn.Module,
) -> Tuple[torch.Tensor, List[float]]:
    losses = []
    accs = []
    for task_idx, n_cls in enumerate(class_counts):
        logit = logits[task_idx, :, :n_cls]
        target = labels[:, task_idx]
        loss = criterion(logit, target)
        losses.append(loss)
        pred = logit.argmax(dim=-1)
        accs.append((pred == target).float().mean().item())
    return sum(losses), accs


def _eval_predictions(
    *,
    model: HookedTransformer,
    prober: BatchedLinear,
    loader: DataLoader,
    act_name: str,
    class_counts: List[int],
    pool: str,
    model_name: str,
) -> Tuple[List[float], np.ndarray, np.ndarray]:
    device = model.cfg.device
    total_acc = np.zeros(len(class_counts), dtype=np.float64)
    total = 0
    all_labels: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []

    prober.eval()
    for prompts, labels in tqdm(loader, desc="eval batch", leave=False):
        labels = labels.to(device)
        toks = model.to_tokens(prompts).to(device)
        token_indices = None
        if pool == "fim":
            token_indices = _fim_token_indices(
                toks=toks, tokenizer=model.tokenizer, model_name=model_name
            )
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, names_filter=lambda name: name == act_name)
            act = cache[act_name]
        feats = _pool_activation(act, pool=pool, token_indices=token_indices)
        if feats.dtype != prober.weight.dtype:
            feats = feats.to(prober.weight.dtype)
        with torch.no_grad():
            logits = prober(feats)
            preds = torch.empty_like(labels)
            for task_idx, n_cls in enumerate(class_counts):
                logit = logits[task_idx, :, :n_cls]
                pred = logit.argmax(dim=-1)
                preds[:, task_idx] = pred
                total_acc[task_idx] += (pred == labels[:, task_idx]).float().sum().item()

        all_labels.append(labels.detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())
        total += labels.shape[0]

    if total == 0:
        empty = np.empty((0, len(class_counts)), dtype=np.int64)
        return [0.0 for _ in class_counts], empty, empty
    return (
        (total_acc / total).tolist(),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_preds, axis=0),
    )


def _eval_predictions_cached(
    *,
    prober: BatchedLinear,
    loader: DataLoader,
    class_counts: List[int],
) -> Tuple[List[float], np.ndarray, np.ndarray]:
    device = prober.weight.device
    total_acc = np.zeros(len(class_counts), dtype=np.float64)
    total = 0
    all_labels: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []

    prober.eval()
    for feats, labels in tqdm(loader, desc="eval batch", leave=False):
        labels = labels.to(device)
        feats = feats.to(device)
        if feats.dtype != prober.weight.dtype:
            feats = feats.to(prober.weight.dtype)
        with torch.no_grad():
            logits = prober(feats)
            preds = torch.empty_like(labels)
            for task_idx, n_cls in enumerate(class_counts):
                logit = logits[task_idx, :, :n_cls]
                pred = logit.argmax(dim=-1)
                preds[:, task_idx] = pred
                total_acc[task_idx] += (pred == labels[:, task_idx]).float().sum().item()

        all_labels.append(labels.detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())
        total += labels.shape[0]

    if total == 0:
        empty = np.empty((0, len(class_counts)), dtype=np.int64)
        return [0.0 for _ in class_counts], empty, empty
    return (
        (total_acc / total).tolist(),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_preds, axis=0),
    )


def _causality_metrics(labels: np.ndarray, preds: np.ndarray) -> Dict[str, object]:
    if labels.size == 0:
        return {
            "n_samples": 0,
            "n_task01_correct": 0,
            "n_task01_wrong": 0,
            "n_task0_correct_task1_wrong": 0,
            "n_task0_wrong_task1_correct": 0,
            "task2_acc_overall": None,
            "task2_acc_task01_correct": None,
            "task2_acc_task01_wrong": None,
            "task2_acc_task0_correct_task1_wrong": None,
            "task2_acc_task0_wrong_task1_correct": None,
        }
    task2_correct = preds[:, 2] == labels[:, 2]
    task01_correct = (preds[:, 0] == labels[:, 0]) & (preds[:, 1] == labels[:, 1])
    task01_wrong = (preds[:, 0] != labels[:, 0]) & (preds[:, 1] != labels[:, 1])
    task0_correct_task1_wrong = (preds[:, 0] == labels[:, 0]) & (
        preds[:, 1] != labels[:, 1]
    )
    task0_wrong_task1_correct = (preds[:, 0] != labels[:, 0]) & (
        preds[:, 1] == labels[:, 1]
    )
    overall_acc = float(task2_correct.mean())
    acc_task01_correct = (
        float(task2_correct[task01_correct].mean()) if task01_correct.any() else None
    )
    acc_task01_wrong = (
        float(task2_correct[task01_wrong].mean()) if task01_wrong.any() else None
    )
    acc_task0_correct_task1_wrong = (
        float(task2_correct[task0_correct_task1_wrong].mean())
        if task0_correct_task1_wrong.any()
        else None
    )
    acc_task0_wrong_task1_correct = (
        float(task2_correct[task0_wrong_task1_correct].mean())
        if task0_wrong_task1_correct.any()
        else None
    )
    return {
        "n_samples": int(labels.shape[0]),
        "n_task01_correct": int(task01_correct.sum()),
        "n_task01_wrong": int(task01_wrong.sum()),
        "n_task0_correct_task1_wrong": int(task0_correct_task1_wrong.sum()),
        "n_task0_wrong_task1_correct": int(task0_wrong_task1_correct.sum()),
        "task2_acc_overall": overall_acc,
        "task2_acc_task01_correct": acc_task01_correct,
        "task2_acc_task01_wrong": acc_task01_wrong,
        "task2_acc_task0_correct_task1_wrong": acc_task0_correct_task1_wrong,
        "task2_acc_task0_wrong_task1_correct": acc_task0_wrong_task1_correct,
    }


def _write_causality_report(*, save_dir: Path, layer: int, fold: int, metrics: Dict[str, object]) -> None:
    import json

    report = {"layer": layer, "fold": fold, **metrics}
    out_path = save_dir / f"causality_layer{layer}_fold{fold}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(
        "[causality] "
        f"layer={layer} fold={fold} "
        f"task2_acc_overall={report['task2_acc_overall']} "
        f"task2_acc_task01_correct={report['task2_acc_task01_correct']} "
        f"task2_acc_task01_wrong={report['task2_acc_task01_wrong']} "
        f"task2_acc_task0_correct_task1_wrong={report['task2_acc_task0_correct_task1_wrong']} "
        f"task2_acc_task0_wrong_task1_correct={report['task2_acc_task0_wrong_task1_correct']} "
        f"n_correct={report['n_task01_correct']} "
        f"n_wrong={report['n_task01_wrong']} "
        f"n_cw={report['n_task0_correct_task1_wrong']} "
        f"n_wc={report['n_task0_wrong_task1_correct']}"
    )


def _aggregate_causality_metrics(*, save_dir: Path, layers: List[int], k_folds: int, disable_kfold: bool) -> Dict[str, List[float]]:
    keys = [
        "task2_acc_task01_correct",
        "task2_acc_task01_wrong",
        "task2_acc_task0_correct_task1_wrong",
        "task2_acc_task0_wrong_task1_correct",
    ]
    series = {key: [] for key in keys}
    for layer in layers:
        values_by_key = {key: [] for key in keys}
        folds = [0] if disable_kfold else list(range(k_folds))
        for fold in folds:
            path = save_dir / f"causality_layer{layer}_fold{fold}.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            for key in keys:
                value = data.get(key)
                if value is not None:
                    values_by_key[key].append(float(value))
        for key in keys:
            if values_by_key[key]:
                series[key].append(float(np.mean(values_by_key[key])))
            else:
                series[key].append(float("nan"))
    return series


def _run_epoch(
    *,
    model: HookedTransformer,
    prober: BatchedLinear,
    loader: DataLoader,
    act_name: str,
    class_counts: List[int],
    optimizer: Optional[optim.Optimizer],
    pool: str,
    model_name: str,
) -> Tuple[float, List[float]]:
    criterion = nn.CrossEntropyLoss()
    device = model.cfg.device
    total_loss = 0.0
    total_acc = np.zeros(len(class_counts), dtype=np.float64)
    total = 0

    train_mode = optimizer is not None
    prober.train(train_mode)

    for prompts, labels in tqdm(
        loader,
        desc="train batch" if train_mode else "eval batch",
        leave=False,
    ):
        labels = labels.to(device)
        toks = model.to_tokens(prompts).to(device)
        token_indices = None
        if pool == "fim":
            token_indices = _fim_token_indices(
                toks=toks, tokenizer=model.tokenizer, model_name=model_name
            )
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, names_filter=lambda name: name == act_name)
            act = cache[act_name]
        feats = _pool_activation(act, pool=pool, token_indices=token_indices)
        if feats.dtype != prober.weight.dtype:
            feats = feats.to(prober.weight.dtype)
        logits = prober(feats)

        loss, accs = _task_losses(logits, labels, class_counts, criterion)

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_acc += np.array(accs) * batch_size
        total += batch_size

    if total == 0:
        return 0.0, [0.0 for _ in class_counts]
    return total_loss / total, (total_acc / total).tolist()


def _run_epoch_cached(
    *,
    prober: BatchedLinear,
    loader: DataLoader,
    class_counts: List[int],
    optimizer: Optional[optim.Optimizer],
) -> Tuple[float, List[float]]:
    criterion = nn.CrossEntropyLoss()
    device = prober.weight.device
    total_loss = 0.0
    total_acc = np.zeros(len(class_counts), dtype=np.float64)
    total = 0

    train_mode = optimizer is not None
    prober.train(train_mode)

    for feats, labels in tqdm(
        loader,
        desc="train batch" if train_mode else "eval batch",
        leave=False,
    ):
        labels = labels.to(device)
        feats = feats.to(device)
        if feats.dtype != prober.weight.dtype:
            feats = feats.to(prober.weight.dtype)
        logits = prober(feats)

        loss, accs = _task_losses(logits, labels, class_counts, criterion)

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_acc += np.array(accs) * batch_size
        total += batch_size

    if total == 0:
        return 0.0, [0.0 for _ in class_counts]
    return total_loss / total, (total_acc / total).tolist()


def _k_fold_indices(n: int, k: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    return [fold.astype(int) for fold in np.array_split(indices, k)]


@dataclass
class FoldResult:
    fold: int
    best_val_score: float
    test_acc: List[float]


def _run_kfold(
    *,
    dataset: V2PromptDataset,
    model: HookedTransformer,
    act_name: str,
    class_counts: List[int],
    k_folds: int,
    val_frac: float,
    seed: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    layer: int,
    save_dir: Path,
    causality: bool,
    pool: str,
    model_name: str,
    cache_fingerprint: Optional[Dict[str, object]] = None,
    train_fingerprint: Optional[Dict[str, object]] = None,
) -> List[FoldResult]:
    n_samples = len(dataset)
    if n_samples == 0:
        raise ValueError("Dataset is empty; cannot run k-fold training.")
    if k_folds < 2:
        raise ValueError("k_folds must be at least 2 to run cross-validation.")
    if k_folds > n_samples:
        raise ValueError(
            f"k_folds={k_folds} is larger than dataset size ({n_samples})."
        )

    folds = _k_fold_indices(n_samples, k_folds, seed)
    max_classes = max(class_counts)
    device = model.cfg.device

    results: List[FoldResult] = []
    best_fold_score = -1.0
    best_payload = None
    for fold_idx in range(k_folds):
        test_idx = folds[fold_idx]
        trainval_idx = np.concatenate([f for i, f in enumerate(folds) if i != fold_idx])
        rng = np.random.default_rng(seed + fold_idx)
        rng.shuffle(trainval_idx)

        trainval_len = len(trainval_idx)
        val_size = int(trainval_len * val_frac)
        if val_frac > 0 and trainval_len > 1 and val_size == 0:
            val_size = 1
        if val_size >= trainval_len:
            val_size = max(0, trainval_len - 1)
        val_idx = trainval_idx[:val_size]
        train_idx = trainval_idx[val_size:]
        if len(train_idx) == 0:
            raise ValueError(
                "Training split is empty. Reduce k_folds or val_frac, or use a larger dataset."
            )

        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_batch,
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_batch,
        )
        test_loader = DataLoader(
            Subset(dataset, test_idx),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_batch,
        )

        sample_prompts, _ = next(iter(train_loader))
        toks = model.to_tokens(sample_prompts).to(device)
        token_indices = None
        if pool == "fim":
            token_indices = _fim_token_indices(
                toks=toks, tokenizer=model.tokenizer, model_name=model_name
            )
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, names_filter=lambda name: name == act_name)
            act = cache[act_name]
        in_features = _pool_activation(act, pool=pool, token_indices=token_indices).shape[-1]

        n_tasks = len(class_counts)
        prober = BatchedLinear(n_tasks, in_features, max_classes).to(device)
        optimizer = optim.Adam(prober.parameters(), lr=lr, weight_decay=weight_decay)

        best_val = -1.0
        best_state = None
        for epoch_idx in tqdm(range(epochs), desc=f"fold {fold_idx} epochs", leave=False):
            train_loss, train_acc = _run_epoch(
                model=model,
                prober=prober,
                loader=train_loader,
                act_name=act_name,
                class_counts=class_counts,
                optimizer=optimizer,
                pool=pool,
                model_name=model_name,
            )
            val_loss, val_acc = _run_epoch(
                model=model,
                prober=prober,
                loader=val_loader,
                act_name=act_name,
                class_counts=class_counts,
                optimizer=None,
                pool=pool,
                model_name=model_name,
            )
            train_acc_str = ", ".join(f"{a:.3f}" for a in train_acc)
            val_acc_str = ", ".join(f"{a:.3f}" for a in val_acc)
            tqdm.write(
                f"[fold {fold_idx} epoch {epoch_idx + 1}/{epochs}] "
                f"train_loss={train_loss:.4f} train_acc=[{train_acc_str}] "
                f"val_loss={val_loss:.4f} val_acc=[{val_acc_str}]"
            )
            val_score = sum(val_acc) / len(val_acc)
            if val_score > best_val:
                best_val = val_score
                best_state = {k: v.detach().cpu() for k, v in prober.state_dict().items()}

        if best_state is not None:
            prober.load_state_dict(best_state)

        if causality:
            test_acc, labels_np, preds_np = _eval_predictions(
                model=model,
                prober=prober,
                loader=test_loader,
                act_name=act_name,
                class_counts=class_counts,
                pool=pool,
                model_name=model_name,
            )
            metrics = _causality_metrics(labels_np, preds_np)
            _write_causality_report(
                save_dir=save_dir, layer=layer, fold=fold_idx, metrics=metrics
            )
        else:
            _, test_acc = _run_epoch(
                model=model,
                prober=prober,
                loader=test_loader,
                act_name=act_name,
                class_counts=class_counts,
                optimizer=None,
                pool=pool,
                model_name=model_name,
            )
        results.append(FoldResult(fold=fold_idx, best_val_score=best_val, test_acc=test_acc))
        fold_score = float(np.mean(test_acc)) if test_acc else -1.0
        if best_state is not None and fold_score > best_fold_score:
            best_fold_score = fold_score
            best_payload = {
                "layer": layer,
                "fold": fold_idx,
                "act_name": act_name,
                "class_counts": class_counts,
                "best_val_score": best_val,
                "test_acc": test_acc,
                "state_dict": best_state,
                "cache_fingerprint": cache_fingerprint,
                "train_fingerprint": train_fingerprint,
            }

    if best_payload is not None:
        save_path = save_dir / f"prober_layer{layer}_best.pt"
        torch.save(best_payload, save_path)
    return results


def _run_single_split(
    *,
    dataset: V2PromptDataset,
    model: HookedTransformer,
    act_name: str,
    class_counts: List[int],
    val_frac: float,
    test_frac: float,
    seed: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    layer: int,
    save_dir: Path,
    causality: bool,
    pool: str,
    model_name: str,
    cache_fingerprint: Optional[Dict[str, object]] = None,
    train_fingerprint: Optional[Dict[str, object]] = None,
) -> List[FoldResult]:
    n_samples = len(dataset)
    if n_samples == 0:
        raise ValueError("Dataset is empty; cannot run single-split training.")
    if not (0.0 < test_frac < 1.0):
        raise ValueError("test_frac must be between 0 and 1 for single-split training.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    test_size = int(n_samples * test_frac)
    if test_size == 0:
        test_size = 1
    if test_size >= n_samples:
        test_size = n_samples - 1
    test_idx = indices[:test_size]
    trainval_idx = indices[test_size:]
    rng.shuffle(trainval_idx)

    trainval_len = len(trainval_idx)
    val_size = int(trainval_len * val_frac)
    if val_frac > 0 and trainval_len > 1 and val_size == 0:
        val_size = 1
    if val_size >= trainval_len:
        val_size = max(0, trainval_len - 1)
    val_idx = trainval_idx[:val_size]
    train_idx = trainval_idx[val_size:]
    if len(train_idx) == 0:
        raise ValueError(
            "Training split is empty. Reduce test_frac or val_frac, or use a larger dataset."
        )

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    sample_prompts, _ = next(iter(train_loader))
    device = model.cfg.device
    toks = model.to_tokens(sample_prompts).to(device)
    token_indices = None
    if pool == "fim":
        token_indices = _fim_token_indices(
            toks=toks, tokenizer=model.tokenizer, model_name=model_name
        )
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=lambda name: name == act_name)
        act = cache[act_name]
    in_features = _pool_activation(act, pool=pool, token_indices=token_indices).shape[-1]

    max_classes = max(class_counts)
    n_tasks = len(class_counts)
    prober = BatchedLinear(n_tasks, in_features, max_classes).to(device)
    optimizer = optim.Adam(prober.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = -1.0
    best_state = None
    for epoch_idx in tqdm(range(epochs), desc="single split epochs", leave=False):
        train_loss, train_acc = _run_epoch(
            model=model,
            prober=prober,
            loader=train_loader,
            act_name=act_name,
            class_counts=class_counts,
            optimizer=optimizer,
            pool=pool,
            model_name=model_name,
        )
        val_loss, val_acc = _run_epoch(
            model=model,
            prober=prober,
            loader=val_loader,
            act_name=act_name,
            class_counts=class_counts,
            optimizer=None,
            pool=pool,
            model_name=model_name,
        )
        train_acc_str = ", ".join(f"{a:.3f}" for a in train_acc)
        val_acc_str = ", ".join(f"{a:.3f}" for a in val_acc)
        tqdm.write(
            f"[single split epoch {epoch_idx + 1}/{epochs}] "
            f"train_loss={train_loss:.4f} train_acc=[{train_acc_str}] "
            f"val_loss={val_loss:.4f} val_acc=[{val_acc_str}]"
        )
        val_score = sum(val_acc) / len(val_acc)
        if val_score > best_val:
            best_val = val_score
            best_state = {k: v.detach().cpu() for k, v in prober.state_dict().items()}

    if best_state is not None:
        prober.load_state_dict(best_state)

    if causality:
        test_acc, labels_np, preds_np = _eval_predictions(
            model=model,
            prober=prober,
            loader=test_loader,
            act_name=act_name,
            class_counts=class_counts,
            pool=pool,
            model_name=model_name,
        )
        metrics = _causality_metrics(labels_np, preds_np)
        _write_causality_report(save_dir=save_dir, layer=layer, fold=0, metrics=metrics)
    else:
        _, test_acc = _run_epoch(
            model=model,
            prober=prober,
            loader=test_loader,
            act_name=act_name,
            class_counts=class_counts,
            optimizer=None,
            pool=pool,
            model_name=model_name,
        )
    results = [FoldResult(fold=0, best_val_score=best_val, test_acc=test_acc)]

    if best_state is not None:
        save_path = save_dir / f"prober_layer{layer}_best.pt"
        payload = {
            "layer": layer,
            "fold": 0,
            "act_name": act_name,
            "class_counts": class_counts,
            "best_val_score": best_val,
            "test_acc": test_acc,
            "state_dict": best_state,
            "cache_fingerprint": cache_fingerprint,
            "train_fingerprint": train_fingerprint,
        }
        torch.save(payload, save_path)
    return results


def _run_kfold_cached(
    *,
    dataset: CachedActivationDataset,
    class_counts: List[int],
    k_folds: int,
    val_frac: float,
    seed: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    layer: int,
    save_dir: Path,
    causality: bool,
    cache_fingerprint: Optional[Dict[str, object]] = None,
    train_fingerprint: Optional[Dict[str, object]] = None,
) -> List[FoldResult]:
    n_samples = len(dataset)
    if n_samples == 0:
        raise ValueError("Cached dataset is empty; cannot run k-fold training.")
    if k_folds < 2:
        raise ValueError("k_folds must be at least 2 to run cross-validation.")
    if k_folds > n_samples:
        raise ValueError(
            f"k_folds={k_folds} is larger than dataset size ({n_samples})."
        )

    folds = _k_fold_indices(n_samples, k_folds, seed)
    max_classes = max(class_counts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_features = dataset.in_features

    results: List[FoldResult] = []
    best_fold_score = -1.0
    best_payload = None
    for fold_idx in range(k_folds):
        test_idx = folds[fold_idx]
        trainval_idx = np.concatenate([f for i, f in enumerate(folds) if i != fold_idx])
        rng = np.random.default_rng(seed + fold_idx)
        rng.shuffle(trainval_idx)

        trainval_len = len(trainval_idx)
        val_size = int(trainval_len * val_frac)
        if val_frac > 0 and trainval_len > 1 and val_size == 0:
            val_size = 1
        if val_size >= trainval_len:
            val_size = max(0, trainval_len - 1)
        val_idx = trainval_idx[:val_size]
        train_idx = trainval_idx[val_size:]
        if len(train_idx) == 0:
            raise ValueError(
                "Training split is empty. Reduce k_folds or val_frac, or use a larger dataset."
            )

        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_batch_cached,
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_batch_cached,
        )
        test_loader = DataLoader(
            Subset(dataset, test_idx),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_batch_cached,
        )

        n_tasks = len(class_counts)
        prober = BatchedLinear(n_tasks, in_features, max_classes).to(device)
        optimizer = optim.Adam(prober.parameters(), lr=lr, weight_decay=weight_decay)

        best_val = -1.0
        best_state = None
        for epoch_idx in tqdm(range(epochs), desc=f"fold {fold_idx} epochs", leave=False):
            train_loss, train_acc = _run_epoch_cached(
                prober=prober,
                loader=train_loader,
                class_counts=class_counts,
                optimizer=optimizer,
            )
            val_loss, val_acc = _run_epoch_cached(
                prober=prober,
                loader=val_loader,
                class_counts=class_counts,
                optimizer=None,
            )
            train_acc_str = ", ".join(f"{a:.3f}" for a in train_acc)
            val_acc_str = ", ".join(f"{a:.3f}" for a in val_acc)
            tqdm.write(
                f"[fold {fold_idx} epoch {epoch_idx + 1}/{epochs}] "
                f"train_loss={train_loss:.4f} train_acc=[{train_acc_str}] "
                f"val_loss={val_loss:.4f} val_acc=[{val_acc_str}]"
            )
            val_score = sum(val_acc) / len(val_acc)
            if val_score > best_val:
                best_val = val_score
                best_state = {k: v.detach().cpu() for k, v in prober.state_dict().items()}

        if best_state is not None:
            prober.load_state_dict(best_state)

        if causality:
            test_acc, labels_np, preds_np = _eval_predictions_cached(
                prober=prober,
                loader=test_loader,
                class_counts=class_counts,
            )
            metrics = _causality_metrics(labels_np, preds_np)
            _write_causality_report(
                save_dir=save_dir, layer=layer, fold=fold_idx, metrics=metrics
            )
        else:
            _, test_acc = _run_epoch_cached(
                prober=prober,
                loader=test_loader,
                class_counts=class_counts,
                optimizer=None,
            )
        results.append(FoldResult(fold=fold_idx, best_val_score=best_val, test_acc=test_acc))
        fold_score = float(np.mean(test_acc)) if test_acc else -1.0
        if best_state is not None and fold_score > best_fold_score:
            best_fold_score = fold_score
            best_payload = {
                "layer": layer,
                "class_counts": class_counts,
                "best_val_score": best_val,
                "test_acc": test_acc,
                "state_dict": best_state,
                "cache_fingerprint": cache_fingerprint,
                "train_fingerprint": train_fingerprint,
            }

    if best_payload is not None:
        save_path = save_dir / f"prober_layer{layer}_best.pt"
        torch.save(best_payload, save_path)
    return results


def _run_single_split_cached(
    *,
    dataset: CachedActivationDataset,
    class_counts: List[int],
    val_frac: float,
    test_frac: float,
    seed: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    layer: int,
    save_dir: Path,
    causality: bool,
    cache_fingerprint: Optional[Dict[str, object]] = None,
    train_fingerprint: Optional[Dict[str, object]] = None,
) -> List[FoldResult]:
    n_samples = len(dataset)
    if n_samples == 0:
        raise ValueError("Cached dataset is empty; cannot run single-split training.")
    if not (0.0 < test_frac < 1.0):
        raise ValueError("test_frac must be between 0 and 1 for single-split training.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    test_size = int(n_samples * test_frac)
    if test_size == 0:
        test_size = 1
    if test_size >= n_samples:
        test_size = n_samples - 1
    test_idx = indices[:test_size]
    trainval_idx = indices[test_size:]
    rng.shuffle(trainval_idx)

    trainval_len = len(trainval_idx)
    val_size = int(trainval_len * val_frac)
    if val_frac > 0 and trainval_len > 1 and val_size == 0:
        val_size = 1
    if val_size >= trainval_len:
        val_size = max(0, trainval_len - 1)
    val_idx = trainval_idx[:val_size]
    train_idx = trainval_idx[val_size:]
    if len(train_idx) == 0:
        raise ValueError(
            "Training split is empty. Reduce test_frac or val_frac, or use a larger dataset."
        )

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch_cached,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch_cached,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch_cached,
    )

    max_classes = max(class_counts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_tasks = len(class_counts)
    prober = BatchedLinear(n_tasks, dataset.in_features, max_classes).to(device)
    optimizer = optim.Adam(prober.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = -1.0
    best_state = None
    for epoch_idx in tqdm(range(epochs), desc="single split epochs", leave=False):
        train_loss, train_acc = _run_epoch_cached(
            prober=prober,
            loader=train_loader,
            class_counts=class_counts,
            optimizer=optimizer,
        )
        val_loss, val_acc = _run_epoch_cached(
            prober=prober,
            loader=val_loader,
            class_counts=class_counts,
            optimizer=None,
        )
        train_acc_str = ", ".join(f"{a:.3f}" for a in train_acc)
        val_acc_str = ", ".join(f"{a:.3f}" for a in val_acc)
        tqdm.write(
            f"[single split epoch {epoch_idx + 1}/{epochs}] "
            f"train_loss={train_loss:.4f} train_acc=[{train_acc_str}] "
            f"val_loss={val_loss:.4f} val_acc=[{val_acc_str}]"
        )
        val_score = sum(val_acc) / len(val_acc)
        if val_score > best_val:
            best_val = val_score
            best_state = {k: v.detach().cpu() for k, v in prober.state_dict().items()}

    if best_state is not None:
        prober.load_state_dict(best_state)

    if causality:
        test_acc, labels_np, preds_np = _eval_predictions_cached(
            prober=prober,
            loader=test_loader,
            class_counts=class_counts,
        )
        metrics = _causality_metrics(labels_np, preds_np)
        _write_causality_report(save_dir=save_dir, layer=layer, fold=0, metrics=metrics)
    else:
        _, test_acc = _run_epoch_cached(
            prober=prober,
            loader=test_loader,
            class_counts=class_counts,
            optimizer=None,
        )
    results = [FoldResult(fold=0, best_val_score=best_val, test_acc=test_acc)]

    if best_state is not None:
        save_path = save_dir / f"prober_layer{layer}_best.pt"
        payload = {
            "layer": layer,
            "class_counts": class_counts,
            "best_val_score": best_val,
            "test_acc": test_acc,
            "state_dict": best_state,
            "cache_fingerprint": cache_fingerprint,
            "train_fingerprint": train_fingerprint,
        }
        torch.save(payload, save_path)
    return results
