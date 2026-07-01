from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from prober_plots import plot_cm
from prober_data import (
    CachedActivationDataset,
    collate_batch_cached,
)


class BatchedLinear(nn.Module):
    def __init__(
        self, n_tasks: int, in_features: int, out_features: int, normalize: bool
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_tasks, out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(n_tasks, out_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        self.normalize = normalize
        if normalize:
            self.register_buffer("feature_mean", torch.zeros(in_features))
            self.register_buffer("feature_std", torch.ones(in_features))

    def set_normalization_stats(self, dataset, device, train_loader):
        if not self.normalize:
            print("Skipping normalization")
            return
        print("Calculating feature-wise Z-score normalization stats on training set...")
        sum_feats = torch.zeros(dataset.in_features, device=device, dtype=torch.float64)
        sq_sum_feats = torch.zeros(
            dataset.in_features, device=device, dtype=torch.float64
        )
        n_train_samples = 0

        with torch.no_grad():
            for feats, _ in train_loader:
                feats = feats.to(device).double()
                sum_feats += feats.sum(dim=0)
                sq_sum_feats += (feats**2).sum(dim=0)
                n_train_samples += feats.shape[0]

        train_mu = (sum_feats / n_train_samples).to(torch.float32)
        train_var = (sq_sum_feats / n_train_samples) - (train_mu.double() ** 2)
        train_sigma = torch.sqrt(train_var.clamp(min=1e-8)).to(torch.float32)
        self.feature_mean.copy_(train_mu.squeeze())
        self.feature_std.copy_(train_sigma.squeeze())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            x_norm = (x - self.feature_mean) / (self.feature_std + 1e-8)
        else:
            x_norm = x
        return torch.einsum("bi,toi->tbo", x_norm, self.weight) + self.bias[:, None, :]


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
                total_acc[task_idx] += (
                    (pred == labels[:, task_idx]).float().sum().item()
                )

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


def _run_epoch_cached(
    *,
    prober: BatchedLinear,
    loader: DataLoader,
    class_counts: List[int],
    optimizer: Optional[optim.Optimizer],
    scheduler,
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
            if scheduler is not None:
                scheduler.step()

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
    val_acc: List[float]
    test_acc: List[float]


def fold_impl(
    *,
    args,
    test_idx,
    results,
    layer,
    val_frac,
    dataset,
    batch_size,
    fold_idx,
    trainval_idx,
    rng,
    normalize: bool,
    class_counts,
    epochs: int,
    lr: float,
    weight_decay: float,
    cache_fingerprint: Optional[Dict[str, object]],
    train_fingerprint: Optional[Dict[str, object]],
    best_payload,
    best_fold_score,
    sched: bool,
    optim_impl,
):
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
    n_tasks = len(class_counts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prober = BatchedLinear(n_tasks, dataset.in_features, max_classes, normalize).to(
        device
    )
    prober.set_normalization_stats(dataset, device, train_loader)
    optimizer = optim_impl(prober.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if sched:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_per_task = np.full(n_tasks, -1.0)
    best_state = {k: v.detach().cpu().clone() for k, v in prober.state_dict().items()}
    val_acc_glob = [0.0] * n_tasks
    for epoch_idx in tqdm(range(epochs), desc=f"fold {fold_idx} epochs", leave=False):
        train_loss, train_acc = _run_epoch_cached(
            prober=prober,
            loader=train_loader,
            class_counts=class_counts,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        val_loss, val_acc = _run_epoch_cached(
            prober=prober,
            loader=val_loader,
            class_counts=class_counts,
            optimizer=None,
            scheduler=None,
        )

        train_acc_str = ", ".join(f"{a:.3f}" for a in train_acc)
        val_acc_str = ", ".join(f"{a:.3f}" for a in val_acc)
        tqdm.write(
            f"[fold {fold_idx} epoch {epoch_idx + 1}/{epochs}] "
            f"train_loss={train_loss:.4f} train_acc=[{train_acc_str}] "
            f"val_loss={val_loss:.4f} val_acc=[{val_acc_str}]"
        )

        current_state = prober.state_dict()
        for t_idx in range(n_tasks):
            if val_acc[t_idx] > best_val_per_task[t_idx]:
                best_val_per_task[t_idx] = val_acc[t_idx]
                val_acc_glob[t_idx] = val_acc[t_idx]

                for k in best_state.keys():
                    if "weight" in k or "bias" in k:
                        best_state[k][t_idx].copy_(
                            current_state[k][t_idx].detach().cpu()
                        )

    prober.load_state_dict(best_state)
    #_, test_acc = _run_epoch_cached(
     #   prober=prober,
      #  loader=test_loader,
       # class_counts=class_counts,
        #optimizer=None,
        #scheduler=None,
    #)
    test_acc, labels_np, preds_np = _eval_predictions_cached(
        prober=prober,
        loader=test_loader,
        class_counts=class_counts,
    )
    best_val_mean = float(np.mean(best_val_per_task))
    plot_cm(args, class_counts, labels_np, layer, preds_np)
    results.append(
        FoldResult(fold=fold_idx, best_val_score=best_val_mean, test_acc=test_acc,  val_acc=best_val_per_task.tolist())
    )
    fold_score = float(np.mean(test_acc)) if test_acc else -1.0
    if best_state is not None and best_val_mean > best_fold_score:
        best_fold_score = best_val_mean
        best_payload = {
            "layer": layer,
            "normalize": normalize,
            "class_counts": class_counts,
            "best_val_score_per_task": best_val_per_task.tolist(),
            "best_val_score": best_val_mean,
            "test_acc": test_acc,
            "state_dict": best_state,
            "cache_fingerprint": cache_fingerprint,
            "train_fingerprint": train_fingerprint,
        }
    return best_fold_score, best_payload


def train_probeV2(
    *,
    args,
    dataset: CachedActivationDataset,
    class_counts: List[int],
    val_frac: float,
    k_folds: int,
    test_frac: float,
    seed: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    layer: int,
    save_dir: Path,
    cache_fingerprint: Optional[Dict[str, object]] = None,
    train_fingerprint: Optional[Dict[str, object]] = None,
    optim_impl,
    normalize: bool,
    sched: bool,
) -> List[FoldResult]:
    n_samples = len(dataset)
    if n_samples == 0:
        raise ValueError("Cached dataset is empty; cannot run k-fold training.")
    results: List[FoldResult] = []
    disable_kfolds = k_folds == -1
    if not disable_kfolds:
        if k_folds < 2:
            raise ValueError("k_folds must be at least 2 to run cross-validation.")
        if k_folds > n_samples:
            raise ValueError(
                f"k_folds={k_folds} is larger than dataset size ({n_samples})."
            )
        folds = _k_fold_indices(n_samples, k_folds, seed)
        best_fold_score = -1.0
        best_payload = None
        for fold_idx in range(k_folds):
            test_idx = folds[fold_idx]
            trainval_idx = np.concatenate(
                [f for i, f in enumerate(folds) if i != fold_idx]
            )
            rng = np.random.default_rng(seed + fold_idx)
            best_fold_score, best_payload = fold_impl(
                args=args,
                test_idx=test_idx,
                results=results,
                layer=layer,
                val_frac=val_frac,
                dataset=dataset,
                batch_size=batch_size,
                fold_idx=fold_idx,
                trainval_idx=trainval_idx,
                rng=rng,
                normalize=normalize,
                class_counts=class_counts,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                cache_fingerprint=cache_fingerprint,
                train_fingerprint=train_fingerprint,
                best_payload=best_payload,
                best_fold_score=best_fold_score,
                sched=sched,
                optim_impl=optim_impl,
            )
        best_state = best_payload
    else:
        if not (0.0 < test_frac < 1.0):
            raise ValueError(
                "test_frac must be between 0 and 1 for single-split training."
            )
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_samples)
        test_size = int(n_samples * test_frac)
        if test_size == 0:
            test_size = 1
        if test_size >= n_samples:
            test_size = n_samples - 1
        test_idx = indices[:test_size]
        trainval_idx = indices[test_size:]
        _, best_state = fold_impl(
            args=args,
            test_idx=test_idx,
            results=results,
            layer=layer,
            val_frac=val_frac,
            dataset=dataset,
            batch_size=batch_size,
            fold_idx=0,
            trainval_idx=trainval_idx,
            rng=rng,
            normalize=normalize,
            class_counts=class_counts,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            cache_fingerprint=cache_fingerprint,
            train_fingerprint=train_fingerprint,
            best_payload=None,
            best_fold_score=-1.0,
            sched=sched,
            optim_impl=optim_impl,
        )
    save_path = save_dir / f"prober_layer{layer}_best.pt"
    torch.save(best_state, save_path)
    return results
