#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

DEFAULT_LABELS = ["int", "str", "bool", "l[int]", "l[str]", "l[bool]", "float", "l[float]"]
DEFAULT_CLASS_ORDER = [0, 1, 2, 6, 3, 4, 5, 7]


def row_normalize(cm: np.ndarray) -> np.ndarray:
    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return cm / row_sums


def load_preds(npz_path: Path):
    data = np.load(npz_path)
    return data["y_true"], data["y_pred"]


def infer_classes(y_true_a, y_pred_a, y_true_b, y_pred_b, num_classes=None):
    if num_classes is not None:
        return np.arange(num_classes)
    max_label = int(max(
        np.max(y_true_a), np.max(y_pred_a), np.max(y_true_b), np.max(y_pred_b)
    ))
    return np.arange(max_label + 1)


def reorder_classes(classes, class_order=None):
    classes = list(classes)
    if class_order is None:
        if len(classes) == 8 and set(classes) == set(range(8)):
            class_order = DEFAULT_CLASS_ORDER
        else:
            return np.array(classes)
    ordered = [c for c in class_order if c in classes]
    remaining = [c for c in classes if c not in ordered]
    return np.array(ordered + remaining)


def pick_labels(classes, labels):
    if labels is None:
        labels = DEFAULT_LABELS
    labels = list(labels)
    if len(labels) == len(DEFAULT_LABELS) and set(classes).issubset(set(range(len(labels)))):
        return [labels[c] for c in classes]
    if len(labels) < len(classes):
        raise ValueError(f"Need at least {len(classes)} labels, got {len(labels)}.")
    return labels[:len(classes)]


def build_normalized_cm(y_true, y_pred, classes):
    return row_normalize(confusion_matrix(y_true, y_pred, labels=classes))


def plot_compare(
    standard_npz: Path,
    comparison_npz: Path,
    output_path: Path,
    labels=None,
    num_classes=None,
    class_order=None,
    mode="delta",
    title_left="Standard",
    title_right=None,
    std_cmap="Blues",
    delta_cmap="RdBu_r",
    comparison_cmap="Blues",
    delta_absmax=None,
    figsize=(8.6, 3.8),
    dpi=300,
    verbose=False,
):
    y_true_std, y_pred_std = load_preds(standard_npz)
    y_true_cmp, y_pred_cmp = load_preds(comparison_npz)

    classes = infer_classes(y_true_std, y_pred_std, y_true_cmp, y_pred_cmp, num_classes=num_classes)
    classes = reorder_classes(classes, class_order=class_order)
    display_labels = pick_labels(classes, labels)

    cm_std_norm = build_normalized_cm(y_true_std, y_pred_std, classes)
    cm_cmp_norm = build_normalized_cm(y_true_cmp, y_pred_cmp, classes)

    if mode == "delta":
        right_matrix = cm_cmp_norm - cm_std_norm
        if title_right is None:
            title_right = "Comparison - Standard"
        if delta_absmax is None:
            delta_absmax = float(np.max(np.abs(right_matrix)))
            delta_absmax = max(delta_absmax, 0.15)
        right_kwargs = dict(cmap=delta_cmap, vmin=-delta_absmax, vmax=delta_absmax)
        right_cbar_label = "Δ"
    elif mode == "side_by_side":
        right_matrix = cm_cmp_norm
        if title_right is None:
            title_right = "Comparison"
        right_kwargs = dict(cmap=comparison_cmap, vmin=0, vmax=1)
        right_cbar_label = ""
    else:
        raise ValueError("mode must be 'delta' or 'side_by_side'")

    if verbose:
        print(f"standard n={len(y_true_std)}, comparison n={len(y_true_cmp)}")
        print(f"classes={classes.tolist()}")
        print(f"mode={mode}")
        print(f"std observed true classes={np.unique(y_true_std).tolist()}")
        print(f"cmp observed true classes={np.unique(y_true_cmp).tolist()}")

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    im0 = axes[0].imshow(cm_std_norm, cmap=std_cmap, vmin=0, vmax=1)
    axes[0].set_title(title_left, fontsize=11)
    axes[1].set_title(title_right, fontsize=11)

    for ax in axes:
        ax.set_xticks(range(len(display_labels)))
        ax.set_yticks(range(len(display_labels)))
        ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(display_labels, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(length=0)

    im1 = axes[1].imshow(right_matrix, **right_kwargs)

    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.ax.tick_params(labelsize=8)
    # cbar0.set_label("Std", fontsize=8)

    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=8)
    cbar1.set_label(right_cbar_label, fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    if output_path.suffix.lower() != ".png":
        fig.savefig(output_path.with_suffix('.png'), bbox_inches="tight", dpi=dpi)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot standard vs comparison confusion matrices from saved NPZ files.")
    parser.add_argument("--standard", type=Path, required=True, help="Path to baseline NPZ with y_true and y_pred")
    parser.add_argument("--comparison", type=Path, required=True, help="Path to comparison NPZ with y_true and y_pred")
    parser.add_argument("--out", type=Path, required=True, help="Output figure path (.pdf or .png)")
    parser.add_argument("--mode", choices=["delta", "side_by_side"], default="delta", help="delta: comparison-standard; side_by_side: raw normalized matrices")
    parser.add_argument("--title-left", default="Standard", help="Left panel title")
    parser.add_argument("--title-right", default=None, help="Right panel title")
    parser.add_argument("--labels", nargs="+", default=None, help="Display labels in class index order")
    parser.add_argument("--num-classes", type=int, default=None, help="Force number of classes")
    parser.add_argument("--class-order", nargs="+", type=int, default=None, help="Optional class index order, e.g. 0 1 2 6 3 4 5 7")
    parser.add_argument("--delta-absmax", type=float, default=None, help="Color scale limit for delta heatmap")
    parser.add_argument("--figwidth", type=float, default=8.6)
    parser.add_argument("--figheight", type=float, default=3.8)
    parser.add_argument("--verbose", action="store_true", help="Print dataset sizes and observed classes")
    args = parser.parse_args()

    plot_compare(
        standard_npz=args.standard,
        comparison_npz=args.comparison,
        output_path=args.out,
        labels=args.labels,
        num_classes=args.num_classes,
        class_order=args.class_order,
        mode=args.mode,
        title_left=args.title_left,
        title_right=args.title_right,
        delta_absmax=args.delta_absmax,
        figsize=(args.figwidth, args.figheight),
        verbose=args.verbose,
    )
