import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parents[1]
V2_DIR = ROOT_DIR / "dataset"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from metadata import class_to_label_type  # noqa: E402


def plot_cm(args, class_counts_eval, labels_np, layer,  preds_np):
    for task_idx in range(len(class_counts_eval)):
        y_true = labels_np[:, task_idx]
        y_pred = preds_np[:, task_idx]

        raw_data_path = args.eval_out_dir / f"cm_raw_layer{layer}_task{task_idx}.npz"
        np.savez_compressed(raw_data_path, y_true=y_true, y_pred=y_pred)

        cm = confusion_matrix(y_true, y_pred)
        unique_classes = np.unique(np.concatenate((y_true, y_pred)))
        if task_idx == 0:
            class_names = ["First Function", "Second Function"]
        else:
            class_names = [
                class_to_label_type(c, is_minimal=False) for c in unique_classes
            ]

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title(f"Confusion Matrix: Layer {layer}, Task {task_idx}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()

        cm_plot_path = args.eval_out_dir / f"cm_layer{layer}_task{task_idx}.pdf"
        plt.savefig(cm_plot_path)
        plt.close()
