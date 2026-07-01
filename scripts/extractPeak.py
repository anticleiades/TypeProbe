#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import torch
import json

MODEL_MAP = {
    "santacoder": "bigcode/santacoder",
    "codellama-7b-hf": "codellama/CodeLlama-7b-hf",
}


def extract_peaks(work_dir_root: Path, output_json: Path):
    models = ["bigcode/santacoder", "codellama/CodeLlama-7b-hf"]
    acts = ["resid_post"]

    train_tag_map = {
        ("tagged", "py"): "pyTag",
        ("untagged", "py"): "pyUnt",
        ("tagged", "java"): "java",
        ("adv_tagged", "py"): "adv_pyTag",
        ("adv_untagged", "py"): "adv_pyUnt",
        ("adv_tagged", "java"): "adv_java",
    }
    train_datasets = ["tagged", "untagged", "adv_tagged", "adv_untagged"]

    peak_layers = {"task0": {}, "task1": {}, "task2": {}}

    for act in acts:
        act_tag = act.replace("-", "_").lower()
        for model_name in models:
            model_slug = model_name.split("/")[-1].lower().replace(" ", "-")

            for task in ["task0", "task1", "task2"]:
                if model_slug not in peak_layers[task]:
                    peak_layers[task][model_slug] = {}

            for train_ds in train_datasets:
                root = work_dir_root / f"{model_slug}_{act_tag}_{train_ds}"
                if not root.exists():
                    continue

                for lang in ("py", "java"):
                    train_tag = train_tag_map.get((train_ds, lang))
                    if train_tag is None:
                        continue

                    prober_dir = root / "probers" / lang / "results"
                    _extract_from_dir(prober_dir, peak_layers, model_slug, train_tag)

                    control_dir = root / "probers" / f"control_{lang}" / "results"
                    _extract_from_dir(
                        control_dir, peak_layers, model_slug, f"control_{train_tag}"
                    )

    with open(output_json, "w") as f:
        json.dump(peak_layers, f, indent=2)
    print(f"[OK]: {output_json}")


def _extract_from_dir(
    prober_dir: Path, peak_layers: dict, model_slug: str, domain_key: str
):
    if not prober_dir.exists():
        return


    pt_files = list(prober_dir.glob("*.pt"))
    if not pt_files:
        return

    layer_scores = []
    for pt_file in pt_files:
        try:
            payload = torch.load(pt_file, map_location="cpu", weights_only=False)
            layer = payload.get("layer")
            val_scores = payload.get("best_val_score_per_task")
            if layer is not None and val_scores is not None:
                layer_scores.append((layer, val_scores))
        except Exception as e:
            print(f"[WARN] Impossibile leggere {pt_file}: {e}")

    if not layer_scores:
        return

    num_tasks = len(layer_scores[0][1])
    for task_idx in range(min(num_tasks, 3)):
        task_key = f"task{task_idx}"

        best_layer = None
        best_score = -1.0

        for layer, scores in layer_scores:
            if task_idx < len(scores):
                score = scores[task_idx]
                if score > best_score:
                    best_score = score
                    best_layer = layer

        if best_layer is not None:
            peak_layers[task_key][model_slug][domain_key] = best_layer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("peak_layers.json"))
    args = parser.parse_args()
    extract_peaks(args.work_dir, args.out)
