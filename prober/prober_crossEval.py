#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
from pathlib import Path

MODEL_MAP = {
    "santacoder": "bigcode/santacoder",
    "codellama-7b-hf": "codellama/CodeLlama-7b-hf",
}

def get_model_from_slug(slug: str) -> str:
    try:
        return MODEL_MAP.get(slug)
    except KeyError:
        raise ValueError(f"Unknown model slug: {slug}")

def run_cross_evaluations(
    work_dir_root: Path,
    model_name: str,
    act: str,
    datasets_dir: Path,
    dry_run: bool = False,
):
    model_slug = model_name.split("/")[-1].lower().replace(" ", "-")
    act_tag = act.replace(" ", "_").lower()

    train_tag_map = {
        ("tagged", "py"): "pyTag",
        ("untagged", "py"): "pyUnt",
        ("tagged", "java"): "java",
        ("adv_tagged", "py"): "adv_pyTag",
        ("adv_untagged", "py"): "adv_pyUnt",
        ("adv_tagged", "java"): "adv_java",
    }

    train_datasets = ["tagged", "untagged", "adv_tagged", "adv_untagged"]

    test_configs = {
        "pyTag": ("py", Path("tagged.parquet"), False),
        "pyUnt": ("py", Path("untagged.parquet"), False),
        "java": ("java", Path("tagged.parquet"), True),
        "adv_pyTag": ("py", Path("adv_tagged.parquet"), False),
        "adv_pyUnt": ("py", Path("adv_untagged.parquet"), False),
        "adv_java": ("java", Path("adv_tagged.parquet"), True),
    }

    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")

    for train_ds in train_datasets:
        root = work_dir_root / f"{model_slug}_{act_tag}_{train_ds}"
        if not root.exists():
            continue

        for lang in ("py", "java"):
            train_tag = train_tag_map.get((train_ds, lang))
            if train_tag is None:
                continue

            prober_name = lang
            prober_dir = root / "probers" / prober_name / "results"
            if prober_dir.exists():
                _run_cross_eval_for_probe(
                    work_dir_root=work_dir_root,
                    model_slug=model_slug,
                    act=act,
                    root=root,
                    prober_dir=prober_dir,
                    train_tag=train_tag,
                    lang=lang,
                    test_configs=test_configs,
                    datasets_dir=datasets_dir,
                    control=False,
                    env=env,
                    dry_run=dry_run,
                )

            control_prober_name = f"control_{lang}"
            control_dir = root / "probers" / control_prober_name / "results"
            if control_dir.exists():
                _run_cross_eval_for_probe(
                    work_dir_root=work_dir_root,
                    model_slug=model_slug,
                    act=act,
                    root=root,
                    prober_dir=control_dir,
                    train_tag=train_tag,
                    lang=lang,
                    test_configs=test_configs,
                    datasets_dir=datasets_dir,
                    control=True,
                    env=env,
                    dry_run=dry_run,
                )


def _run_cross_eval_for_probe(
    *,
    work_dir_root: Path,
    model_slug: str,
    act: str,
    root: Path,
    prober_dir: Path,
    train_tag: str,
    lang: str,
    test_configs: dict,
    datasets_dir: Path,
    control: bool,
    env: dict,
    dry_run: bool,
):
    prober_label = f"{'control_' if control else ''}{lang}"

    for test_tag, (test_lang, rel_path, use_java_flag) in test_configs.items():
        ds_path = datasets_dir / rel_path
        if not ds_path.exists():
            print(f"[WARN] Missing dataset for test_tag={test_tag}: {ds_path}")
            continue

        if (not control) and (test_tag == train_tag):
            print(
                f"[skip] Self-test (already done by trainer): "
                f"train_tag={train_tag}, prober={prober_label}, test={test_tag}"
            )
            continue

        if control:
            eval_tag = f"prober_control_{train_tag}_test_control_{test_tag}"
        else:
            eval_tag = f"prober_{train_tag}_test_{test_tag}"

        result_json = root / "eval" / eval_tag / "results.json"
        if result_json.exists():
            print(f"[skip] Already done: {result_json}")
            continue

        print(
            f"\n>>> [{model_slug}] {act} | Train tag: {train_tag} | "
            f"Eval: {prober_label} on {test_tag}"
        )

        cmd = [
            sys.executable,
            "prober/prober.py",
            "--work-dir",
            str(work_dir_root),  # ROOT GLOBALE, NON root
            "--dataset",
            str(ds_path),
            "--act",
            act,
            "--use-cache",
            "--eval-only",
            "--eval-dir",
            str(prober_dir),
            "--eval-tag",
            eval_tag,
            "--model",
            get_model_from_slug(slug=model_slug),
        ]
        if use_java_flag:
            cmd.append("--java")

        if dry_run:
            print("[DRY-RUN] Command:", " ".join(cmd))
        else:
            subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--datasets-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for path in [args.work_dir, args.datasets_dir]:
        if not path.exists():
            print(f"Error: Path {path} does not exist.")
            sys.exit(1)

    models = ["bigcode/santacoder", "codellama/CodeLlama-7b-hf"]
    acts = ["resid_post"]

    for act in acts:
        for model in models:
            run_cross_evaluations(
                args.work_dir,
                model,
                act,
                args.datasets_dir,
                dry_run=args.dry_run,
            )

