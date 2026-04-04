#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
from pathlib import Path
import time


def run_cross_evaluations(args, model_name: str, act: str):
    work_dir_root = args.work_dir
    datasets_dir = args.datasets_dir
    dry_run = args.dry_run
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

    slurm_log_root = Path("slurmV4") / "crossEval"
    slurm_log_root.mkdir(parents=True, exist_ok=True)

    job_counter = 0
    skipped_selftest = 0
    skipped_done = 0

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
                job_counter, skipped_selftest, skipped_done = _run_cross_eval_for_probe(
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
                    slurm_log_root=slurm_log_root,
                    job_counter=job_counter,
                    skipped_selftest=skipped_selftest,
                    skipped_done=skipped_done,
                    gpu_partition=args.gpu_partition
                )

            control_prober_name = f"control_{lang}"
            control_dir = root / "probers" / control_prober_name / "results"
            if control_dir.exists():
                job_counter, skipped_selftest, skipped_done = _run_cross_eval_for_probe(
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
                    slurm_log_root=slurm_log_root,
                    job_counter=job_counter,
                    skipped_selftest=skipped_selftest,
                    skipped_done=skipped_done,
                    gpu_partition=args.gpu_partition
                )

    print(f"Totale job sottomessi: {job_counter}")
    print(f"Self-test skippati      : {skipped_selftest}")
    print(f"Già completati (results): {skipped_done}")


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
        slurm_log_root: Path,
        job_counter: int,
        skipped_selftest: int,
        skipped_done: int,
        gpu_partition: str
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
            skipped_selftest += 1
            continue

        if control:
            eval_tag = f"prober_control_{train_tag}_test_control_{test_tag}"
        else:
            eval_tag = f"prober_{train_tag}_test_{test_tag}"

        result_json = root / "eval" / eval_tag / "results.json"
        if result_json.exists():
            print(f"[skip] Already done: {result_json}")
            skipped_done += 1
            continue

        print(
            f"\n>>> [{model_slug}] {act} | Train tag: {train_tag} | "
            f"Eval: {prober_label} on {test_tag}"
        )

        cmd = [
            sys.executable,
            "prober/prober.py",
            "--work-dir",
            str(work_dir_root),
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
            "bigcode/santacoder"
            if "santacoder" in model_slug
            else "codellama/CodeLlama-7b-hf",
        ]
        if use_java_flag:
            cmd.append("--java")

        cmd_str = " ".join(cmd)

        job_name = "typeProbe"
        slurm_out = slurm_log_root / f"{job_name}-%j.out"
        slurm_err = slurm_log_root / f"{job_name}-%j.err"

        slurm_script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={slurm_out}
#SBATCH --error={slurm_err}
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition={gpu_partition}
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

conda activate TypeProbe
exec {cmd_str}
"""

        eval_dir_path = root / "eval" / eval_tag
        eval_dir_path.mkdir(parents=True, exist_ok=True)
        slurm_script_path = eval_dir_path / "submit.sh"
        with open(slurm_script_path, "w") as f:
            f.write(slurm_script_content)

        if dry_run:
            print("[DRY-RUN] Command:", cmd_str)
            print("[DRY-RUN] Slurm script path:", slurm_script_path)
        else:
            job_counter += 1
            print(f"Submitting job #{job_counter}: {job_name} ({eval_tag})")
            subprocess.run(["sbatch", str(slurm_script_path)], check=True)
            time.sleep(1)

    return job_counter, skipped_selftest, skipped_done


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--datasets-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--gpu-partition", required=True, type=str, help="GPU partition to use. Refer to your cluster documentation.")
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
                args,
                model,
                act,
            )
