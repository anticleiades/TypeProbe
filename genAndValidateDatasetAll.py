import argparse
import os
import subprocess
import shutil
import sys
from pathlib import Path
from collections import defaultdict


def generateDataset(policy: int, tag_type: str, seed: int, description: str):
    print(f"--- {description}. Seed={seed} ---")
    cmd = [
        sys.executable, "dataset/genV2.py",
        "--identifierPolicyAll", str(policy),
        tag_type,
        "--seed", str(seed)
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")

    subprocess.run(cmd, env=env, check=True)


def fixDatasetAll(validations, env):
    clean_out_dir = Path("cleanOut")
    clean_out_dir.mkdir(exist_ok=True)

    # Group all bad-indices files by dataset path and take the UNION per dataset
    dataset_groups = defaultdict(list)
    for _, _, ds_path, bad_path in validations:
        dataset_groups[Path(ds_path)].append(Path(bad_path))

    for ds_file, bad_files in dataset_groups.items():
        target_path = clean_out_dir / ds_file.name
        merged_indices = set()

        for bf in bad_files:
            if bf.exists():
                with bf.open("r") as f:
                    # all non-empty lines are considered row indices
                    merged_indices.update(line.strip() for line in f if line.strip())

        if merged_indices:
            merged_bad_file = ds_file.parent / f"merged_bad_{ds_file.stem}.txt"
            with merged_bad_file.open("w") as f:
                f.write("\n".join(sorted(merged_indices, key=int)))

            print(f"Cleaning {ds_file.name}: {len(merged_indices)} total bad indices (merged).")
            subprocess.run(
                [
                    sys.executable,
                    "scripts/cleanup_dataset.py",
                    "--dataset",
                    str(ds_file),
                    "--output",
                    str(target_path),
                    "--bad-indices",
                    str(merged_bad_file),
                ],
                env=env,
                check=True,
            )
        else:
            print(f"No bad examples for {ds_file.name}. Copying directly...")
            shutil.copy2(ds_file, target_path)

        # Always propagate metadata alongside the cleaned / copied dataset
        meta_src = ds_file.with_suffix(".meta.parquet")
        if not meta_src.exists():
            raise FileNotFoundError(f"Missing metadata file for {ds_file}: {meta_src}")
        meta_target = clean_out_dir / meta_src.name
        shutil.copy2(meta_src, meta_target)

    print("\nDone! All files merged and cleaned in 'cleanOut/'.")

def finalValidation(env, ap):
    validations = [
        ("Adv_Py_Tag", "scripts/check_python_prompts.py", "cleanOut/adv_tagged.parquet"),
        ("Adv_Py_UnTag", "scripts/check_python_prompts.py", "cleanOut/adv_untagged.parquet"),
        ("Py_Tag", "scripts/check_python_prompts.py", "cleanOut/tagged.parquet"),
        ("Py_UnTag", "scripts/check_python_prompts.py", "cleanOut/untagged.parquet"),
        ("Adv_Java", "scripts/check_java_prompts.py", "cleanOut/adv_tagged.parquet"),
        ("Java", "scripts/check_java_prompts.py", "cleanOut/tagged.parquet"),
    ]

    print("\nFinal validation...")
    for label, script, dataset in validations:
        print(f"--- Validating {label} in {dataset} ---")
        subprocess.run([
            sys.executable, script,
            "--dataset", dataset,
            "--chunk-size",
            str(ap.chunk_size),
            "--workers",
            str(ap.workers)
        ], env=env, check=True)


def validateDatasetAll(ap):
    validations = [
        ("Adv_Java", "scripts/check_java_prompts.py", "dataset/outV2/adv_tagged.parquet", "dataset/outV2/cleanMeta/bad_Adv_Java.txt"),
        ("Java", "scripts/check_java_prompts.py", "dataset/outV2/tagged.parquet", "dataset/outV2/cleanMeta/bad_Java.txt"),
        ("Adv_Py_Tag", "scripts/check_python_prompts.py", "dataset/outV2/adv_tagged.parquet", "dataset/outV2/cleanMeta/bad_Adv_Py_Tag.txt"),
        ("Adv_Py_UnTag", "scripts/check_python_prompts.py", "dataset/outV2/adv_untagged.parquet", "dataset/outV2/cleanMeta/bad_Adv_Py_UnTag.txt"),
        ("Py_Tag", "scripts/check_python_prompts.py", "dataset/outV2/tagged.parquet", "dataset/outV2/cleanMeta/bad_Py_Tag.txt"),
        ("Py_UnTag", "scripts/check_python_prompts.py", "dataset/outV2/untagged.parquet", "dataset/outV2/cleanMeta/bad_Py_UnTag.txt"),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")

    for label, script, dataset, out_bad in validations:
        print(f"--- Validating {label} ---")
        Path(out_bad).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            sys.executable, script,
            "--dataset", dataset,
            "--chunk-size",
            str(ap.chunk_size),
            "--workers",
            str(ap.workers),
            "--out-bad-indices", out_bad
        ], env=env, check=True)

    fixDatasetAll(validations, env)
    finalValidation(env, ap)




if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--validation-only",
        action="store_true",
        help="Skip generation, run validation only on existing datasets.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1)),
        help="[Validation] Number of parallel workers.",
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=4,
        help="[Validation] Number of samples to compile per worker batch.",
    )
    runs = [
        (1, "--noTypeTag", 2020, "Random identifier (NO type tags)"),
        (1, "--onlyTypeTag", 2003, "Random identifier (with type tags)"),
        (3, "--onlyTypeTag", 42, "Adversarial (with type tags)"),
        (3, "--noTypeTag", 666, "Adversarial (NO type tags)")
    ]
    args = ap.parse_args()
    if not args.validation_only:
        for p, t, s, desc in runs:
            generateDataset(p, t, s, desc)
    else:
        print(f"Validation only mode. Ensure to have already generated the datasets")

    validateDatasetAll(args)
