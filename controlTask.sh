#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${1:-}"
DATASET_PREFIX="${2:-cleanOut}"

if [[ -z "$WORKDIR" ]]; then
  echo "ERROR: WORKDIR (arg1) is required" >&2
  exit 1
fi

if [[ ! -d "$WORKDIR" ]]; then
  echo "ERROR: WORKDIR '$WORKDIR' does not exist or is not a directory" >&2
  exit 1
fi

echo "Using WORKDIR=$WORKDIR"
echo "Using DATASET_PREFIX=$DATASET_PREFIX"

# SANTACODER - Py tagged
sbatch --job-name="Ctl_SPyTag_L0-23" slurmV4/runWithEnv_H100_5hours.sh \
  python prober/prober.py \
    --work-dir "$WORKDIR" \
    --dataset "$DATASET_PREFIX/tagged.parquet" \
    --act resid_post \
    --use-cache --control-task \
    --layer-start 0 \
    --layer-end 23 \
    --model "bigcode/santacoder" \
    --batch-size 2048 \
    --epochs 30 \
    --weight-decay 0.1 \
    --pool fim \
    --repro \
    --model-revision main

# SANTACODER - Py untagged
sbatch --job-name="Ctl_SPyUnTag_L0-23" slurmV4/runWithEnv_H100_5hours.sh \
  python prober/prober.py \
    --work-dir "$WORKDIR" \
    --dataset cleanOut/untagged.parquet \
    --act resid_post \
    --use-cache --control-task \
    --layer-start 0 \
    --layer-end 23 \
    --model "bigcode/santacoder" \
    --batch-size 2048 \
    --epochs 30 \
    --weight-decay 0.1 \
    --pool fim \
    --repro \
    --model-revision main

# SANTACODER - Java tagged
sbatch --job-name="Ctl_SJava_L0-23" slurmV4/runWithEnv_H100_5hours.sh \
  python prober/prober.py \
    --work-dir "$WORKDIR" \
    --dataset "$DATASET_PREFIX/tagged.parquet" \
    --act resid_post \
    --use-cache --control-task \
    --layer-start 0 \
    --layer-end 23 \
    --model "bigcode/santacoder" \
    --batch-size 2048 \
    --epochs 30 \
    --weight-decay 0.1 \
    --java \
    --pool fim \
    --repro \
    --model-revision main

# SANTACODER - Adv Py tagged
sbatch --job-name="Ctl_Adv_SPyTag_L0-23" slurmV4/runWithEnv_H100_5hours.sh \
  python prober/prober.py \
    --work-dir "$WORKDIR" \
    --dataset "$DATASET_PREFIX/adv_tagged.parquet" \
    --act resid_post \
    --use-cache --control-task \
    --layer-start 0 \
    --layer-end 23 \
    --model "bigcode/santacoder" \
    --batch-size 2048 \
    --epochs 30 \
    --weight-decay 0.1 \
    --pool fim \
    --repro \
    --model-revision main

# SANTACODER - Adv Py untagged
sbatch --job-name="Ctl_Adv_SPyUnTag_L0-23" slurmV4/runWithEnv_H100_5hours.sh \
  python prober/prober.py \
    --work-dir "$WORKDIR" \
    --dataset "$DATASET_PREFIX/adv_untagged.parquet" \
    --act resid_post \
    --use-cache --control-task \
    --layer-start 0 \
    --layer-end 23 \
    --model "bigcode/santacoder" \
    --batch-size 2048 \
    --epochs 30 \
    --weight-decay 0.1 \
    --pool fim \
    --repro \
    --model-revision main

# SANTACODER - Adv Java tagged
sbatch --job-name="Ctl_Adv_SJava_L0-23" slurmV4/runWithEnv_H100_5hours.sh \
  python prober/prober.py \
    --work-dir "$WORKDIR" \
    --dataset "$DATASET_PREFIX/adv_tagged.parquet" \
    --act resid_post \
    --use-cache --control-task \
    --layer-start 0 \
    --layer-end 23 \
    --model "bigcode/santacoder" \
    --batch-size 2048 \
    --epochs 30 \
    --weight-decay 0.1 \
    --java \
    --pool fim \
    --repro \
    --model-revision main


# CODELLAMA - Py tagged
sbatch --job-name="Ctl_CPyTag_L0-31" slurmV4/runWithEnv_H100_5hours.sh \
  python prober/prober.py \
    --work-dir "$WORKDIR" \
    --dataset "$DATASET_PREFIX/tagged.parquet" \
    --act resid_post \
    --use-cache --control-task \
    --layer-start 0 \
    --layer-end 31 \
    --model "codellama/CodeLlama-7b-hf" \
    --batch-size 2048 \
    --epochs 30 \
    --weight-decay 0.1 \
    --pool fim \
    --repro \
    --model-revision main

# CODELLAMA - Py untagged
sbatch --job-name="Ctl_CPyUnTag_L0-31" slurmV4/runWithEnv_H100_5hours.sh \
  python prober/prober.py \
    --work-dir "$WORKDIR" \
    --dataset cleanOut/untagged.parquet \
    --act resid_post \
    --use-cache --control-task \
    --layer-start 0 \
    --layer-end 31 \
    --model "codellama/CodeLlama-7b-hf" \
    --batch-size 2048 \
    --epochs 30 \
    --weight-decay 0.1 \
    --pool fim \
    --repro \
    --model-revision main

# CODELLAMA - Java tagged
sbatch --job-name="Ctl_CJava_L0-31" slurmV4/runWithEnv_H100_5hours.sh \
  python prober/prober.py \
    --work-dir "$WORKDIR" \
    --dataset "$DATASET_PREFIX/tagged.parquet" \
    --act resid_post \
    --use-cache --control-task \
    --layer-start 0 \
    --layer-end 31 \
    --model "codellama/CodeLlama-7b-hf" \
    --batch-size 2048 \
    --epochs 30 \
    --weight-decay 0.1 \
    --java \
    --pool fim \
    --repro \
    --model-revision main

# CODELLAMA - Adv Py tagged
sbatch --job-name="Ctl_Adv_CPyTag_L0-31" slurmV4/runWithEnv_H100_5hours.sh \
  python prober/prober.py \
    --work-dir "$WORKDIR" \
    --dataset "$DATASET_PREFIX/adv_tagged.parquet" \
    --act resid_post \
    --use-cache --control-task \
    --layer-start 0 \
    --layer-end 31 \
    --model "codellama/CodeLlama-7b-hf" \
    --batch-size 2048 \
    --epochs 30 \
    --weight-decay 0.1 \
    --pool fim \
    --repro \
    --model-revision main

# CODELLAMA - Adv Py untagged
sbatch --job-name="Ctl_Adv_CPyUnTag_L0-31" slurmV4/runWithEnv_H100_5hours.sh \
  python prober/prober.py \
    --work-dir "$WORKDIR" \
    --dataset "$DATASET_PREFIX/adv_untagged.parquet" \
    --act resid_post \
    --use-cache --control-task \
    --layer-start 0 \
    --layer-end 31 \
    --model "codellama/CodeLlama-7b-hf" \
    --batch-size 2048 \
    --epochs 30 \
    --weight-decay 0.1 \
    --pool fim \
    --repro \
    --model-revision main

# CODELLAMA - Adv Java tagged
sbatch --job-name="Ctl_Adv_CJava_L0-31" slurmV4/runWithEnv_H100_5hours.sh \
  python prober/prober.py \
    --work-dir "$WORKDIR" \
    --dataset "$DATASET_PREFIX/adv_tagged.parquet" \
    --act resid_post \
    --use-cache --control-task \
    --layer-start 0 \
    --layer-end 31 \
    --model "codellama/CodeLlama-7b-hf" \
    --batch-size 2048 \
    --epochs 30 \
    --weight-decay 0.1 \
    --pool fim \
    --repro \
    --model-revision main

