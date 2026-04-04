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

########## SANTACODER ##########

# SANTACODER - Py tagged
sbatch --job-name="SPyTag" slurmV4/runWithEnv_H100_5hours.sh \
python prober/prober.py \
  --work-dir "$WORKDIR" \
  --dataset "$DATASET_PREFIX/tagged.parquet" \
  --act resid_post \
  --cache-acts \
  --model "bigcode/santacoder" \
  --batch-size 256 \
  --pool fim \
  --repro \
  --model-revision main

# SANTACODER - Py untagged
sbatch --job-name="SPyUnTag" slurmV4/runWithEnv_H100_5hours.sh \
python prober/prober.py \
  --work-dir "$WORKDIR" \
  --dataset "$DATASET_PREFIX/untagged.parquet" \
  --act resid_post \
  --cache-acts \
  --model "bigcode/santacoder" \
  --batch-size 256 \
  --pool fim \
  --repro \
  --model-revision main

# SANTACODER - Java tagged
sbatch --job-name="SJava" slurmV4/runWithEnv_H100_5hours.sh \
python prober/prober.py \
  --work-dir "$WORKDIR" \
  --dataset "$DATASET_PREFIX/tagged.parquet" \
  --act resid_post \
  --cache-acts \
  --model "bigcode/santacoder" \
  --batch-size 256 \
  --java \
  --pool fim \
  --repro \
  --model-revision main

# SANTACODER - Adv Py tagged
sbatch --job-name="Adv_SPyTag" slurmV4/runWithEnv_H100_5hours.sh \
python prober/prober.py \
  --work-dir "$WORKDIR" \
  --dataset "$DATASET_PREFIX/adv_tagged.parquet" \
  --act resid_post \
  --cache-acts \
  --model "bigcode/santacoder" \
  --batch-size 256 \
  --pool fim \
  --repro \
  --model-revision main

# SANTACODER - Adv Py untagged
sbatch --job-name="Adv_SPyUnTag" slurmV4/runWithEnv_H100_5hours.sh \
python prober/prober.py \
  --work-dir "$WORKDIR" \
  --dataset "$DATASET_PREFIX/adv_untagged.parquet" \
  --act resid_post \
  --cache-acts \
  --model "bigcode/santacoder" \
  --batch-size 256 \
  --pool fim \
  --repro \
  --model-revision main

# SANTACODER - Adv Java tagged
sbatch --job-name="Adv_SJava" slurmV4/runWithEnv_H100_5hours.sh \
python prober/prober.py \
  --work-dir "$WORKDIR" \
  --dataset "$DATASET_PREFIX/adv_tagged.parquet" \
  --act resid_post \
  --cache-acts \
  --model "bigcode/santacoder" \
  --batch-size 256 \
  --java \
  --pool fim \
  --repro \
  --model-revision main

########## CODELLAMA ##########

# CODELLAMA - Py tagged
sbatch --job-name="CPyTag" slurmV4/runWithEnv_H100_16hours.sh \
python prober/prober.py \
  --work-dir "$WORKDIR" \
  --dataset "$DATASET_PREFIX/tagged.parquet" \
  --act resid_post \
  --cache-acts \
  --model "codellama/CodeLlama-7b-hf" \
  --batch-size 256 \
  --pool fim \
  --repro \
  --model-revision main

# CODELLAMA - Py untagged
sbatch --job-name="CPyUnTag" slurmV4/runWithEnv_H100_16hours.sh \
python prober/prober.py \
  --work-dir "$WORKDIR" \
  --dataset "$DATASET_PREFIX/untagged.parquet" \
  --act resid_post \
  --cache-acts \
  --model "codellama/CodeLlama-7b-hf" \
  --batch-size 256 \
  --pool fim \
  --repro \
  --model-revision main

# CODELLAMA - Java tagged
sbatch --job-name="CJava" slurmV4/runWithEnv_H100_16hours.sh \
python prober/prober.py \
  --work-dir "$WORKDIR" \
  --dataset "$DATASET_PREFIX/tagged.parquet" \
  --act resid_post \
  --cache-acts \
  --model "codellama/CodeLlama-7b-hf" \
  --batch-size 256 \
  --java \
  --pool fim \
  --repro \
  --model-revision main

# CODELLAMA - Adv Py tagged
sbatch --job-name="Adv_CPyTag" slurmV4/runWithEnv_H100_16hours.sh \
python prober/prober.py \
  --work-dir "$WORKDIR" \
  --dataset "$DATASET_PREFIX/adv_tagged.parquet" \
  --act resid_post \
  --cache-acts \
  --model "codellama/CodeLlama-7b-hf" \
  --batch-size 256 \
  --pool fim \
  --repro \
  --model-revision main

# CODELLAMA - Adv Py untagged
sbatch --job-name="Adv_CPyUnTag" slurmV4/runWithEnv_H100_16hours.sh \
python prober/prober.py \
  --work-dir "$WORKDIR" \
  --dataset "$DATASET_PREFIX/adv_untagged.parquet" \
  --act resid_post \
  --cache-acts \
  --model "codellama/CodeLlama-7b-hf" \
  --batch-size 256 \
  --pool fim \
  --repro \
  --model-revision main

# CODELLAMA - Adv Java tagged
sbatch --job-name="Adv_CJava" slurmV4/runWithEnv_H100_16hours.sh \
python prober/prober.py \
  --work-dir "$WORKDIR" \
  --dataset "$DATASET_PREFIX/adv_tagged.parquet" \
  --act resid_post \
  --cache-acts \
  --model "codellama/CodeLlama-7b-hf" \
  --batch-size 256 \
  --java \
  --pool fim \
  --repro \
  --model-revision main
