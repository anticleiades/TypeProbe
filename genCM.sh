#!/bin/bash

# Define the working directory from the first passed parameter.
# (Note: If no parameter is passed, this will evaluate to an empty string)
WORKDIR="${1:-paperData}"

# Create required directories
mkdir -p "${WORKDIR}/interference"
mkdir -p "${WORKDIR}/transfer"

# ---------------------------------------------------------
# SantaCoder - Task 1
# ---------------------------------------------------------
python scripts/plot_cm_compare.py \
  --standard "${WORKDIR}/results/santacoder/prober_java_test_java/cm_raw_layer13_task1.npz" \
  --comparison "${WORKDIR}/results/santacoder/prober_java_test_adv_java/cm_raw_layer13_task1.npz" \
  --out "${WORKDIR}/interference/santacoder_java_task1.pdf" \
  --mode side_by_side --title-left "Standard" --title-right "Adversarial" \
  --num-classes 8 --class-order 0 1 2 6 3 4 5 7

python scripts/plot_cm_compare.py \
  --standard "${WORKDIR}/results/santacoder/prober_pyTag_test_pyTag/cm_raw_layer11_task1.npz" \
  --comparison "${WORKDIR}/results/santacoder/prober_pyTag_test_adv_pyTag/cm_raw_layer11_task1.npz" \
  --out "${WORKDIR}/interference/santacoder_pyTag_task1.pdf" \
  --mode side_by_side --title-left "Standard" --title-right "Adversarial" \
  --num-classes 8 --class-order 0 1 2 6 3 4 5 7

python scripts/plot_cm_compare.py \
  --standard "${WORKDIR}/results/santacoder/prober_pyUnt_test_pyUnt/cm_raw_layer11_task1.npz" \
  --comparison "${WORKDIR}/results/santacoder/prober_pyUnt_test_adv_pyUnt/cm_raw_layer11_task1.npz" \
  --out "${WORKDIR}/interference/santacoder_pyUnt_task1.pdf" \
  --mode side_by_side --title-left "Standard" --title-right "Adversarial" \
  --num-classes 8 --class-order 0 1 2 6 3 4 5 7

# ---------------------------------------------------------
# SantaCoder - Task 2
# ---------------------------------------------------------
python scripts/plot_cm_compare.py \
  --standard "${WORKDIR}/results/santacoder/prober_java_test_java/cm_raw_layer12_task2.npz" \
  --comparison "${WORKDIR}/results/santacoder/prober_java_test_adv_java/cm_raw_layer12_task2.npz" \
  --out "${WORKDIR}/interference/santacoder_java_task2.pdf" \
  --mode side_by_side --title-left "Standard" --title-right "Adversarial" \
  --num-classes 8 --class-order 0 1 2 6 3 4 5 7

python scripts/plot_cm_compare.py \
  --standard "${WORKDIR}/results/santacoder/prober_pyTag_test_pyTag/cm_raw_layer12_task2.npz" \
  --comparison "${WORKDIR}/results/santacoder/prober_pyTag_test_adv_pyTag/cm_raw_layer12_task2.npz" \
  --out "${WORKDIR}/interference/santacoder_pyTag_task2.pdf" \
  --mode side_by_side --title-left "Standard" --title-right "Adversarial" \
  --num-classes 8 --class-order 0 1 2 6 3 4 5 7

python scripts/plot_cm_compare.py \
  --standard "${WORKDIR}/results/santacoder/prober_pyUnt_test_pyUnt/cm_raw_layer12_task2.npz" \
  --comparison "${WORKDIR}/results/santacoder/prober_pyUnt_test_adv_pyUnt/cm_raw_layer12_task2.npz" \
  --out "${WORKDIR}/interference/santacoder_pyUnt_task2.pdf" \
  --mode side_by_side --title-left "Standard" --title-right "Adversarial" \
  --num-classes 8 --class-order 0 1 2 6 3 4 5 7

# ---------------------------------------------------------
# CodeLlama (7B) - Task 1
# ---------------------------------------------------------
python scripts/plot_cm_compare.py \
  --standard "${WORKDIR}/results/codellama-7b-hf/prober_java_test_java/cm_raw_layer16_task1.npz" \
  --comparison "${WORKDIR}/results/codellama-7b-hf/prober_java_test_adv_java/cm_raw_layer16_task1.npz" \
  --out "${WORKDIR}/interference/codellama_java_task1.pdf" \
  --mode side_by_side --title-left "Standard" --title-right "Adversarial" \
  --num-classes 8 --class-order 0 1 2 6 3 4 5 7

python scripts/plot_cm_compare.py \
  --standard "${WORKDIR}/results/codellama-7b-hf/prober_pyTag_test_pyTag/cm_raw_layer16_task1.npz" \
  --comparison "${WORKDIR}/results/codellama-7b-hf/prober_pyTag_test_adv_pyTag/cm_raw_layer16_task1.npz" \
  --out "${WORKDIR}/interference/codellama_pyTag_task1.pdf" \
  --mode side_by_side --title-left "Standard" --title-right "Adversarial" \
  --num-classes 8 --class-order 0 1 2 6 3 4 5 7

python scripts/plot_cm_compare.py \
  --standard "${WORKDIR}/results/codellama-7b-hf/prober_pyUnt_test_pyUnt/cm_raw_layer16_task1.npz" \
  --comparison "${WORKDIR}/results/codellama-7b-hf/prober_pyUnt_test_adv_pyUnt/cm_raw_layer16_task1.npz" \
  --out "${WORKDIR}/interference/codellama_pyUnt_task1.pdf" \
  --mode side_by_side --title-left "Standard" --title-right "Adversarial" \
  --num-classes 8 --class-order 0 1 2 6 3 4 5 7

# ---------------------------------------------------------
# CodeLlama (7B) - Task 2
# ---------------------------------------------------------
python scripts/plot_cm_compare.py \
  --standard "${WORKDIR}/results/codellama-7b-hf/prober_java_test_java/cm_raw_layer14_task2.npz" \
  --comparison "${WORKDIR}/results/codellama-7b-hf/prober_java_test_adv_java/cm_raw_layer14_task2.npz" \
  --out "${WORKDIR}/interference/codellama_java_task2.pdf" \
  --mode side_by_side --title-left "Standard" --title-right "Adversarial" \
  --num-classes 8 --class-order 0 1 2 6 3 4 5 7

python scripts/plot_cm_compare.py \
  --standard "${WORKDIR}/results/codellama-7b-hf/prober_pyTag_test_pyTag/cm_raw_layer17_task2.npz" \
  --comparison "${WORKDIR}/results/codellama-7b-hf/prober_pyTag_test_adv_pyTag/cm_raw_layer17_task2.npz" \
  --out "${WORKDIR}/interference/codellama_pyTag_task2.pdf" \
  --mode side_by_side --title-left "Standard" --title-right "Adversarial" \
  --num-classes 8 --class-order 0 1 2 6 3 4 5 7

python scripts/plot_cm_compare.py \
  --standard "${WORKDIR}/results/codellama-7b-hf/prober_pyUnt_test_pyUnt/cm_raw_layer18_task2.npz" \
  --comparison "${WORKDIR}/results/codellama-7b-hf/prober_pyUnt_test_adv_pyUnt/cm_raw_layer18_task2.npz" \
  --out "${WORKDIR}/interference/codellama_pyUnt_task2.pdf" \
  --mode side_by_side --title-left "Standard" --title-right "Adversarial" \
  --num-classes 8 --class-order 0 1 2 6 3 4 5 7

# ---------------------------------------------------------
# Transfer Comparisons (SantaCoder vs CodeLlama; Task 2)
# ---------------------------------------------------------
python scripts/plot_cm_compare.py \
  --standard "${WORKDIR}/results/santacoder/prober_pyUnt_test_pyUnt/cm_raw_layer12_task2.npz" \
  --comparison "${WORKDIR}/results/codellama-7b-hf/prober_pyUnt_test_pyUnt/cm_raw_layer18_task2.npz" \
  --out "${WORKDIR}/transfer/task2ComparisonBaselinePyUnt.pdf" \
  --mode side_by_side --title-left "SantaCoder (1.1B)" --title-right "CodeLlama (7B)" \
  --num-classes 8 --class-order 0 1 2 6 3 4 5 7

python scripts/plot_cm_compare.py \
  --standard "${WORKDIR}/results/santacoder/prober_pyUnt_test_java/cm_raw_layer12_task2.npz" \
  --comparison "${WORKDIR}/results/codellama-7b-hf/prober_pyUnt_test_java/cm_raw_layer18_task2.npz" \
  --out "${WORKDIR}/transfer/task2ComparisonTransferFromPyUnt.pdf" \
  --mode side_by_side --title-left "SantaCoder (1.1B)" --title-right "CodeLlama (7B)" \
  --num-classes 8 --class-order 0 1 2 6 3 4 5 7