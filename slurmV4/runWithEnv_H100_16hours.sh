#!/bin/bash
#SBATCH --job-name=typeProbe
#SBATCH --output=slurmV4/%x-%j.out
#SBATCH --time=16:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1


conda activate TypeProbe
exec "$@"
