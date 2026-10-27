#!/bin/bash
#SBATCH --job-name=video-compression
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
# Supply the Gipdeep partition/account/QOS with sbatch flags as required locally.
set -euo pipefail

PROJECT_ROOT="${VIDEO_COMPRESSION_ROOT:-$HOME/video-compression}"
cd "$PROJECT_ROOT"
# Initialize Conda in the submitting shell before sbatch if conda is not on PATH.
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# CUDA_VISIBLE_DEVICES is managed by SLURM. cuda:0 is the first allocated GPU.
conda run --no-capture-output -n turbo_ddcm python -m video_compression evaluate \
    --config "${1:-configs/baseline.json}" \
    --paths "${2:-configs/paths.local.json}" \
    --name "${3:-baseline-${SLURM_JOB_ID}}"
