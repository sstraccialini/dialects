#!/bin/bash
#SBATCH --job-name=ltp_15_flores_multilingual_adapted
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"

nvidia-smi || true
python -u analysis_flores/multilingual_adapted/src/run_pipeline.py
