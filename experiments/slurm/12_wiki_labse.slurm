#!/bin/bash
#SBATCH --job-name=ltp_12_wiki_labse
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"

nvidia-smi || true
python -u analysis_wiki/labse/src/run_sentence_baseline.py
