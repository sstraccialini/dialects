#!/bin/bash
#SBATCH --job-name=ltp_10_wiki_sentence_baseline
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"

nvidia-smi || true
python -u analysis_wiki/sentence_baseline/src/run_sentence_baseline.py
