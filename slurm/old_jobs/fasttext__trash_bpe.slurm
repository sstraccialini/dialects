#!/bin/bash
#SBATCH --job-name=ltp_fasttext_trash_bpe
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# BPE + TF-IDF (sub-pipeline storica, da buttare) — kept as a sanity baseline.
# Self-contained: nothing else in the project imports from this folder.

set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

python -u analysis/fasttext/experiments/trash/run.py
