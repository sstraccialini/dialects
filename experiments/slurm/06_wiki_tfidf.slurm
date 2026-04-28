#!/bin/bash
#SBATCH --job-name=ltp_06_wiki_tfidf
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"

python -u analysis_wiki/tfidf_baseline/src/run_baseline.py
