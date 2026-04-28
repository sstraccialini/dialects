#!/bin/bash
#SBATCH --job-name=ltp_wiki_multilingual
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# Standalone re-run of step 09 (wiki multilingual XLM-R) after fixing the
# umap-learn missing dependency.

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

nvidia-smi || true
echo ""

python -u analysis/multilingual_xlmr/wiki/src/run_pipeline.py
