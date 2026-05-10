#!/bin/bash
#SBATCH --job-name=ltp_minilm_mnrl_oldi_to_flores
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# Sentence-MiniLM — MNRL contrastive on OLDI italian↔dialect pairs only,
# then eval on FLORES+ (centroid + parallel).
# See analysis/sentence_minilm/experiments/mnrl_oldi_to_flores/run.py.

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

nvidia-smi || true
echo ""

python -u analysis/sentence_minilm/experiments/mnrl_oldi_to_flores/run.py
