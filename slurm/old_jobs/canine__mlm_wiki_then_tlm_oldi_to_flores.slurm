#!/bin/bash
#SBATCH --job-name=ltp_canine_mlm_wiki_then_tlm_oldi
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# CANINE (char-level) — sequential MLM on Wiki (stage 1) → sentence-pair MLM
# on OLDI (stage 2), then eval on FLORES+ (centroid + parallel).
# Char-level sequenze sono pesanti; allocazione generosa.
# See analysis/canine/experiments/mlm_wiki_then_tlm_oldi_to_flores/run.py.

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

nvidia-smi || true
echo ""

python -u analysis/canine/experiments/mlm_wiki_then_tlm_oldi_to_flores/run.py
