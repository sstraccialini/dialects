#!/bin/bash
#SBATCH --job-name=ltp_canine_dial_native
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# ==============================================================================
# ABLATION CELL — CANINE + NATIVE text + dialect-only training (cap 50k/var).
# Completes the 2x2 ablation matrix (text x training) for CANINE.
# Expected runtime: 1.5-2h training + 30 min eval = ~2-2.5h.
# Output: analysis/canine/experiments/mlm_wiki_dialects_native/
# ==============================================================================

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

nvidia-smi || true
df -h "$HOME" || true
echo ""
date
python -u analysis/canine/experiments/mlm_wiki_dialects_native/run.py
status=$?
date
echo "exit status: $status"
exit $status
