#!/bin/bash
#SBATCH --job-name=ltp_12_canine_mlm
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=18:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# ============================================================================
# Cell 12 — CANINE continued char-MLM on Wiki+OLDI cap-rule (DIALECTS ONLY, NATIVE).
# Char sequences are heavy: smaller batch (8) × grad_accum 8.
# Training: 3 epochs × ~600k sentences ≈ 8-12h GPU. Eval ~45 min.
# ============================================================================

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

nvidia-smi || true
df -h "$HOME" || true
echo ""
date
python -u analysis/canine/experiments/canine_finetuned_wikiOLDI_dialects_native/run.py
status=$?
date
echo "exit status: $status"
exit $status
