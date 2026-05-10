#!/bin/bash
#SBATCH --job-name=ltp_11_xlmr_mlm
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# ============================================================================
# Cell 11 — XLM-R continued MLM on Wiki+OLDI cap-rule (DIALECTS ONLY, NATIVE).
# Training: 3 epochs × ~600k sentences (6 dialects × 100k) ≈ 4-6h GPU.
# Eval ~45 min.
# ============================================================================

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

nvidia-smi || true
df -h "$HOME" || true
echo ""
date
python -u analysis/multilingual_xlmr/experiments/xlmr_finetuned_wikiOLDI_dialects_native/run.py
status=$?
date
echo "exit status: $status"
exit $status
