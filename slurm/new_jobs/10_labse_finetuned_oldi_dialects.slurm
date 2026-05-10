#!/bin/bash
#SBATCH --job-name=ltp_10_labse_mnrl
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# ============================================================================
# Cell 10 — LaBSE MNRL contrastive on OLDI ita↔dialect pairs (counter-example).
# Training: 5 epochs × 31k pairs ≈ 2-3h GPU; eval ~30 min.
# ============================================================================

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

nvidia-smi || true
date
python -u analysis/labse/experiments/labse_finetuned_oldi_dialects_native/run.py
status=$?
date
echo "exit status: $status"
exit $status
