#!/bin/bash
#SBATCH --job-name=ltp_07_labse_zeroshot
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# ============================================================================
# Cell 7 — LaBSE zero-shot, eval on FLORES (NATIVE). No training. ~30 min GPU.
# ============================================================================

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

nvidia-smi || true
date
python -u analysis/labse/experiments/labse_zeroshot_native/run.py
status=$?
date
echo "exit status: $status"
exit $status
