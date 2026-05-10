#!/bin/bash
#SBATCH --job-name=ltp_xlmr_rehearsal_norm
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# ==============================================================================
# ABLATION CELL — XLM-R + NORMALIZED text + 13-var rehearsal (cap 50k/var).
# Completes the 2x2 ablation matrix (text x training) for XLM-R.
# Expected runtime: 2.5-3.5h training + 30 min eval = ~3.5h.
# Output: analysis/multilingual_xlmr/experiments/mlm_wiki_rehearsal_normalized/
# ==============================================================================

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

nvidia-smi || true
df -h "$HOME" || true
echo ""
date
python -u analysis/multilingual_xlmr/experiments/mlm_wiki_rehearsal_normalized/run.py
status=$?
date
echo "exit status: $status"
exit $status
