#!/bin/bash
#SBATCH --job-name=ltp_labse_mnrl_oldi
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# ==============================================================================
# LaBSE COUNTER-EXAMPLE — MNRL contrastive on OLDI parallel pairs (ita<->dial).
# Atteso: silhouette_romance crolla (le traduzioni collassano nello stesso punto).
# Lezione metodologica: contrastive task fine-tuning DISTRUGGE la similarità
# linguistica che vogliamo misurare.
# Expected runtime: 1-2h training + 30 min eval = ~2-2.5h.
# Output: analysis/labse/experiments/mnrl_oldi_to_flores/
# ==============================================================================

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

# Compute nodes have no internet. Force HF offline mode.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

nvidia-smi || true
df -h "$HOME" || true
echo ""
date
python -u analysis/labse/experiments/mnrl_oldi_to_flores/run.py
status=$?
date
echo "exit status: $status"
exit $status
