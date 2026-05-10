#!/bin/bash
#SBATCH --job-name=ltp_canine_mlm_dialects_norm
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# ===========================================================================
# Ablation: CANINE MLM con OLD setup (normalized + dialect-only).
# Per verificare se la regressione del run principale (silhouette_romance
# da +0.60 a +0.22) è dovuta al rehearsal training su 13 varietà o al
# passaggio a not_normalized text.
#
# Setup:
#   - text variant: normalized (lowercase ASCII)
#   - training corpus: Wiki, 6 dialetti soltanto, cap 100k
#   - evaluation:    FLORES + OLDI (centroid + parallel) su tutte 13
#
# Output dedicato in:
#   analysis/canine/experiments/mlm_wiki_dialects_normalized/
#
# Compute stimato: ~7-9h training + 30 min eval ≈ ~8-10h totali.
# ===========================================================================

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

nvidia-smi || true
df -h "$HOME" || true
echo ""
date
python -u analysis/canine/experiments/mlm_wiki_dialects_normalized/run.py
status=$?
date
echo "exit status: $status"
exit $status
