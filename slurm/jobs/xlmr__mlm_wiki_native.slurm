#!/bin/bash
#SBATCH --job-name=ltp_xlmr_mlm_wiki_native
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=23:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# ===========================================================================
# XLM-R continual MLM pretraining on Wiki not_normalized,
# all 13 varieties (rehearsal-style) — eval on FLORES + OLDI.
#
# Setup:
#   - text variant: not_normalized
#   - training corpus: Wiki, 13 varieties, sample cap 100k each (~1.1M tot)
#   - evaluation:    FLORES + OLDI (centroid + parallel) with isotropy fix
#   - estimated time: ~10-15h for 3 epochs at batch=32
# ===========================================================================

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

nvidia-smi || true
df -h "$HOME" || true
echo ""
date
python -u analysis/multilingual_xlmr/experiments/mlm_wiki_to_flores_oldi/run.py
status=$?
date
echo "exit status: $status"
exit $status
