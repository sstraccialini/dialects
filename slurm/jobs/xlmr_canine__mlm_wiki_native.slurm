#!/bin/bash
#SBATCH --job-name=ltp_xlmr_canine_mlm_wiki_native
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# ===========================================================================
# Single-job continual MLM pretraining + evaluation on dialect data,
# native (not_normalized) text variant, all 13 varieties (rehearsal-style).
#
# Setup:
#   - text variant: not_normalized (cased, accented, punctuated)
#   - training corpus: Wiki, 13 varieties (6 dialects + 7 standards)
#                      sample cap 100k sentences per variety
#   - models trained sequentially in this single job:
#       1. multilingual_xlmr (xlm-roberta-base) + MLM Wiki + eval FLORES/OLDI
#       2. canine (google/canine-c)             + MLM Wiki + eval FLORES/OLDI
#   - evaluation: centroid + parallel alignment on FLORES, centroid on OLDI
#                 with anisotropy correction (Mu & Viswanath 2018) at variety
#                 level (set in core/evaluate.py: isotropy=True)
#
# Compute estimate:
#   XLM-R MLM Wiki (~286k training sentences, 3 epochs, batch=32)   ~ 5-7 h
#   CANINE MLM Wiki (~286k, 3 epochs, batch=16, longer seq=512)     ~ 7-10 h
#   evaluation on FLORES/OLDI per model                              ~ 1 h each
#   total                                                            ~ 14-19 h
# ===========================================================================

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

nvidia-smi || true
df -h "$HOME" || true
echo ""
echo "==========================================================="
echo "PHASE 1/2  —  XLM-R + MLM Wiki (native, 13 varieties)"
echo "==========================================================="
date
python -u analysis/multilingual_xlmr/experiments/mlm_wiki_to_flores_oldi/run.py
status_xlmr=$?
echo ""
echo "==========================================================="
echo "PHASE 2/2  —  CANINE + MLM Wiki (native, 13 varieties)"
echo "==========================================================="
date
python -u analysis/canine/experiments/mlm_wiki_to_flores_oldi/run.py
status_canine=$?
echo ""
echo "==========================================================="
echo "DONE"
echo "  xlmr   exit status: $status_xlmr"
echo "  canine exit status: $status_canine"
echo "==========================================================="
date

# Exit non-zero if either phase failed
if [ $status_xlmr -ne 0 ] || [ $status_canine -ne 0 ]; then
    exit 1
fi
