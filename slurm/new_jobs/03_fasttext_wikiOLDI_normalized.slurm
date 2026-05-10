#!/bin/bash
#SBATCH --job-name=ltp_03_fasttext_norm
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# ============================================================================
# Cell 3 — FastText subword on Wiki+OLDI cap-rule, eval on FLORES (NORMALIZED).
# CPU only. Training takes ~1-2h; centroid + parallel eval ~10 min.
# ============================================================================

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

date
python -u analysis/fasttext/experiments/fasttext_wikiOLDI_normalized/run.py
status=$?
date
echo "exit status: $status"
exit $status
