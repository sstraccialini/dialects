#!/bin/bash
#SBATCH --job-name=ltp_01_tfidf_norm
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# ============================================================================
# Cell 1 — TF-IDF (sentence-level fit, b-style) on Wiki+OLDI cap-rule dialects,
# eval on FLORES (NORMALIZED). Word + char_wb pipelines. CPU only.
# ============================================================================

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

date
python -u analysis/tfidf/experiments/tfidf_wikiOLDI_normalized/run.py
status=$?
date
echo "exit status: $status"
exit $status
