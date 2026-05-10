#!/bin/bash
#SBATCH --job-name=ltp_02_tfidf_native
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# ============================================================================
# Cell 2 — TF-IDF on Wiki+OLDI cap-rule, eval on FLORES (NATIVE). CPU only.
# ============================================================================

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

date
python -u analysis/tfidf/experiments/tfidf_wikiOLDI_native/run.py
status=$?
date
echo "exit status: $status"
exit $status
