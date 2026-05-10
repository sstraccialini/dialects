#!/bin/bash
#SBATCH --job-name=ltp_06_w2v_native
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# ============================================================================
# Cell 6 — Word2Vec on Wiki+OLDI cap-rule, eval on FLORES (NATIVE).
# ============================================================================

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

date
python -u analysis/word2vec/experiments/word2vec_wikiOLDI_native/run.py
status=$?
date
echo "exit status: $status"
exit $status
