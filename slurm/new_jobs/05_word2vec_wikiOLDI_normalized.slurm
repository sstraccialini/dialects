#!/bin/bash
#SBATCH --job-name=ltp_05_w2v_norm
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# ============================================================================
# Cell 5 — Word2Vec on Wiki+OLDI cap-rule, eval on FLORES (NORMALIZED).
# ============================================================================

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

date
python -u analysis/word2vec/experiments/word2vec_wikiOLDI_normalized/run.py
status=$?
date
echo "exit status: $status"
exit $status
