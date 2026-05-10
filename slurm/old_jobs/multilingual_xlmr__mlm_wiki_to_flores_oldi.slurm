#!/bin/bash
#SBATCH --job-name=ltp_xlmr_mlm_wiki_to_flores_oldi
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# Multilingual XLM-R — continued MLM pretraining on Wiki (13 varieties),
# then evaluation on FLORES+ and OLDI (centroid + parallel for each).
# See analysis/multilingual_xlmr/experiments/mlm_wiki_to_flores_oldi/run.py.

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

nvidia-smi || true
df -h "$HOME" || true
echo ""

python -u analysis/multilingual_xlmr/experiments/mlm_wiki_to_flores_oldi/run.py
