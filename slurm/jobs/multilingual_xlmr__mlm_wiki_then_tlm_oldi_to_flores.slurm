#!/bin/bash
#SBATCH --job-name=ltp_xlmr_mlm_wiki_then_tlm_oldi
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# XLM-R — sequential MLM on Wiki (stage 1) → TLM on OLDI (stage 2),
# then eval on FLORES+ (centroid + parallel). Stage 1 is the same MLM
# of mlm_wiki_to_flores_oldi but the resulting checkpoint is consumed
# as the starting point for stage 2 instead of being evaluated directly.
# See analysis/multilingual_xlmr/experiments/mlm_wiki_then_tlm_oldi_to_flores/run.py.

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

nvidia-smi || true
echo ""

python -u analysis/multilingual_xlmr/experiments/mlm_wiki_then_tlm_oldi_to_flores/run.py
