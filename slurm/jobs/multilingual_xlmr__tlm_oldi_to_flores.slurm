#!/bin/bash
#SBATCH --job-name=ltp_xlmr_tlm_oldi_to_flores
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# XLM-R — TLM (sentence-pair MLM) on OLDI italian↔dialect pairs only,
# then eval on FLORES+ (centroid + parallel).
# See analysis/multilingual_xlmr/experiments/tlm_oldi_to_flores/run.py.

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

nvidia-smi || true
echo ""

python -u analysis/multilingual_xlmr/experiments/tlm_oldi_to_flores/run.py
