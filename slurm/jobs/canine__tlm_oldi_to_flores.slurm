#!/bin/bash
#SBATCH --job-name=ltp_canine_tlm_oldi_to_flores
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# CANINE (char-level) — sentence-pair MLM on OLDI italian↔dialect pairs only,
# then eval on FLORES+ (centroid + parallel). Sequenze char concatenate
# fino a 1024 char, batch piccolo + grad-accumulation.
# See analysis/canine/experiments/tlm_oldi_to_flores/run.py.

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

nvidia-smi || true
echo ""

python -u analysis/canine/experiments/tlm_oldi_to_flores/run.py
