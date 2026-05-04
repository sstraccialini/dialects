#!/bin/bash
#SBATCH --job-name=ltp_19_xlmr_ft
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# Task 3: XLM-R fine-tuning experiments
# Runs all four conditions in sequence:
#   baseline      — no adaptation (XLM-R-base as-is)
#   mlm_wiki      — continued MLM on Wikipedia dialect text
#   tlm_oldi      — TLM on OLDI parallel pairs
#   mlm_then_tlm  — MLM on wiki, then TLM on OLDI (sequential)
# Evaluation is always on FLORES+ devtest (never seen in training).

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

nvidia-smi || true
df -h "$HOME" || true
echo ""

# 4 conditions, each its own experiment script. mlm_then_tlm reuses
# the mlm_wiki checkpoint (--skip-train) once it's been trained.
python -u analysis/xlmr_finetuned/experiments/baseline/run.py
python -u analysis/xlmr_finetuned/experiments/mlm_wiki/run.py \
    --mlm-epochs 3 \
    --wiki-samples 10000
python -u analysis/xlmr_finetuned/experiments/tlm_oldi/run.py \
    --tlm-epochs 5
python -u analysis/xlmr_finetuned/experiments/mlm_then_tlm/run.py \
    --mlm-epochs 3 \
    --tlm-epochs 5 \
    --wiki-samples 10000 \
    --skip-train  # reuse mlm_wiki checkpoint trained above
