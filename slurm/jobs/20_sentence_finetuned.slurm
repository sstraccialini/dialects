#!/bin/bash
#SBATCH --job-name=ltp_20_sentence_ft
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# Sentence Baseline fine-tuning experiments
# Runs all four conditions in sequence:
#   baseline      — no adaptation (paraphrase-multilingual-MiniLM-L12-v2 as-is)
#   tsdae_wiki      — Unsupervised learning (TSDAE) on Wikipedia dialect text
#   mnrl_oldi      — Contrastive learning (MNRL) on OLDI parallel pairs
#   tsdae_then_mnrl  — Unsupervised on wiki, then Contrastive on OLDI (sequential)
# Evaluation is always on FLORES+ devtest (never seen in training).
set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

nvidia-smi || true
df -h "$HOME" || true
echo ""

# 4 conditions, each its own experiment script. tsdae_then_mnrl reuses
# the tsdae_wiki checkpoint (--skip-train) once it's been trained.
python -u analysis/sentence_finetuned/experiments/baseline/run.py
python -u analysis/sentence_finetuned/experiments/tsdae_wiki/run.py \
    --tsdae-epochs 3 \
    --wiki-samples 10000
python -u analysis/sentence_finetuned/experiments/mnrl_oldi/run.py \
    --mnrl-epochs 5
python -u analysis/sentence_finetuned/experiments/tsdae_then_mnrl/run.py \
    --tsdae-epochs 3 \
    --mnrl-epochs 5 \
    --wiki-samples 10000 \
    --skip-train  # reuse tsdae_wiki checkpoint trained above
