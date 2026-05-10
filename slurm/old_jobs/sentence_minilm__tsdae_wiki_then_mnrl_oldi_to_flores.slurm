#!/bin/bash
#SBATCH --job-name=ltp_minilm_tsdae_wiki_then_mnrl_oldi
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=09:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# Sentence-MiniLM — sequential TSDAE on Wiki (stage 1) → MNRL on OLDI (stage 2),
# then eval on FLORES+ (centroid + parallel).
# See analysis/sentence_minilm/experiments/tsdae_wiki_then_mnrl_oldi_to_flores/run.py.

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

nvidia-smi || true
echo ""

python -u analysis/sentence_minilm/experiments/tsdae_wiki_then_mnrl_oldi_to_flores/run.py
