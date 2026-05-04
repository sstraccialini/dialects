#!/bin/bash
#SBATCH --job-name=ltp_minilm_tsdae_wiki_to_flores_oldi
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# Sentence-MiniLM — continued TSDAE pretraining on Wiki (13 varieties),
# then evaluation on FLORES+ and OLDI (centroid + parallel for each).
# See analysis/sentence_minilm/experiments/tsdae_wiki_to_flores_oldi/run.py.

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

nvidia-smi || true
df -h "$HOME" || true
echo ""

python -u analysis/sentence_minilm/experiments/tsdae_wiki_to_flores_oldi/run.py
