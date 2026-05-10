#!/bin/bash
#SBATCH --job-name=ltp_word2vec_wiki_to_flores_oldi
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# Word2Vec — train on Wikipedia, evaluate on FLORES+ and OLDI separately
# (centroid + parallel-alignment for each test corpus).
# See analysis/word2vec/experiments/wiki_to_flores_oldi/run.py for details.

set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

python -u analysis/word2vec/experiments/wiki_to_flores_oldi/run.py
