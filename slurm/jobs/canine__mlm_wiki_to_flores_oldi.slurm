#!/bin/bash
#SBATCH --job-name=ltp_canine_mlm_wiki_to_flores_oldi
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# CANINE (char-level) — continued MLM pretraining on Wiki (13 varieties),
# then evaluation on FLORES+ and OLDI (centroid + parallel for each).
# CANINE has heavier sequences than XLM-R (char-level, max_length=512),
# so allocation is generous on time and memory.
# See analysis/canine/experiments/mlm_wiki_to_flores_oldi/run.py.

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

nvidia-smi || true
df -h "$HOME" || true
echo ""

python -u analysis/canine/experiments/mlm_wiki_to_flores_oldi/run.py
