#!/bin/bash
#SBATCH --job-name=eml_wiki_dl
#SBATCH --output=slurm/logs/eml_wiki_%j.out
#SBATCH --error=slurm/logs/eml_wiki_%j.err
#SBATCH --partition=medium_cpu
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#
# Download + extract the emlwiki Wikipedia dump for the new exp1 experiment.
#
# Steps performed by Dataset/wiki/scripts/create.py + generation.py:
#   1. wget the .xml.bz2
#   2. wikiextractor → JSON
#   3. invoke generation.py to clean + split + save
#      Dataset/wiki/normalized/others_dialects/eml.csv
#      Dataset/wiki/not_normalized/others_dialects/eml.csv
#
# IMPORTANT: before submitting, verify the actual emlwiki page range
# at https://dumps.wikimedia.org/emlwiki/2026-04-01/ and update the
# placeholder URL in Dataset/wiki/scripts/create.py.

set -euo pipefail

source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

cd "$PROJECT_ROOT"

# Existing pipeline already handles caching + skipping previously-fetched
# dumps; running create.py just adds eml without recomputing the others.
python Dataset/wiki/scripts/create.py
python Dataset/wiki/scripts/generation.py
python Dataset/wiki/scripts/generation_native.py

echo
echo "After this job, eml.csv should be at:"
echo "  Dataset/wiki/normalized/others_dialects/eml.csv"
echo "  Dataset/wiki/not_normalized/others_dialects/eml.csv"
