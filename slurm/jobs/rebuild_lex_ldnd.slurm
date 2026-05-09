#!/bin/bash
#SBATCH --job-name=lex_rebuild
#SBATCH --output=slurm/logs/lex_rebuild_%j.out
#SBATCH --error=slurm/logs/lex_rebuild_%j.err
#SBATCH --partition=medium_cpu
#SBATCH --qos=normal
#SBATCH --time=01:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=16
#
# Rebuild the lexicostatistical LDND gold matrix from Wiktionary, with
# 1000-bootstrap CI and 16-core parallelism over variety pairs.
#
# Run this whenever the variety set in gold/lexicostatistical/varieties.py
# changes (more Wikipedia languages added, dialect set restructured).
#
# Pipeline:
#   1. fetch_swadesh.py    — scrape Wiktionary, write wordlists/wordlist_swadesh207.csv
#   2. expand_gendered.py  — m/f split, write wordlists/wordlist_swadesh207_split.csv
#   3. build_ldnd.py       — ASJPcode + Levenshtein + bootstrap → matrices/lexicostat_ldnd.npz
#
# Submit from the repo root:
#     sbatch slurm/jobs/rebuild_lex_ldnd.slurm

set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

# Optional fast Levenshtein (~50× speedup); falls back to pure Python if unavailable.
python -c "import Levenshtein" 2>/dev/null || pip install --quiet python-Levenshtein || true

cd "$PROJECT_ROOT"

LEX_DIR=$PROJECT_ROOT/gold/lexicostatistical
WL_DIR=$LEX_DIR/wordlists
OUT_DIR=$LEX_DIR/matrices

mkdir -p "$WL_DIR" "$OUT_DIR"

python -m gold.lexicostatistical.fetch_swadesh \
    --out "$WL_DIR/wordlist_swadesh207.csv"

python -m gold.lexicostatistical.expand_gendered \
    --in  "$WL_DIR/wordlist_swadesh207.csv" \
    --out "$WL_DIR/wordlist_swadesh207_split.csv"

python -m gold.lexicostatistical.build_ldnd \
    --wordlist "$WL_DIR/wordlist_swadesh207_split.csv" \
    --out-dir  "$OUT_DIR" \
    --bootstrap 1000 \
    --n-jobs   ${SLURM_CPUS_PER_TASK:-16}

echo
echo "Output:"
ls -la "$OUT_DIR"
