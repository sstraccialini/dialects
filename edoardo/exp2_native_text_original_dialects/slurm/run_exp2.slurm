#!/bin/bash
#SBATCH --job-name=exp2_richer_golds
#SBATCH --output=slurm/logs/exp2_%j.out
#SBATCH --error=slurm/logs/exp2_%j.err
#SBATCH --partition=medium_cpu
#SBATCH --qos=normal
#SBATCH --time=00:45:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#
# Experiment 2 — original 6 dialects with a richer gold-reference set.
# (Re-uses already-trained models from analysis/<method>/experiments/.)
#
# Steps:
#   1. download / cache Grambank, PHOIBLE, Lexibank, ASJP CLDF dumps
#   2. build all 8 gold matrices for the 13 varieties
#   3. correlate every existing model output (same as `edoardo/`) against
#      every gold; produce shift / cluster / CKA tables
#
# Submit from the repo root:
#     sbatch edoardo/exp2_native_text_original_dialects/slurm/run_exp2.slurm

set -euo pipefail

source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

python -c "import lang2vec" 2>/dev/null || pip install --quiet lang2vec

cd "$PROJECT_ROOT"

VARIETIES_MOD=edoardo.exp2_native_text_original_dialects.varieties
GOLD_DIR=$PROJECT_ROOT/edoardo/exp2_native_text_original_dialects/gold_references/matrices
RESULTS_DIR=$PROJECT_ROOT/edoardo/exp2_native_text_original_dialects/results

# 1. Build every gold matrix (URIEL + Glottolog + Grambank + PHOIBLE +
#    Lexibank + ASJP-lexical + ASJP-genealogy + Geographic).
#    Use ``|| true`` so partial failures (e.g. one source missing for our
#    varieties) do not abort the job — the analysis step uses whatever
#    matrices were successfully written.
python -m edoardo._shared_gold_builders.build_all \
    --varieties-module "$VARIETIES_MOD" \
    --out-dir "$GOLD_DIR" || true

# 2. Run the analysis pipeline against the new gold dir.
python -m edoardo.analysis.run_all \
    --gold-dir "$GOLD_DIR" \
    --out-dir  "$RESULTS_DIR" \
    --include-old \
    --permutations 999

echo
echo "Results in: $RESULTS_DIR"
ls -la "$RESULTS_DIR"
