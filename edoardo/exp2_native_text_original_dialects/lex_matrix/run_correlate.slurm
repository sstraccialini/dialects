#!/bin/bash
#SBATCH --job-name=lex_correlate
#SBATCH --output=slurm/logs/lex_correlate_%j.out
#SBATCH --error=slurm/logs/lex_correlate_%j.err
#SBATCH --partition=debug_cpu
#SBATCH --qos=debug
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#
# Correlate every model under analysis/ against the team-curated
# lexicostatistical gold matrices (LD = mean Levenshtein, and LDND).
# This is now the SOLE gold reference set we care about for the original
# 6 Italian dialects (fur, lij, lmo, sc, scn, vec) + 7 standards.
#
# All previous gold matrices (URIEL, Glottolog tree, Expert, Geographic,
# Grambank, PHOIBLE, Lexibank, ASJP) have been removed because either
# (a) they did not cover all 6 dialects, or (b) URIEL produced collapsed
# imputed vectors with severe loss of discriminative power across our
# Italian dialects (see paper Limitations section).
#
# CPU-only, ~3-5 minutes total.
#
# Submit from the repo root:
#     sbatch edoardo/exp2_native_text_original_dialects/lex_matrix/run_correlate.slurm

set -euo pipefail

source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

cd "$PROJECT_ROOT"

GOLD_DIR=$PROJECT_ROOT/edoardo/exp2_native_text_original_dialects/lex_matrix/matrices
RESULTS_DIR=$PROJECT_ROOT/edoardo/exp2_native_text_original_dialects/lex_matrix/correlation_results

mkdir -p "$RESULTS_DIR"

python -m edoardo.analysis.run_all \
    --gold-dir "$GOLD_DIR" \
    --out-dir  "$RESULTS_DIR" \
    --include-old \
    --permutations 999

echo
echo "Results written to: $RESULTS_DIR"
ls -la "$RESULTS_DIR"
