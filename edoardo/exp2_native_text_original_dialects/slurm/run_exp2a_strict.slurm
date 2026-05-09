#!/bin/bash
#SBATCH --job-name=exp2a_strict
#SBATCH --output=slurm/logs/exp2a_strict_%j.out
#SBATCH --error=slurm/logs/exp2a_strict_%j.err
#SBATCH --partition=debug_cpu
#SBATCH --qos=debug
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#
# Experiment 2a (STRICT) — original 6 dialects + ONLY golds that cover all 6.
#
#   Strict-coverage golds for {fur, lij, lmo, sc, scn, vec}:
#     - glottolog_tree         (hand-coded Glottolog 5.x with distinct leaves)
#     - expert_dialectology    (hand-curated by team, Pellegrini-informed)
#     - geographic_glottolog   (Haversine on Glottolog languoid coordinates)
#
#   ANY gold that requires k-NN imputation or has missing entries for any
#   of the 6 dialects is excluded.  No URIEL (fur/lmo/sc imputed), no
#   Grambank (3/6 missing), no PHOIBLE (4/6 missing), no Lexibank (lmo
#   missing), no ASJP (sc/vec missing).
#
# Models analysed: every method's existing FLORES-based output under
# analysis/<method>/experiments/.  No retraining — uses normalized-Wiki
# variant the team already trained.
#
# Submit from the repo root:
#     sbatch edoardo/exp2_native_text_original_dialects/slurm/run_exp2a_strict.slurm

set -euo pipefail

source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

cd "$PROJECT_ROOT"

VARIETIES_MOD=edoardo.exp2_native_text_original_dialects.varieties
GOLD_DIR=$PROJECT_ROOT/edoardo/exp2_native_text_original_dialects/gold_references/matrices_strict
RESULTS_DIR=$PROJECT_ROOT/edoardo/exp2_native_text_original_dialects/results_strict

mkdir -p "$GOLD_DIR" "$RESULTS_DIR"

# 1. Build only the 3 strict-coverage gold matrices.
python -m edoardo._shared_gold_builders.build_all \
    --varieties-module "$VARIETIES_MOD" \
    --out-dir "$GOLD_DIR" \
    --only glottolog_tree,geographic_glottolog || true

# 2. Build the expert matrix (it lives in edoardo/gold_references; copy it
#    over so the strict gold dir is self-contained).
python -m edoardo.gold_references.build_expert \
    --out-dir "$GOLD_DIR" || true

# 3. Run the analysis pipeline against the strict gold dir only.
python -m edoardo.analysis.run_all \
    --gold-dir "$GOLD_DIR" \
    --out-dir  "$RESULTS_DIR" \
    --include-old \
    --permutations 999

echo
echo "Strict-coverage results in: $RESULTS_DIR"
ls -la "$RESULTS_DIR"
