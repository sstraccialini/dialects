#!/bin/bash
#SBATCH --job-name=ltp_14_bootstrap_tfidf
#SBATCH --partition=medium_cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# ============================================================================
# TF-IDF bootstrap CI for all four (experiment × pipeline) combinations:
#   tfidf_wikiOLDI_native       × {char, word}
#   tfidf_wikiOLDI_normalized   × {char, word}
#
# Re-fits the TfidfVectorizer from Wiki+OLDI (deterministic, sklearn fit is
# pure counting) and runs B=1000 sparse-aware stratified bootstrap of the
# Spearman ρ against the 3 gold matrices.  No GPU needed.
#
# Char pipelines dominate the cost: char_wb (3,5)-grams on ~1.7M training
# sentences → vocab ~500k → ~5-15 min fit each.  Both experiments together
# should finish well under the 2 h budget.
# ============================================================================

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

date
echo ""

declare -a STEPS=(
    "analysis.tfidf.core.bootstrap --experiment tfidf_wikiOLDI_native"
    "analysis.tfidf.core.bootstrap --experiment tfidf_wikiOLDI_normalized"
)

declare -a RESULTS=()
overall=0
for step in "${STEPS[@]}"; do
    echo "================================================================"
    echo ">>> $step"
    echo "================================================================"
    date
    # shellcheck disable=SC2086
    python -u -m $step
    rc=$?
    date
    RESULTS+=("$rc  $step")
    if [ "$rc" -ne 0 ]; then
        overall=$rc
        echo "!! step failed (exit $rc) — continuing with the rest"
    fi
    echo ""
done

echo "================================================================"
echo "Summary (exit_code  step)"
echo "================================================================"
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo ""
echo "overall exit: $overall"
exit $overall
