#!/bin/bash
#SBATCH --job-name=ltp_13_bootstrap_deep
#SBATCH --partition=debug_gpunew
#SBATCH --qos=debug
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# ============================================================================
# Bootstrap CI on Spearman gold correlations for all 6 deep-model experiments.
# Each step loads the already-trained model, embeds FLORES once, runs B=1000
# stratified resamples, and writes ~1 KB bootstrap_results.csv next to the
# existing gold_correlations.csv.  NO TRAINING.
#
# Ordered cheapest → priciest so partial timeouts still yield partial results:
#   LaBSE  zero-shot / finetuned
#   XLM-R  zero-shot / finetuned
#   CANINE zero-shot / finetuned
# ============================================================================

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

nvidia-smi || true
date
echo ""

declare -a STEPS=(
    "analysis.labse.core.bootstrap              --experiment labse_zeroshot_native"
    "analysis.labse.core.bootstrap              --experiment labse_finetuned_oldi_dialects_native"
    "analysis.multilingual_xlmr.core.bootstrap  --experiment xlmr_zeroshot_native"
    "analysis.multilingual_xlmr.core.bootstrap  --experiment xlmr_finetuned_wikiOLDI_dialects_native"
    "analysis.canine.core.bootstrap             --experiment canine_zeroshot_native"
    "analysis.canine.core.bootstrap             --experiment canine_finetuned_wikiOLDI_dialects_native"
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
