#!/bin/bash
#SBATCH --job-name=edoardo_compare
#SBATCH --output=slurm/logs/edoardo_compare_%j.out
#SBATCH --error=slurm/logs/edoardo_compare_%j.err
#SBATCH --partition=medium_cpu
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#
# CPU-only job: lang2vec feature lookup + correlations (Spearman, Mantel,
# CKA, Procrustes) on tiny matrices.  No GPU needed.  Total runtime
# expected ~1-3 minutes; 30-minute wall is for the lang2vec import +
# safety margin.
#
# Submit from the repo root:
#     sbatch edoardo/slurm/run_edoardo.slurm
#
# To use a different partition (e.g. debug_cpu for fastest turnaround):
#     sbatch -p debug_cpu --qos=debug --time=00:15:00 edoardo/slurm/run_edoardo.slurm
#
# Optional environment variables:
#     EDOARDO_INCLUDE_OLD=1            also walk old_experiments/
#     EDOARDO_BASELINE_EXP=<name>      pin a baseline for shift / cka
#     EDOARDO_PERMUTATIONS=999         Mantel permutations (default 999)
#     EDOARDO_SKIP_ASJP=1              skip ASJP gold (no internet on compute nodes?)

set -euo pipefail

source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

# Make sure lang2vec is available; install into the user venv if missing.
python -c "import lang2vec" 2>/dev/null || {
    echo "[edoardo] installing lang2vec..."
    pip install --quiet lang2vec
}

cd "$PROJECT_ROOT"

# ---- 1. Build gold reference matrices ---------------------------------------
gold_args=()
if [ "${EDOARDO_SKIP_ASJP:-0}" = "1" ]; then
    gold_args+=(--skip asjp)
fi
python -m edoardo.gold_references.build_all "${gold_args[@]}"

# ---- 2. Run the full analysis pipeline --------------------------------------
runall_args=(--out-dir "$PROJECT_ROOT/edoardo/results"
             --permutations "${EDOARDO_PERMUTATIONS:-999}")
if [ "${EDOARDO_INCLUDE_OLD:-0}" = "1" ]; then
    runall_args+=(--include-old)
fi
if [ -n "${EDOARDO_BASELINE_EXP:-}" ]; then
    runall_args+=(--baseline-experiment "$EDOARDO_BASELINE_EXP")
fi

python -m edoardo.analysis.run_all "${runall_args[@]}"

echo
echo "Results in: $PROJECT_ROOT/edoardo/results"
ls -la "$PROJECT_ROOT/edoardo/results"
