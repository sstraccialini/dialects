#!/bin/bash
#SBATCH --job-name=edoardo_compare
#SBATCH --output=slurm/logs/edoardo_compare_%j.out
#SBATCH --error=slurm/logs/edoardo_compare_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#
# Build all gold-reference matrices and run the full edoardo/ pipeline.
# Submit from the repo root:
#     sbatch edoardo/slurm/run_edoardo.slurm
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
