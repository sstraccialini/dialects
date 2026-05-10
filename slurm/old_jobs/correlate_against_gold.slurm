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
# Correlate every model under analysis/ against every .npz gold matrix
# under gold/*/matrices/.  Output: gold/<name>/results/correlation_with_gold_pivot.csv
# with both rho (full upper triangle) and rho2 (dialect × external block).
#
# Submit from the repo root:
#     sbatch slurm/jobs/correlate_against_gold.slurm

set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"
cd "$PROJECT_ROOT"

OUT_DIR=$PROJECT_ROOT/gold/_correlations

mkdir -p "$OUT_DIR"

# All gold dirs we want to correlate against; add new ones here when more
# gold matrix families are added under gold/<family>/matrices/.
python -u -m evaluation.correlate_against_gold \
    --gold-dir \
        "$PROJECT_ROOT/gold/lexicostatistical/matrices" \
        "$PROJECT_ROOT/gold/geographic/matrices" \
    --analysis-root "$PROJECT_ROOT/analysis" \
    --out-dir "$OUT_DIR"

echo
echo "Files produced:"
ls -la "$OUT_DIR"
