#!/bin/bash
#SBATCH --job-name=hist_influence
#SBATCH --output=slurm/logs/hist_influence_%j.out
#SBATCH --error=slurm/logs/hist_influence_%j.err
#SBATCH --partition=debug_cpu
#SBATCH --qos=debug
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#
# Evaluate every model under analysis/ against the team-curated
# historical-influence gold (gold/historical_influence/influences.csv).
# Set-based metric: Mean Precision@3 (top-3 model neighbours vs top-3
# documented historical influences per dialect).
#
# Submit from the repo root:
#     sbatch slurm/jobs/check_historical_influence.slurm

set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"
cd "$PROJECT_ROOT"

OUT_DIR=$PROJECT_ROOT/gold/historical_influence/results
mkdir -p "$OUT_DIR"

python -u -m evaluation.check_historical_influence \
    --gold-csv "$PROJECT_ROOT/gold/historical_influence/influences.csv" \
    --analysis-root "$PROJECT_ROOT/analysis" \
    --out-dir "$OUT_DIR"

echo
echo "Files produced:"
ls -la "$OUT_DIR"
