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

GOLD_DIR=$PROJECT_ROOT/gold/lexicostatistical/matrices
OUT_CSV=$PROJECT_ROOT/gold/lexicostatistical/results/correlation_with_gold_pivot.csv

mkdir -p "$(dirname "$OUT_CSV")"

python -u -m evaluation.correlate_against_gold \
    --gold-dir "$GOLD_DIR" \
    --analysis-root "$PROJECT_ROOT/analysis" \
    --out "$OUT_CSV"

echo
echo "Top 10 models by rho (against lex LDND):"
python3 -c "
import pandas as pd
df = pd.read_csv('$OUT_CSV')
print(df.sort_values('rho', ascending=False).head(10)[['model_id','n_shared_full','rho','rho2']].to_string(index=False))
"
