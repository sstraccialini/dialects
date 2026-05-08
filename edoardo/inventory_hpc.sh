#!/bin/bash
# Quick HPC inventory of trained models / embeddings / evaluation outputs.
#
# Run from the repo root on the HPC login node:
#     bash edoardo/inventory_hpc.sh
#     bash edoardo/inventory_hpc.sh --include-old
#
# Writes a CSV next to this script and prints a summary to stdout.

set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$THIS_DIR/.." && pwd)"

cd "$REPO_ROOT"

# Activate venv if env.sh is set up; otherwise rely on system python.
if [ -f "$REPO_ROOT/slurm/tools/env.local.sh" ] && [ -f "$REPO_ROOT/slurm/tools/env.sh" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/slurm/tools/env.sh" || true
fi

OUT_CSV="$THIS_DIR/results/hpc_inventory.csv"
mkdir -p "$(dirname "$OUT_CSV")"

python -m edoardo.analysis.inventory_models --csv "$OUT_CSV" "$@"

echo
echo "Inventory CSV: $OUT_CSV"
