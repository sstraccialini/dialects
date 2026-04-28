#!/bin/bash
# Convenience: sbatch all 10 pipeline jobs in one go.
# Run from experiments/:  bash slurm/submit_all.sh
set -euo pipefail

SLURM_DIR="$(cd "$(dirname "$0")" && pwd)"

for f in "$SLURM_DIR"/[0-9][0-9]_*.slurm; do
    sbatch "$f"
done

echo ""
echo "submitted. check queue:  squeue -u $USER"
