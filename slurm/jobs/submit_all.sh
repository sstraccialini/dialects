#!/bin/bash
# Convenience: sbatch every numbered pipeline job in one go.
# Run from project root:  bash slurm/jobs/submit_all.sh
set -euo pipefail

JOBS_DIR="$(cd "$(dirname "$0")" && pwd)"

for f in "$JOBS_DIR"/[0-9][0-9]_*.slurm; do
    sbatch "$f"
done

echo ""
echo "submitted. check queue:  squeue -u $USER"
