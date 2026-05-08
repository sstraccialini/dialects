#!/bin/bash
#SBATCH --job-name=exp1_build_golds
#SBATCH --output=slurm/logs/exp1_golds_%j.out
#SBATCH --error=slurm/logs/exp1_golds_%j.err
#SBATCH --partition=medium_cpu
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#
# Experiment 1 — build all gold reference matrices for nap/scn/lij/vec/pms/eml.
# CPU-only; downloads Grambank/PHOIBLE/Lexibank/ASJP into ~/.cache/edoardo_gold.
#
# Submit from the repo root:
#     sbatch edoardo/exp1_uriel_native/slurm/build_golds.slurm

set -euo pipefail

source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

python -c "import lang2vec" 2>/dev/null || pip install --quiet lang2vec

cd "$PROJECT_ROOT"

VARIETIES_MOD=edoardo.exp1_uriel_native.varieties
GOLD_DIR=$PROJECT_ROOT/edoardo/exp1_uriel_native/gold_references/matrices

python -m edoardo._shared_gold_builders.build_all \
    --varieties-module "$VARIETIES_MOD" \
    --out-dir "$GOLD_DIR"

echo
echo "Gold matrices in: $GOLD_DIR"
ls -la "$GOLD_DIR"
