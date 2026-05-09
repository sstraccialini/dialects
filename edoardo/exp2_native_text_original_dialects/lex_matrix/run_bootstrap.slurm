#!/bin/bash
#SBATCH --job-name=lex_bootstrap
#SBATCH --output=slurm/logs/lex_bootstrap_%j.out
#SBATCH --error=slurm/logs/lex_bootstrap_%j.err
#SBATCH --partition=medium_cpu
#SBATCH --qos=normal
#SBATCH --time=01:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=16
#
# Compute mean Levenshtein (LD) + LDND distance matrices on the
# Wiktionary Swadesh-207 wordlist (split into m/f for gendered concepts),
# with 1000 bootstrap re-samples per pair to estimate 95% CI.
#
# CPU-only.  Parallelises across 8 workers, one per (i, j) pair.
# Tries to use python-Levenshtein (C-backed, ~50× faster) if installed.
#
# Submit from the repo root:
#     sbatch edoardo/exp2_native_text_original_dialects/lex_matrix/run_bootstrap.slurm

set -euo pipefail

source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

# Try to install the optional fast Levenshtein backend.  If it fails (e.g.
# no compiler available), the pure-Python fallback in build_lex_matrix.py
# kicks in automatically — slower but correct.
python -c "import Levenshtein" 2>/dev/null || pip install --quiet python-Levenshtein || true

cd "$PROJECT_ROOT"

LEX_DIR=$PROJECT_ROOT/edoardo/exp2_native_text_original_dialects/lex_matrix

python -m edoardo.exp2_native_text_original_dialects.lex_matrix.build_lex_matrix \
    --wordlist "$LEX_DIR/wordlist_swadesh207_split.csv" \
    --out-dir  "$LEX_DIR/matrices" \
    --bootstrap 1000 \
    --n-jobs ${SLURM_CPUS_PER_TASK:-8}

echo
echo "Generated files:"
ls -la "$LEX_DIR/matrices"
