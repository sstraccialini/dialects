#!/bin/bash
#SBATCH --job-name=ltp_run_remaining
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# Resume after disk-full crash on step 08.
# Runs only the missing pipelines (08, 09, 10) and aggressively cleans
# the gensim FastText n-gram .npy blobs (1.5 GB each) right after each
# subword_fasttext step.

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"

nvidia-smi || true
echo ""

echo "=== pre-flight disk check ==="
df -h "$HOME" || true
echo ""

# Make sure nothing old is hogging quota before the heavy steps.
echo "=== pre-clean: any leftover fasttext .bin/.npy blobs ==="
find "$EXPERIMENTS_ROOT" -path '*/subword_fasttext/results/models*' \
    \( -name "*.bin*" -o -name "*.npy" -size +50M \) -print -delete 2>/dev/null || true
echo ""

run_step() {
    local label="$1"
    shift
    echo "=============================================================="
    echo "=== $label"
    echo "=============================================================="
    local t0
    t0=$(date +%s)
    if "$@"; then
        echo "[$label] OK ($(($(date +%s) - t0))s)"
    else
        echo "[$label] FAILED ($(($(date +%s) - t0))s) — continuing"
    fi
    echo ""
}

cleanup_fasttext_blobs() {
    # gensim FastText saves .bin + several huge sidecar .npy files
    # (vectors_ngrams.npy is ~1.5 GB). We don't reuse the model after
    # extracting variety vectors, so drop them immediately.
    local d="$1"
    if [ -d "$d" ]; then
        echo "[cleanup] removing fasttext model blobs in $d"
        rm -f "$d"/fasttext_model.bin* "$d"/*.bin* || true
        du -sh "$d" 2>/dev/null || true
    fi
}

run_step "08 wiki subword_fasttext"     python -u analysis_wiki/subword_fasttext/src/run_approach.py
cleanup_fasttext_blobs "$EXPERIMENTS_ROOT/analysis_wiki/subword_fasttext/results/models"

run_step "09 wiki multilingual"         python -u analysis_wiki/multilingual_embeddings/run_pipeline.py
run_step "10 wiki sentence_baseline"    python -u analysis_wiki/sentence_baseline/src/run_sentence_baseline.py

echo "=============================================================="
echo "=== run_remaining done"
echo "=============================================================="
df -h "$HOME" 2>/dev/null || true
