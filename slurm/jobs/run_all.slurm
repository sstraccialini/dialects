#!/bin/bash
#SBATCH --job-name=ltp_run_all
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# Single-job umbrella: runs all 10 pipelines sequentially.
# Submit:  cd experiments && sbatch slurm/run_all.slurm

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

nvidia-smi || true
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
    # gensim FastText writes .bin + sidecar .npy files including
    # vectors_ngrams.npy (~1.5 GB). We've already extracted variety
    # vectors, so drop the model files immediately to free quota.
    local d="$1"
    if [ -d "$d" ]; then
        echo "[cleanup] removing fasttext model blobs in $d"
        rm -f "$d"/fasttext_model.bin* "$d"/*.bin* || true
        du -sh "$d" 2>/dev/null || true
    fi
}

run_step "01 flores tfidf"              python -u analysis/tfidf/flores/src/run_baseline.py
run_step "02 flores word2vec"           python -u analysis/word2vec/flores/src/run_word2vec.py
run_step "03 flores subword_fasttext"   python -u analysis/fasttext/flores/src/run_approach.py
cleanup_fasttext_blobs "$PROJECT_ROOT/analysis/fasttext/flores/method_outputs/models"
run_step "04 flores multilingual"       python -u analysis/multilingual_xlmr/flores/src/run_pipeline.py
run_step "05 flores sentence_baseline"  python -u analysis/sentence_baseline/flores/src/run_sentence_baseline.py
run_step "06 wiki tfidf"                python -u analysis/tfidf/wiki/src/run_baseline.py
run_step "07 wiki word2vec"             python -u analysis/word2vec/wiki/src/run_word2vec.py
run_step "08 wiki subword_fasttext"     python -u analysis/fasttext/wiki/src/run_approach.py
cleanup_fasttext_blobs "$PROJECT_ROOT/analysis/fasttext/wiki/method_outputs/models"
run_step "09 wiki multilingual"         python -u analysis/multilingual_xlmr/wiki/src/run_pipeline.py
run_step "10 wiki sentence_baseline"    python -u analysis/sentence_baseline/wiki/src/run_sentence_baseline.py

echo "=============================================================="
echo "=== run_all done"
echo "=============================================================="
