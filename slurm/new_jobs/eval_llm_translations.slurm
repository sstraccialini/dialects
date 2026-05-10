#!/bin/bash
#SBATCH --job-name=ltp_eval_llm_translations
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# ============================================================================
# Translation evaluation pipeline for two LLMs (ChatGPT vs Gemini).
#
# Runs five jobs in sequence:
#   1. compute_metrics.py   on data-chatgpt   (chrF++, BLEU, BERTScore F1)
#   2. compute_metrics.py   on data-gemini    (chrF++, BLEU, BERTScore F1)
#   3. bootstrap_significance.py on data-chatgpt
#         (within-LLM: which source language is best for ChatGPT?)
#   4. bootstrap_significance.py on data-gemini
#         (within-LLM: which source language is best for Gemini?)
#   5. cross_llm_bootstrap.py
#         (head-to-head: ChatGPT vs Gemini for every shared file)
#
# Resource notes:
#   * 1 GPU is needed for BERTScore (xlm-roberta-large).
#   * 16 CPUs are used for parallel paired bootstrap (joblib loky).
#   * BERTScore is the slowest single step; bootstrap is parallelised.
#
# Environment / paths:
#   * Override default folder layout via env vars (defaults shown in []).
#       PRED_DIR_CHATGPT  [llm_translations/data-chatgpt]
#       PRED_DIR_GEMINI   [llm_translations/data-gemini]
#       GOLD_FILE         [llm_translations/data/gold_table.csv]
#       OUT_DIR           [llm_translations/eval_results]
#       N_BOOTSTRAP       [1000]
#
# Submit:
#   sbatch slurm/jobs/eval_llm_translations.slurm
# ============================================================================

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

# --- configurable paths ---
: "${PRED_DIR_CHATGPT:=$PROJECT_ROOT/llm_translations/data-chatgpt}"
: "${PRED_DIR_GEMINI:=$PROJECT_ROOT/llm_translations/data-gemini}"
: "${GOLD_FILE:=$PROJECT_ROOT/llm_translations/oldi_selected.tsv}"
: "${OUT_DIR:=$PROJECT_ROOT/llm_translations/eval_results}"
: "${N_BOOTSTRAP:=1000}"
: "${BERTSCORE_MODEL:=xlm-roberta-large}"

mkdir -p "$OUT_DIR"

# --- guardrails ---
echo "==================================================================="
echo "Translation evaluation pipeline"
echo "==================================================================="
echo "  PROJECT_ROOT        = $PROJECT_ROOT"
echo "  PRED_DIR_CHATGPT    = $PRED_DIR_CHATGPT"
echo "  PRED_DIR_GEMINI     = $PRED_DIR_GEMINI"
echo "  GOLD_FILE           = $GOLD_FILE"
echo "  OUT_DIR             = $OUT_DIR"
echo "  N_BOOTSTRAP         = $N_BOOTSTRAP"
echo "  BERTSCORE_MODEL     = $BERTSCORE_MODEL"
echo "  CPUs available      = ${SLURM_CPUS_PER_TASK:-?}"
echo

nvidia-smi || true
echo

for f in "$PRED_DIR_CHATGPT" "$PRED_DIR_GEMINI" "$GOLD_FILE"; do
    if [ ! -e "$f" ]; then
        echo "ERROR: missing $f" >&2
        exit 1
    fi
done

# --- ensure deps (no-op if already in venv) ---
python -c "import sacrebleu, bert_score, joblib, numpy, pandas" 2>/dev/null \
    || pip install --quiet sacrebleu bert-score joblib

# Use all CPUs SLURM gave us for the bootstrap.
N_JOBS="${SLURM_CPUS_PER_TASK:--1}"

# ============================================================================
# 1. ChatGPT — compute chrF++, BLEU, BERTScore F1
# ============================================================================
echo
echo "############################################################"
echo "# Step 1/5  compute_metrics.py  on  ChatGPT"
echo "############################################################"
python -u "$PROJECT_ROOT/llm_translations/compute_metrics.py" \
    --pred_dir       "$PRED_DIR_CHATGPT" \
    --gold           "$GOLD_FILE" \
    --out            "$OUT_DIR/metrics_chatgpt.csv" \
    --on_mismatch    truncate \
    --bertscore_model "$BERTSCORE_MODEL"

# ============================================================================
# 2. Gemini — compute chrF++, BLEU, BERTScore F1
# ============================================================================
echo
echo "############################################################"
echo "# Step 2/5  compute_metrics.py  on  Gemini"
echo "############################################################"
python -u "$PROJECT_ROOT/llm_translations/compute_metrics.py" \
    --pred_dir       "$PRED_DIR_GEMINI" \
    --gold           "$GOLD_FILE" \
    --out            "$OUT_DIR/metrics_gemini.csv" \
    --on_mismatch    truncate \
    --bertscore_model "$BERTSCORE_MODEL"

# ============================================================================
# 3. ChatGPT — within-LLM paired bootstrap (which source language wins?)
# ============================================================================
echo
echo "############################################################"
echo "# Step 3/5  bootstrap_significance.py  on  ChatGPT"
echo "############################################################"
python -u "$PROJECT_ROOT/llm_translations/bootstrap_significance.py" \
    --pred_dir    "$PRED_DIR_CHATGPT" \
    --gold        "$GOLD_FILE" \
    --out         "$OUT_DIR/bootstrap_within_chatgpt.csv" \
    --n_bootstrap "$N_BOOTSTRAP" \
    --metrics     chrfpp bleu \
    --on_mismatch truncate \
    --n_jobs      "$N_JOBS" \
    --seed        42

# ============================================================================
# 4. Gemini — within-LLM paired bootstrap
# ============================================================================
echo
echo "############################################################"
echo "# Step 4/5  bootstrap_significance.py  on  Gemini"
echo "############################################################"
python -u "$PROJECT_ROOT/llm_translations/bootstrap_significance.py" \
    --pred_dir    "$PRED_DIR_GEMINI" \
    --gold        "$GOLD_FILE" \
    --out         "$OUT_DIR/bootstrap_within_gemini.csv" \
    --n_bootstrap "$N_BOOTSTRAP" \
    --metrics     chrfpp bleu \
    --on_mismatch truncate \
    --n_jobs      "$N_JOBS" \
    --seed        42

# ============================================================================
# 5. Head-to-head — ChatGPT vs Gemini, file by file
# ============================================================================
echo
echo "############################################################"
echo "# Step 5/5  cross_llm_bootstrap.py   ChatGPT  vs  Gemini"
echo "############################################################"
python -u "$PROJECT_ROOT/llm_translations/cross_llm_bootstrap.py" \
    --pred_dir_a  "$PRED_DIR_CHATGPT" \
    --pred_dir_b  "$PRED_DIR_GEMINI" \
    --label_a     chatgpt \
    --label_b     gemini \
    --gold        "$GOLD_FILE" \
    --out         "$OUT_DIR/cross_llm_chatgpt_vs_gemini.csv" \
    --n_bootstrap "$N_BOOTSTRAP" \
    --metrics     chrfpp bleu \
    --on_mismatch truncate \
    --n_jobs      "$N_JOBS" \
    --seed        42

# ============================================================================
echo
echo "==================================================================="
echo "Done. Outputs:"
ls -lh "$OUT_DIR"/*.csv 2>/dev/null || true
echo "==================================================================="
