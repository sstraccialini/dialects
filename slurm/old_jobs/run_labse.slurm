#!/bin/bash
#SBATCH --job-name=ltp_run_labse
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# LaBSE on flores + wiki, in one umbrella job.

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

run_step "11 flores labse"   python -u analysis/sentence_labse/experiments/train_flores_eval_flores/run.py
run_step "12 wiki   labse"   python -u analysis/sentence_labse/experiments/train_wiki_eval_wiki/run.py

df -h "$HOME" 2>/dev/null || true
