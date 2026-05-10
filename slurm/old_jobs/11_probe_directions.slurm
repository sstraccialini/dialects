#!/bin/bash
#SBATCH --job-name=probe_dialects
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/logs/11_probe_%A_%a.out
#SBATCH --error=slurm/logs/11_probe_%A_%a.err
#SBATCH --array=0-3

# Run linear-probe diagnostic across 4 candidate base LLMs.
# Each array task probes one model.
# Output: analysis/steering/{method_outputs,evaluation_results}/<slug>/

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

source ~/ltp_env/bin/activate
export HF_HOME="${HF_HOME:-$HOME/ltp_hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false

MODELS=(
  "utter-project/EuroLLM-1.7B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "meta-llama/Llama-3.2-1B-Instruct"
  "sapienzanlp/Minerva-3B-base-v1.0"
)

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
echo "[$(date)] Task $SLURM_ARRAY_TASK_ID: $MODEL"

python analysis/steering/src/probe_directions.py \
  --model "$MODEL" \
  --n_per_variety 1500 \
  --layer_step 2 \
  --batch_size 16 \
  --dtype bf16

echo "[$(date)] Done $MODEL"
