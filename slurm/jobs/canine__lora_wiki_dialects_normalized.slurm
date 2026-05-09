#!/bin/bash
#SBATCH --job-name=ltp_canine_lora_dial_norm
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# ==============================================================================
# LoRA EXPERIMENT — CANINE + LoRA + 6 dialetti + normalized text (cap 50k/var).
# Confronto contro full-FT baseline mlm_wiki_dialects_normalized (+0.619 sil_rom).
# Atteso: silhouette comparabile o migliore con ~1% dei param trainabili.
# Expected runtime: 1.5-2h training + 30 min eval = ~2-2.5h.
# Output: analysis/canine/experiments/lora_wiki_dialects_normalized/
# ==============================================================================

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

# Make sure peft is installed in the venv (no-op if already present)
pip install -q "peft>=0.7.0" 2>&1 | tail -3 || true

nvidia-smi || true
df -h "$HOME" || true
echo ""
date
python -u analysis/canine/experiments/lora_wiki_dialects_normalized/run.py
status=$?
date
echo "exit status: $status"
exit $status
