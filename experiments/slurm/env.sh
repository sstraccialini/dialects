#!/bin/bash
# Shared HPC environment setup for the experiments/ pipelines.
# Source from every SLURM script: `source "$EXPERIMENTS_ROOT/slurm/env.sh"`.

THIS_FILE="${BASH_SOURCE[0]}"
SLURM_DIR="$(cd "$(dirname "$THIS_FILE")" && pwd)"
export EXPERIMENTS_ROOT="$(cd "$SLURM_DIR/.." && pwd)"

# Our own clean venv (built by setup_env.sh from /usr/bin/python3.9, NO
# --system-site-packages so it stays small).
export LTP_VENV="${LTP_VENV:-$HOME/ltp_env}"

# flux_env site-packages, prepended to PYTHONPATH so torch/transformers/etc.
# are importable from our venv READ-ONLY. flux_env is never modified.
export FLUX_ENV="${FLUX_ENV:-$HOME/FERT_PROJECT/Caches_and_venvs/flux_env}"
export FLUX_SITE="$FLUX_ENV/lib/python3.9/site-packages"

if [ -f "$LTP_VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$LTP_VENV/bin/activate"
else
    echo "[env.sh] WARNING: ltp_env not found at $LTP_VENV — run setup_env.sh first" >&2
fi

# Append flux_env's site-packages so its torch / transformers / scipy /
# numpy / matplotlib / tqdm / huggingface_hub / sentencepiece / tokenizers
# are importable. Append (not prepend) so any package we install into
# ltp_env wins over flux_env's version.
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$FLUX_SITE"

# Isolated HF cache (NOT shared with flux_env's hf_cache).
export LTP_HF_CACHE="${LTP_HF_CACHE:-$HOME/ltp_hf_cache}"
export HF_HOME="$LTP_HF_CACHE"
export HF_HUB_CACHE="$LTP_HF_CACHE/hub"
export TRANSFORMERS_CACHE="$LTP_HF_CACHE"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$EXPERIMENTS_ROOT"
