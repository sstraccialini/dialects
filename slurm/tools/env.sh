#!/bin/bash
# Shared HPC environment setup. Source from every SLURM job:
#     source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"
#
# Per-user paths (LTP_VENV, LTP_HF_CACHE) live in
#     slurm/tools/env.local.sh
# which is gitignored. Each group member copies env.local.example.sh to
# env.local.sh and edits to point at their own HPC paths.

THIS_FILE="${BASH_SOURCE[0]}"
TOOLS_DIR="$(cd "$(dirname "$THIS_FILE")" && pwd)"
SLURM_DIR="$(cd "$TOOLS_DIR/.." && pwd)"
export PROJECT_ROOT="$(cd "$SLURM_DIR/.." && pwd)"

# Per-user paths.
if [ -f "$TOOLS_DIR/env.local.sh" ]; then
    # shellcheck disable=SC1091
    source "$TOOLS_DIR/env.local.sh"
else
    echo "ERROR: $TOOLS_DIR/env.local.sh not found." >&2
    echo "       cp slurm/tools/env.local.example.sh slurm/tools/env.local.sh" >&2
    echo "       and fill in your own HPC paths (LTP_VENV, LTP_HF_CACHE)." >&2
    return 1 2>/dev/null || exit 1
fi

# Defaults (env.local.sh wins).
: "${LTP_VENV:=$HOME/ltp_env}"
: "${LTP_HF_CACHE:=$HOME/ltp_hf_cache}"

# Activate the project venv or conda env.
if [ -f "$LTP_VENV/bin/activate" ]; then
    # standard venv
    # shellcheck disable=SC1091
    source "$LTP_VENV/bin/activate"
elif [ -f "$LTP_VENV/bin/python" ]; then
    # conda env (no bin/activate): prepend bin/ to PATH directly
    export PATH="$LTP_VENV/bin:$PATH"
    export CONDA_PREFIX="$LTP_VENV"
else
    echo "[env.sh] WARNING: $LTP_VENV/bin/activate not found" >&2
    echo "         run 'bash slurm/tools/setup_env.sh' on a login node first." >&2
fi

# HuggingFace cache (isolated).
mkdir -p "$LTP_HF_CACHE"
export HF_HOME="$LTP_HF_CACHE"
export HF_HUB_CACHE="$LTP_HF_CACHE/hub"
export TRANSFORMERS_CACHE="$LTP_HF_CACHE"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$PROJECT_ROOT"
