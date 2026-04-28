#!/bin/bash
# Shared HPC environment setup. Source from every SLURM job:
#     source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"
#
# Per-user paths (LTP_VENV, FLUX_ENV, LTP_HF_CACHE, ...) live in
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
    echo "       and fill in your own HPC paths (LTP_VENV, FLUX_ENV, LTP_HF_CACHE)." >&2
    return 1 2>/dev/null || exit 1
fi

# Defaults (env.local.sh wins).
: "${LTP_VENV:=$HOME/ltp_env}"
: "${LTP_HF_CACHE:=$HOME/ltp_hf_cache}"

# Activate the project venv.
if [ -f "$LTP_VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$LTP_VENV/bin/activate"
else
    echo "[env.sh] WARNING: $LTP_VENV/bin/activate not found" >&2
    echo "         run 'bash slurm/tools/setup_env.sh' on a login node first." >&2
fi

# Optional: prepend a shared read-only flux_env so we don't reinstall torch /
# transformers / scipy / numpy / matplotlib / tqdm / huggingface_hub /
# sentencepiece / tokenizers into our own venv. flux_env is never modified.
if [ -n "${FLUX_ENV:-}" ]; then
    FLUX_SITE="$FLUX_ENV/lib/python3.9/site-packages"
    if [ -d "$FLUX_SITE" ]; then
        # Append (not prepend) so anything we install into LTP_VENV wins.
        export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$FLUX_SITE"
    fi
fi

# HuggingFace cache (isolated from any shared one).
mkdir -p "$LTP_HF_CACHE"
export HF_HOME="$LTP_HF_CACHE"
export HF_HUB_CACHE="$LTP_HF_CACHE/hub"
export TRANSFORMERS_CACHE="$LTP_HF_CACHE"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$PROJECT_ROOT"
