#!/bin/bash
# One-time bootstrap on HPC.
#
# 1) read per-user paths from slurm/tools/env.local.sh (LTP_VENV, FLUX_ENV,
#    LTP_HF_CACHE)
# 2) create a small clean venv at $LTP_VENV (from /usr/bin/python3.9, no
#    --system-site-packages so it stays minimal)
# 3) if FLUX_ENV is set and exists, prepend its site-packages to PYTHONPATH
#    while running pip so pip skips already-installed torch / transformers /
#    scipy / numpy / ...; otherwise pip installs everything into LTP_VENV
# 4) pip install requirements.txt into LTP_VENV
#
# flux_env (when used) is never modified.
#
# Run on a login node (NOT inside an sbatch). Internet access required.
# Usage:
#     cp slurm/tools/env.local.example.sh slurm/tools/env.local.sh
#     ${EDITOR:-nano} slurm/tools/env.local.sh
#     bash slurm/tools/setup_env.sh

set -euo pipefail

THIS_FILE="${BASH_SOURCE[0]}"
TOOLS_DIR="$(cd "$(dirname "$THIS_FILE")" && pwd)"
PROJECT_ROOT="$(cd "$TOOLS_DIR/../.." && pwd)"

if [ -f "$TOOLS_DIR/env.local.sh" ]; then
    # shellcheck disable=SC1091
    source "$TOOLS_DIR/env.local.sh"
else
    echo "ERROR: $TOOLS_DIR/env.local.sh not found." >&2
    echo "       cp slurm/tools/env.local.example.sh slurm/tools/env.local.sh" >&2
    exit 1
fi

LTP_VENV="${LTP_VENV:-$HOME/ltp_env}"
LTP_HF_CACHE="${LTP_HF_CACHE:-$HOME/ltp_hf_cache}"

# Pick a clean Python interpreter (NOT flux_env's python, to avoid pip's
# "no --user inside a venv" guard).
PY_BIN="${PY_BIN:-/usr/bin/python3.9}"
if [ ! -x "$PY_BIN" ]; then
    for cand in python3.10 python3.9 python3; do
        if command -v "$cand" >/dev/null 2>&1; then
            PY_BIN="$(command -v "$cand")"
            break
        fi
    done
fi
echo "[setup] using interpreter: $($PY_BIN --version) at $PY_BIN"

mkdir -p "$LTP_HF_CACHE"

if [ ! -d "$LTP_VENV" ]; then
    echo "[setup] creating clean venv at $LTP_VENV"
    "$PY_BIN" -m venv "$LTP_VENV"
fi

# shellcheck disable=SC1091
source "$LTP_VENV/bin/activate"

# If FLUX_ENV is provided, make its site-packages visible to pip so it
# knows torch / transformers / scipy / ... are already installed and skips
# them. Saves ~3 GB of redundant downloads.
if [ -n "${FLUX_ENV:-}" ]; then
    FLUX_SITE="$FLUX_ENV/lib/python3.9/site-packages"
    if [ -d "$FLUX_SITE" ]; then
        export PYTHONPATH="$FLUX_SITE${PYTHONPATH:+:$PYTHONPATH}"
        echo "[setup] flux_env detected: $FLUX_ENV"
    else
        echo "[setup] FLUX_ENV set but site-packages not found at $FLUX_SITE" >&2
        echo "        falling back to a full install into LTP_VENV." >&2
    fi
else
    echo "[setup] no FLUX_ENV provided — installing all requirements into LTP_VENV"
fi

python -m pip install -U pip wheel

echo ""
echo "[setup] installing requirements ..."
python -m pip install \
    --upgrade-strategy only-if-needed \
    -r "$TOOLS_DIR/requirements.txt"

echo ""
echo "[setup] sanity import check ..."
python - <<'PY'
import importlib, sys
mods = [
    "torch", "transformers", "sentencepiece", "numpy", "scipy",
    "matplotlib", "tqdm", "huggingface_hub",
    "pandas", "sklearn", "seaborn", "gensim", "sentence_transformers",
    "datasets",
]
bad = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        loc = getattr(mod, "__file__", "<builtin>") or ""
        tag = "flux_env" if "flux_env" in loc else ("ltp_env" if "ltp_env" in loc else "?")
        print(f"  ok   [{tag:8s}] {m}")
    except Exception as e:
        bad.append((m, e))
        print(f"  FAIL {m}: {e}")
if bad:
    sys.exit(1)
PY

# Download FLORES+ if missing.
export HF_HOME="$LTP_HF_CACHE"
export HF_HUB_CACHE="$LTP_HF_CACHE/hub"

if [ ! -f "$PROJECT_ROOT/Dataset/flores/flores_plus/parallel.tsv" ]; then
    echo "[setup] FLORES+ not found, downloading ..."
    if [ -f "$PROJECT_ROOT/Dataset/flores/scripts/download_flores.py" ]; then
        cd "$PROJECT_ROOT"
        python Dataset/flores/scripts/download_flores.py
    fi
else
    echo "[setup] FLORES+ already present, skipping download"
fi

echo ""
echo "[setup] done."
echo "        ltp_env:    $LTP_VENV  ($(du -sh "$LTP_VENV" 2>/dev/null | cut -f1))"
echo "        HF cache:   $LTP_HF_CACHE"
[ -n "${FLUX_ENV:-}" ] && echo "        flux_env:   $FLUX_ENV (read-only)"
echo "        next:       sbatch slurm/jobs/run_all.slurm"
