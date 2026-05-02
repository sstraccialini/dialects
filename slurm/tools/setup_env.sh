#!/bin/bash
# One-time bootstrap on HPC.
#
# 1) read per-user paths from slurm/tools/env.local.sh (LTP_VENV,
#    LTP_HF_CACHE)
# 2) create a clean venv at $LTP_VENV (from /usr/bin/python3.9, no
#    --system-site-packages so it stays self-contained)
# 3) pip install requirements.txt into LTP_VENV
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
        tag = "ltp_env" if "ltp_env" in loc else "?"
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
echo "        next:       sbatch slurm/jobs/run_all.slurm"
