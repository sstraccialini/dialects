#!/bin/bash
# One-time bootstrap on HPC.
#
# Strategy:
#   1) create a small clean venv at $LTP_VENV (from /usr/bin/python3.9, no
#      --system-site-packages, so it stays minimal)
#   2) prepend flux_env's site-packages to PYTHONPATH while running pip,
#      so pip SEES flux_env's already-installed torch / transformers /
#      scipy / numpy / ... and marks them as satisfied (skipping ~3GB of
#      duplicate downloads)
#   3) pip install requirements.txt into ltp_env — only the missing ones
#      (pandas, scikit-learn, seaborn, gensim, sentence-transformers,
#      datasets) plus their truly-missing transitive deps actually land in
#      ltp_env (~150 MB total).
#
# flux_env is never modified.
#
# Run on a login node (NOT inside an sbatch). Internet access required.
# Usage:  cd experiments && bash slurm/setup_env.sh

set -euo pipefail

THIS_FILE="${BASH_SOURCE[0]}"
SLURM_DIR="$(cd "$(dirname "$THIS_FILE")" && pwd)"
EXPERIMENTS_ROOT="$(cd "$SLURM_DIR/.." && pwd)"

LTP_VENV="${LTP_VENV:-$HOME/ltp_env}"
LTP_HF_CACHE="${LTP_HF_CACHE:-$HOME/ltp_hf_cache}"
FLUX_ENV="${FLUX_ENV:-$HOME/FERT_PROJECT/Caches_and_venvs/flux_env}"
FLUX_SITE="$FLUX_ENV/lib/python3.9/site-packages"

if [ ! -d "$FLUX_SITE" ]; then
    echo "ERROR: flux_env site-packages not found at $FLUX_SITE" >&2
    exit 1
fi

# Use a clean Python interpreter (NOT flux_env's python, to avoid pip's
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

# CRITICAL: make flux_env visible to pip so it knows torch/transformers/
# scipy/etc are already installed and skips them.
export PYTHONPATH="$FLUX_SITE${PYTHONPATH:+:$PYTHONPATH}"

python -m pip install -U pip wheel

echo ""
echo "[setup] installing requirements (flux_env packages will be skipped) ..."
python -m pip install \
    --upgrade-strategy only-if-needed \
    -r "$SLURM_DIR/requirements.txt"

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
        # mark whether it came from flux_env or ltp_env
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

if [ ! -f "$EXPERIMENTS_ROOT/flores_data/flores_plus/parallel.tsv" ]; then
    echo "[setup] FLORES+ not found, downloading ..."
    if [ -f "$EXPERIMENTS_ROOT/flores_data/scripts/download_flores.py" ]; then
        cd "$EXPERIMENTS_ROOT"
        python flores_data/scripts/download_flores.py
    fi
else
    echo "[setup] FLORES+ already present, skipping download"
fi

echo ""
echo "[setup] done."
echo "        flux_env (read-only):  $FLUX_ENV"
echo "        ltp_env:               $LTP_VENV  ($(du -sh "$LTP_VENV" 2>/dev/null | cut -f1))"
echo "        HF cache:              $LTP_HF_CACHE"
echo "        next:                  sbatch slurm/run_all.slurm"
