#!/bin/bash
# Per-user HPC paths for SLURM jobs. Copy this file to env.local.sh and edit
# to point at YOUR own HPC paths. env.local.sh is gitignored, so every group
# member can keep their own settings without colliding with the others.
#
# Usage (one-time, on the HPC login node):
#     cp slurm/tools/env.local.example.sh slurm/tools/env.local.sh
#     ${EDITOR:-nano} slurm/tools/env.local.sh
#     bash slurm/tools/setup_env.sh

# Where your project's clean Python venv lives. setup_env.sh will create it
# from /usr/bin/python3.9 if missing.
export LTP_VENV="$HOME/ltp_env"

# (Optional) Path to a shared/read-only flux_env whose site-packages should
# be appended to PYTHONPATH. This lets us reuse torch / transformers / scipy /
# numpy / matplotlib / tqdm / huggingface_hub / sentencepiece / tokenizers
# without re-installing them into LTP_VENV (saves ~3 GB).
# Leave empty (or unset) if you don't have one — setup_env.sh will install
# everything into LTP_VENV from requirements.txt.
export FLUX_ENV="$HOME/FERT_PROJECT/Caches_and_venvs/flux_env"

# Isolated HuggingFace cache.
export LTP_HF_CACHE="$HOME/ltp_hf_cache"
