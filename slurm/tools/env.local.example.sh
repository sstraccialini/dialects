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
# from /usr/bin/python3.9 if missing. Each group member maintains their own
# venv — do NOT point at someone else's venv.
export LTP_VENV="$HOME/ltp_env"

# Isolated HuggingFace cache.
export LTP_HF_CACHE="$HOME/ltp_hf_cache"
