#!/bin/bash
#SBATCH --job-name=ltp_reeval_no_iso
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# ===========================================================================
# Re-run evaluation on existing variety_vectors.npz with isotropy=False.
# Diagnostic: check whether the anisotropy correction (top-1 PC removal)
# was destroying the Romance vs non-Romance signal in our last run.
#
# CPU-only (no model loading, just numpy + sklearn). Should run in <2 min.
# ===========================================================================

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

date
python -u <<'PY'
import sys; sys.path.insert(0, '.')
import numpy as np
from pathlib import Path
from evaluation import run_evaluation
from analysis._shared.varieties import (
    VARIETY_GROUP, GROUP_NAMES, GROUP_COLORS, VARIETY_NAMES, ROMANCE_FAMILIES,
)

for method in ["multilingual_xlmr", "canine"]:
    for corpus in ["flores", "oldi"]:
        npz_path = Path(
            f"analysis/{method}/experiments/mlm_wiki_to_flores_oldi/"
            f"method_outputs/{corpus}/variety_vectors.npz"
        )
        if not npz_path.exists():
            print(f"  MISSING: {npz_path}")
            continue
        d = np.load(npz_path, allow_pickle=True)
        X, codes = np.asarray(d["matrix"]), list(d["labels"])

        out_dir = Path(
            f"analysis/{method}/experiments/mlm_wiki_to_flores_oldi/"
            f"evaluation_results/{corpus}/centroid_no_isotropy"
        )
        run_evaluation(
            variety_vectors=X, variety_codes=codes, out_dir=out_dir,
            method_label=f"{method} (no-iso, {corpus})",
            family_groups=VARIETY_GROUP, family_colors=GROUP_COLORS,
            family_display_names=GROUP_NAMES, display_names=VARIETY_NAMES,
            romance_families=ROMANCE_FAMILIES,
            isotropy=False,
        )
        sil_path = out_dir / "silhouette_report.txt"
        print(f"\n=== {method} / {corpus} (NO isotropy correction) ===")
        if sil_path.exists():
            for line in sil_path.read_text().splitlines()[:6]:
                print("  " + line)
        print(f"  → outputs in {out_dir}")
PY
date
echo "exit status: $?"
