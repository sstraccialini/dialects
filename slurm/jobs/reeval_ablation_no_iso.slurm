#!/bin/bash
#SBATCH --job-name=ltp_reeval_ablation_no_iso
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# ===========================================================================
# Re-evaluate the 4 (or 5) ablation matrix cells with isotropy=False.
#
# Edoardo's first batch of jobs ran with the legacy default isotropy=True,
# which destroys the Romance-vs-non-Romance signal on CANINE centroids.
# We've since flipped the default to isotropy=False (commit d92f510). This
# job loads the variety_vectors.npz already saved by each ablation cell and
# re-runs run_evaluation with isotropy=False, writing into a parallel
# `centroid_no_iso/` subdir. Compute: <2 min total (no GPU, no embedding,
# just clustering on 13×768 matrices).
#
# Cells covered (each x {flores, oldi}):
#   - canine/mlm_wiki_dialects_native
#   - canine/mlm_wiki_rehearsal_normalized
#   - multilingual_xlmr/mlm_wiki_dialects_native
#   - multilingual_xlmr/mlm_wiki_rehearsal_normalized
#   - canine/lora_wiki_dialects_normalized   (if present)
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

CELLS = [
    ("canine",            "mlm_wiki_dialects_native"),
    ("canine",            "mlm_wiki_rehearsal_normalized"),
    ("multilingual_xlmr", "mlm_wiki_dialects_native"),
    ("multilingual_xlmr", "mlm_wiki_rehearsal_normalized"),
    ("canine",            "lora_wiki_dialects_normalized"),
]

for method, experiment in CELLS:
    exp = Path(f"analysis/{method}/experiments/{experiment}")
    if not exp.is_dir():
        print(f"\n[skip] {method}/{experiment} — folder not found")
        continue
    for corpus in ["flores", "oldi"]:
        npz_path = exp / "method_outputs" / corpus / "variety_vectors.npz"
        if not npz_path.exists():
            print(f"  [skip] {method}/{experiment}/{corpus} — no variety_vectors.npz yet")
            continue
        d = np.load(npz_path, allow_pickle=True)
        X, codes = np.asarray(d["matrix"]), list(d["labels"])

        out_dir = exp / "evaluation_results" / corpus / "centroid_no_iso"
        run_evaluation(
            variety_vectors=X, variety_codes=codes, out_dir=out_dir,
            method_label=f"{method} ({experiment} no-iso, {corpus})",
            family_groups=VARIETY_GROUP, family_colors=GROUP_COLORS,
            family_display_names=GROUP_NAMES, display_names=VARIETY_NAMES,
            romance_families=ROMANCE_FAMILIES,
            isotropy=False,
        )
        sil_path = out_dir / "silhouette_report.txt"
        print(f"\n=== {method} / {experiment} / {corpus} (NO isotropy) ===")
        if sil_path.exists():
            for line in sil_path.read_text().splitlines()[:6]:
                print("  " + line)
        print(f"  -> outputs in {out_dir}")

print("\nAll re-evals done.")
PY
date
echo "exit status: $?"
