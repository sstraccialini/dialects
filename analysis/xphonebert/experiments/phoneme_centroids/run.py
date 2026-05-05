"""
XPhoneBERT — phoneme-centroid experiment.

For each variety we compute a centroid from its **phoneme inventory** —
no sentences, no eSpeak, no FLORES. Pure phonological signature in
XPhoneBERT's embedding space.

Pipeline:
    1. Standard languages (9): hard-coded IPA inventories.
    2. Dialects (6): inventory extracted empirically from Manzini-Savoia
       transcriptions (greedy IPA tokenisation, drop singletons).
    3. Embed each phoneme via XPhoneBERT (input-layer lookup by default;
       full single-phoneme encoder pass with --mode single).
    4. Mean phonemes per variety → L2-normalised centroid.
    5. variety_eval (silhouette + heatmap + dendrogram + projection).

Outputs:
    method_outputs/
    ├── variety_inventory.csv      # phoneme list per variety
    ├── variety_vectors.{npz,csv}  # centroid matrix
    ├── run_meta.json
    └── ...
    evaluation_results/             # silhouette, heatmap, dendrogram, ...

Launch (locally on Mac, no GPU needed):
    python analysis/xphonebert/experiments/phoneme_centroids/run.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis._shared.run_meta import write_run_meta
from analysis.xphonebert.core.config import (
    VARIETY_CODES, DIALECT_CODES, STANDARD_LANGUAGE_CODES,
    MODEL_NAME, experiment_dirs,
)
from analysis.xphonebert.core.data_loader import (
    dialect_phonemes, standard_phonemes,
)
from analysis.xphonebert.core.embedder import PhonemeEmbedder, stack_centroids
from analysis.xphonebert.core.evaluate import variety_eval


METHOD = "xphonebert"
EXPERIMENT = "phoneme_centroids"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["lookup", "single"], default="lookup",
                        help="lookup = static input embedding; "
                             "single = full encoder on each phoneme")
    parser.add_argument("--min-count", type=int, default=2,
                        help="Drop dialect phonemes that appear < N times in MS")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    print(f"{METHOD} — {EXPERIMENT}")
    print("=" * 60)
    print(f"  base_model    = {MODEL_NAME}")
    print(f"  mode          = {args.mode}")
    print(f"  varieties     = {VARIETY_CODES}  (15 = 13 + arb + ell)")
    print(f"  dialect_min_count = {args.min_count}")
    print()

    mo_root = SCRIPT_DIR / "method_outputs"
    mo_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Step 1: phoneme inventories
    # ------------------------------------------------------------------ #
    print("Building phoneme inventories ...")
    inventories: dict = {}
    inventory_rows = []

    for code in DIALECT_CODES:
        phs, counter = dialect_phonemes(code, min_count=args.min_count)
        inventories[code] = phs
        inventory_rows.append({
            "code": code, "kind": "dialect",
            "n_phonemes": len(phs),
            "n_total_observations": int(sum(counter.values())),
            "phonemes": " ".join(phs),
        })
        print(f"  [dial   {code:>4}] {len(phs):3d} phonemes "
              f"(from {sum(counter.values()):,} obs)")

    for code in STANDARD_LANGUAGE_CODES:
        phs = standard_phonemes(code)
        inventories[code] = phs
        inventory_rows.append({
            "code": code, "kind": "standard",
            "n_phonemes": len(phs),
            "n_total_observations": -1,
            "phonemes": " ".join(phs),
        })
        print(f"  [stand  {code:>4}] {len(phs):3d} phonemes")

    pd.DataFrame(inventory_rows).to_csv(
        mo_root / "variety_inventory.csv", index=False
    )

    # ------------------------------------------------------------------ #
    # Step 2: embed each phoneme + build centroids
    # ------------------------------------------------------------------ #
    print(f"\nLoading XPhoneBERT (mode={args.mode}) ...")
    embedder = PhonemeEmbedder(
        model_name=MODEL_NAME, device=args.device, mode=args.mode,
    )

    print("\nComputing variety centroids ...")
    centroids = embedder.all_centroids(inventories)

    X, codes_out = stack_centroids(centroids, VARIETY_CODES)
    print(f"  centroid matrix: {X.shape}  (codes={codes_out})")

    # Save vectors
    np.savez_compressed(
        mo_root / "variety_vectors.npz",
        matrix=X.astype(np.float32), labels=np.asarray(codes_out),
    )
    pd.DataFrame(X, index=codes_out).to_csv(
        mo_root / "variety_vectors.csv", float_format="%.6f",
    )

    # ------------------------------------------------------------------ #
    # Step 3: full evaluation suite (silhouette + heatmap + dendrogram + ...)
    # ------------------------------------------------------------------ #
    _, er_dir = experiment_dirs(SCRIPT_DIR)
    report = variety_eval(
        X, codes_out, out_dir=er_dir,
        method_label=f"XPhoneBERT phoneme-centroids ({args.mode})",
    )
    print(f"\n  sil(family)  = {report['silhouette_family']:+.4f}")
    print(f"  sil(romance) = {report['silhouette_romance_vs_rest']:+.4f}")

    write_run_meta(
        out_dir=mo_root, method=METHOD, experiment=EXPERIMENT,
        params={
            "base_model":  MODEL_NAME,
            "mode":        args.mode,
            "min_count":   args.min_count,
            "varieties":   codes_out,
            "n_phonemes_per_variety": {c: len(inventories[c]) for c in codes_out},
        },
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
