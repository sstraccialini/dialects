"""
LaBSE base — ZERO-SHOT (no fine-tuning), evaluate on FLORES (NATIVE).

Cell 7 of the FINAL 12-cell experimental matrix (EXPERIMENTAL_PLAN.md §3.2).

Setup:
  model       : sentence-transformers/LaBSE (768-dim, 109 langs)
  fine-tuning : NONE — straight encode() pass
  text        : NATIVE
  eval        : centroid + parallel on FLORES (1827 × 17)

Launch:
    python analysis/labse/experiments/labse_zeroshot_native/run.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis._shared.run_meta import write_run_meta
from analysis._shared.dataset_loaders import load_flores
from analysis._shared.varieties import (
    VARIETY_CODES, VARIETY_GROUP, GROUP_NAMES, GROUP_COLORS, VARIETY_NAMES,
    ROMANCE_FAMILIES, DIALECT_FAMILIES, experiment_dirs,
)
from evaluation import run_evaluation, run_parallel_alignment


METHOD = "labse"
EXPERIMENT = "labse_zeroshot_native"
TEXT_VARIANT = "native"

LABSE_MODEL = "sentence-transformers/LaBSE"
MAX_LENGTH = 128
BATCH_SIZE = 64


def _save_variety_vectors(X: np.ndarray, codes, out_dir: Path) -> None:
    np.savez_compressed(
        out_dir / "variety_vectors.npz",
        matrix=X.astype(np.float32), labels=np.asarray(codes),
    )
    pd.DataFrame(X, index=codes).to_csv(
        out_dir / "variety_vectors.csv", float_format="%.6f",
    )


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / np.where(norms == 0, 1.0, norms)


def _variety_eval(X, codes, out_dir, *, method_label: str):
    return run_evaluation(
        variety_vectors=np.asarray(X), variety_codes=list(codes),
        out_dir=out_dir, method_label=method_label,
        family_groups=VARIETY_GROUP, family_colors=GROUP_COLORS,
        family_display_names=GROUP_NAMES, display_names=VARIETY_NAMES,
        romance_families=ROMANCE_FAMILIES, dialect_families=DIALECT_FAMILIES,
        isotropy=False, isotropy_top_k_pc=1,
    )


def _parallel_eval(per_variety, out_dir, *, method_label: str):
    return run_parallel_alignment(
        sentence_vectors=per_variety, out_dir=out_dir, method_label=method_label,
        family_groups=VARIETY_GROUP, family_colors=GROUP_COLORS,
        family_display_names=GROUP_NAMES, display_names=VARIETY_NAMES,
    )


def main():
    parser = argparse.ArgumentParser(description="LaBSE zero-shot → FLORES (native)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    print(f"{METHOD} — {EXPERIMENT}")
    print("=" * 60)
    print(f"  model         = {LABSE_MODEL}")
    print(f"  text variant  = {TEXT_VARIANT}")
    print(f"  fine-tuning   = NONE (zero-shot)")
    print(f"  varieties     = {VARIETY_CODES}")
    print()

    mo_root = SCRIPT_DIR / "method_outputs"
    mo_root.mkdir(parents=True, exist_ok=True)

    write_run_meta(
        out_dir=mo_root, method=METHOD, experiment=EXPERIMENT,
        params={
            "base_model":   LABSE_MODEL,
            "text_variant": TEXT_VARIANT,
            "fine_tuning":  "none",
            "max_length":   MAX_LENGTH,
            "batch_size":   BATCH_SIZE,
        },
    )

    print(f"Loading FLORES ({TEXT_VARIANT}) ...")
    flores_data, _ = load_flores(text_variant=TEXT_VARIANT, verbose=False)

    print(f"\nLoading {LABSE_MODEL} ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(LABSE_MODEL, device=args.device)

    print("\n--- Encoding FLORES ---")
    per_variety = {}
    for code in VARIETY_CODES:
        if code not in flores_data:
            continue
        emb = model.encode(
            flores_data[code], batch_size=BATCH_SIZE,
            show_progress_bar=True, convert_to_numpy=True,
            normalize_embeddings=True,
        )
        per_variety[code] = emb.astype(np.float32)

    # Centroid: mean per variety + L2 renormalisation
    codes_out = [c for c in VARIETY_CODES if c in per_variety]
    X = np.vstack([per_variety[c].mean(axis=0) for c in codes_out])
    X = _l2_normalize(X).astype(np.float32)

    mo_flores = mo_root / "flores"
    mo_flores.mkdir(parents=True, exist_ok=True)
    _save_variety_vectors(X, codes_out, mo_flores)

    _, er_centroid = experiment_dirs(SCRIPT_DIR, "flores/centroid")
    rep = _variety_eval(X, codes_out, out_dir=er_centroid,
                        method_label="LaBSE zero-shot (FLORES centroid)")
    _, er_parallel = experiment_dirs(SCRIPT_DIR, "flores/parallel")
    _parallel_eval(per_variety, out_dir=er_parallel,
                   method_label="LaBSE zero-shot (FLORES parallel)")

    print(f"\n  centroid: sil_fam={rep['silhouette_family']:+.4f}  "
          f"sil_rom={rep['silhouette_romance_vs_rest']:+.4f}  "
          f"sil_rom_noDial={rep.get('silhouette_romance_no_dialects')}")
    print("\nDone.")


if __name__ == "__main__":
    main()
