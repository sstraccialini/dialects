"""
CANINE base — ZERO-SHOT (no fine-tuning), evaluate on FLORES (NATIVE).

Cell 9 of the FINAL 12-cell experimental matrix (EXPERIMENTAL_PLAN.md §3.2).

Setup:
  model       : google/canine-c (character-level encoder)
  fine-tuning : NONE
  text        : NATIVE
  eval        : centroid + parallel on FLORES

Launch:
    python analysis/canine/experiments/canine_zeroshot_native/run.py
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
from analysis._shared.varieties import VARIETY_CODES
from analysis.canine.core.config import (
    DEFAULT_MODEL_NAME, MAX_LENGTH, BATCH_SIZE, experiment_dirs,
)
from analysis.canine.core.embedder import (
    CanineEmbedder, aggregate_variety_vectors,
)
from analysis.canine.core.evaluate import variety_eval, parallel_eval


METHOD = "canine"
EXPERIMENT = "canine_zeroshot_native"
TEXT_VARIANT = "native"


def _save_variety_vectors(X: np.ndarray, codes, out_dir: Path) -> None:
    np.savez_compressed(
        out_dir / "variety_vectors.npz",
        matrix=X.astype(np.float32), labels=np.asarray(codes),
    )
    pd.DataFrame(X, index=codes).to_csv(
        out_dir / "variety_vectors.csv", float_format="%.6f",
    )


def _flatten(data: dict, codes):
    sents, sent_codes = [], []
    for code in codes:
        if code not in data:
            continue
        for s in data[code]:
            sents.append(s)
            sent_codes.append(code)
    return sents, sent_codes


def evaluate_on(test_name: str, test_data, embedder: CanineEmbedder, codes):
    print(f"\n--- Eval on {test_name.upper()} ---")
    mo = SCRIPT_DIR / "method_outputs" / test_name
    mo.mkdir(parents=True, exist_ok=True)

    sents, sent_codes = _flatten(test_data, codes)
    sent_vecs = embedder.encode(sents, batch_size=BATCH_SIZE)
    X, codes_out = aggregate_variety_vectors(sent_vecs, sent_codes, codes)
    _save_variety_vectors(X, codes_out, mo)
    _, er_centroid = experiment_dirs(SCRIPT_DIR, f"{test_name}/centroid")
    rep = variety_eval(X, codes_out, out_dir=er_centroid,
                       method_label=f"CANINE zero-shot ({test_name} centroid)")

    per_variety = embedder.encode_per_variety(test_data, codes, batch_size=BATCH_SIZE)
    _, er_parallel = experiment_dirs(SCRIPT_DIR, f"{test_name}/parallel")
    parallel_eval(per_variety, out_dir=er_parallel,
                  method_label=f"CANINE zero-shot ({test_name} parallel)")

    print(f"  centroid: sil_fam={rep['silhouette_family']:+.4f}  "
          f"sil_rom={rep['silhouette_romance_vs_rest']:+.4f}  "
          f"sil_rom_noDial={rep.get('silhouette_romance_no_dialects')}")


def main():
    parser = argparse.ArgumentParser(description="CANINE zero-shot → FLORES (native)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    print(f"{METHOD} — {EXPERIMENT}")
    print("=" * 60)
    print(f"  model         = {DEFAULT_MODEL_NAME}")
    print(f"  text variant  = {TEXT_VARIANT}")
    print(f"  fine-tuning   = NONE (zero-shot)")
    print(f"  varieties     = {VARIETY_CODES}")
    print()

    mo_root = SCRIPT_DIR / "method_outputs"
    mo_root.mkdir(parents=True, exist_ok=True)

    write_run_meta(
        out_dir=mo_root, method=METHOD, experiment=EXPERIMENT,
        params={
            "base_model":   DEFAULT_MODEL_NAME,
            "text_variant": TEXT_VARIANT,
            "fine_tuning":  "none",
            "max_length":   MAX_LENGTH,
            "batch_size":   BATCH_SIZE,
        },
    )

    print(f"Loading FLORES ({TEXT_VARIANT}) ...")
    flores_data, _ = load_flores(text_variant=TEXT_VARIANT, verbose=False)

    embedder = CanineEmbedder(
        model_name=DEFAULT_MODEL_NAME, device=args.device, max_length=MAX_LENGTH,
    )
    evaluate_on("flores", flores_data, embedder, VARIETY_CODES)
    print("\nDone.")


if __name__ == "__main__":
    main()
