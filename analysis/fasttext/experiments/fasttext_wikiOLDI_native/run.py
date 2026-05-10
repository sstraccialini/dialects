"""
FastText subword — train Wiki+OLDI dialect, evaluate on FLORES (NATIVE).

Cell 4 of the FINAL 12-cell experimental matrix. Same as the normalized
counterpart but with native (cased, accented, punctuated) text.

Launch:
    python analysis/fasttext/experiments/fasttext_wikiOLDI_native/run.py
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
from analysis._shared.dataset_loaders import (
    load_wiki_plus_oldi_dialect, load_flores,
)
from analysis._shared.varieties import (
    VARIETY_CODES, SAMPLE_SIZE, RANDOM_STATE,
)
from analysis.fasttext.core.config import (
    FT_VECTOR_SIZE, FT_WINDOW, FT_MIN_COUNT, FT_MIN_N, FT_MAX_N,
    FT_EPOCHS, FT_SG, FT_WORKERS,
    experiment_dirs,
)
from analysis.fasttext.core.embed import (
    train_fasttext, variety_embeddings_fasttext, per_sentence_fasttext,
)
from analysis.fasttext.core.evaluate import variety_eval, parallel_eval


METHOD = "fasttext"
EXPERIMENT = "fasttext_wikiOLDI_native"
TEXT_VARIANT = "native"


def _save_variety_vectors(X: np.ndarray, codes, out_dir: Path) -> None:
    np.savez_compressed(
        out_dir / "variety_vectors.npz",
        matrix=X.astype(np.float32), labels=np.asarray(codes),
    )
    pd.DataFrame(X, index=codes).to_csv(
        out_dir / "variety_vectors.csv", float_format="%.6f",
    )


def evaluate_on(test_name: str, test_data, model, codes):
    print(f"\n--- Eval on {test_name.upper()} ---")
    mo = SCRIPT_DIR / "method_outputs" / test_name
    mo.mkdir(parents=True, exist_ok=True)

    X, codes_out = variety_embeddings_fasttext(model, test_data, codes)
    _save_variety_vectors(X, codes_out, mo)
    _, er_centroid = experiment_dirs(SCRIPT_DIR, f"{test_name}/centroid")
    rep = variety_eval(X, codes_out, out_dir=er_centroid,
                       method_label=f"FastText ({EXPERIMENT}, {test_name} centroid)")

    per_sent = per_sentence_fasttext(model, test_data, codes)
    _, er_parallel = experiment_dirs(SCRIPT_DIR, f"{test_name}/parallel")
    parallel_eval(per_sent, out_dir=er_parallel,
                  method_label=f"FastText ({EXPERIMENT}, {test_name} parallel)")

    print(f"  centroid: sil_fam={rep['silhouette_family']:+.4f}  "
          f"sil_rom={rep['silhouette_romance_vs_rest']:+.4f}  "
          f"sil_rom_noDial={rep.get('silhouette_romance_no_dialects')}")


def main():
    parser = argparse.ArgumentParser(description="FastText wikiOLDI → FLORES (native)")
    parser.add_argument("--sample-size",  type=int, default=SAMPLE_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    print(f"{METHOD} — {EXPERIMENT}")
    print("=" * 60)
    print(f"  text variant = {TEXT_VARIANT}")
    print(f"  sample_size  = {args.sample_size}")
    print(f"  random_state = {args.random_state}")
    print(f"  varieties    = {VARIETY_CODES}")
    print()

    print(f"Loading Wiki + OLDI (dialect cap rule, {TEXT_VARIANT}) ...")
    train_data, train_stats = load_wiki_plus_oldi_dialect(
        text_variant=TEXT_VARIANT,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )
    mo_root = SCRIPT_DIR / "method_outputs"
    mo_root.mkdir(parents=True, exist_ok=True)
    train_stats.to_csv(mo_root / "run_stats.csv", index=False)

    print(f"\nLoading FLORES ({TEXT_VARIANT}) ...")
    flores_data, _ = load_flores(text_variant=TEXT_VARIANT, verbose=False)

    print("\n--- Training FastText on Wiki+OLDI ---")
    model = train_fasttext(
        train_data, VARIETY_CODES,
        save_to=mo_root / "models" / "fasttext_model.bin",
    )

    write_run_meta(
        out_dir=mo_root, method=METHOD, experiment=EXPERIMENT,
        params={
            "sample_size":  args.sample_size,
            "random_state": args.random_state,
            "text_variant": TEXT_VARIANT,
            "vector_size":  FT_VECTOR_SIZE,
            "window":       FT_WINDOW,
            "min_count":    FT_MIN_COUNT,
            "min_n":        FT_MIN_N,
            "max_n":        FT_MAX_N,
            "epochs":       FT_EPOCHS,
            "sg":           FT_SG,
            "workers":      FT_WORKERS,
        },
    )

    evaluate_on("flores", flores_data, model, VARIETY_CODES)
    print("\nDone.")


if __name__ == "__main__":
    main()
