"""
FastText subword: train on Wikipedia, evaluate on FLORES+ and OLDI.

A single shared FastText (skip-gram, subword char n-grams 3..6) is
trained on the Wiki corpus (all 13 varieties combined). The trained
embeddings are then applied to FLORES+ and OLDI; for each test corpus
we compute centroid eval + parallel-alignment eval.

Output layout:

  method_outputs/
  ├── models/fasttext_model.bin
  ├── flores/variety_vectors.{npz,csv}
  ├── oldi/variety_vectors.{npz,csv}
  ├── run_stats.csv
  └── run_meta.json
  evaluation_results/
  ├── flores/centroid/...
  ├── flores/parallel/...
  ├── oldi/centroid/...
  └── oldi/parallel/...

Launch:
    python analysis/fasttext/experiments/wiki_to_flores_oldi/run.py
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
from analysis.fasttext.core.config import (
    VARIETY_CODES, SAMPLE_SIZE, RANDOM_STATE,
    FT_VECTOR_SIZE, FT_WINDOW, FT_MIN_COUNT, FT_MIN_N, FT_MAX_N,
    FT_EPOCHS, FT_SG, FT_WORKERS,
    experiment_dirs,
)
from analysis.fasttext.core.data_loader import (
    load_wiki_for_training, load_flores_parallel, load_oldi_parallel,
)
from analysis.fasttext.core.embed import (
    train_fasttext, variety_embeddings_fasttext, per_sentence_fasttext,
)
from analysis.fasttext.core.evaluate import variety_eval, parallel_eval


METHOD = "fasttext"
EXPERIMENT = "wiki_to_flores_oldi"


def _save_variety_vectors(X: np.ndarray, codes, out_dir: Path) -> None:
    np.savez_compressed(
        out_dir / "variety_vectors.npz",
        matrix=X.astype(np.float32), labels=np.asarray(codes),
    )
    pd.DataFrame(X, index=codes).to_csv(
        out_dir / "variety_vectors.csv", float_format="%.6f",
    )


def evaluate_on(test_name: str, test_data, model, codes):
    """Run centroid + parallel eval on one parallel test corpus."""
    print(f"\n--- Eval on {test_name.upper()} ---")
    mo_root = SCRIPT_DIR / "method_outputs" / test_name
    mo_root.mkdir(parents=True, exist_ok=True)

    # Centroid eval
    X, codes_out = variety_embeddings_fasttext(model, test_data, codes)
    _save_variety_vectors(X, codes_out, mo_root)
    _, er_centroid = experiment_dirs(SCRIPT_DIR, f"{test_name}/centroid")
    centroid_report = variety_eval(
        X, codes_out, out_dir=er_centroid,
        method_label=f"FastText ({test_name.upper()} centroid)",
    )

    # Parallel eval
    per_sent = per_sentence_fasttext(model, test_data, codes)
    _, er_parallel = experiment_dirs(SCRIPT_DIR, f"{test_name}/parallel")
    parallel_eval(
        per_sent, out_dir=er_parallel,
        method_label=f"FastText ({test_name.upper()} parallel)",
    )

    print(f"  centroid: sil_family={centroid_report['silhouette_family']:+.4f}  "
          f"sil_romance={centroid_report['silhouette_romance_vs_rest']:+.4f}")


def main():
    parser = argparse.ArgumentParser(description="FastText: train Wiki, eval FLORES + OLDI")
    parser.add_argument("--sample-size",  type=int, default=SAMPLE_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    print(f"{METHOD} — {EXPERIMENT}")
    print("=" * 60)
    print(f"  sample_size  = {args.sample_size}")
    print(f"  random_state = {args.random_state}")
    print(f"  varieties    = {VARIETY_CODES}")
    print()

    print("Loading Wiki (training) ...")
    wiki_data, wiki_stats = load_wiki_for_training(
        sample_size=args.sample_size, random_state=args.random_state,
    )
    print("\nLoading OLDI ...")
    oldi_data, _ = load_oldi_parallel(verbose=False)
    print("\nLoading FLORES ...")
    flores_data, _ = load_flores_parallel(verbose=False)

    mo_root = SCRIPT_DIR / "method_outputs"
    mo_root.mkdir(parents=True, exist_ok=True)
    wiki_stats["sample_size_param"] = args.sample_size
    wiki_stats["random_state"]      = args.random_state
    wiki_stats.to_csv(mo_root / "run_stats.csv", index=False)

    print("\n--- Training FastText on Wiki ---")
    model = train_fasttext(
        wiki_data, VARIETY_CODES,
        save_to=mo_root / "models" / "fasttext_model.bin",
    )

    write_run_meta(
        out_dir=mo_root,
        method=METHOD,
        experiment=EXPERIMENT,
        params={
            "sample_size":  args.sample_size,
            "random_state": args.random_state,
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
    evaluate_on("oldi",   oldi_data,   model, VARIETY_CODES)

    print("\nDone.")


if __name__ == "__main__":
    main()
