"""
[TRASH] BPE + TF-IDF: train SentencePiece BPE on Wiki, evaluate on FLORES+ and OLDI.

Same train/eval split as the main FastText experiment, but the model is
"BPE pieces + TF-IDF" instead of subword embeddings. Kept here as a
sanity-check baseline; the whole `trash/` folder is meant to be
deleted once it's no longer interesting.

Self-contained: nothing in the project imports from this folder.

Output layout:

  method_outputs/
  ├── models/bpe_model.{model,vocab}
  ├── flores/variety_vectors.{npz,csv}
  ├── flores/top_features.csv
  ├── oldi/...
  ├── run_stats.csv
  └── run_meta.json
  evaluation_results/
  ├── flores/centroid/...
  ├── flores/parallel/...
  ├── oldi/centroid/...
  └── oldi/parallel/...

Launch:
    python analysis/fasttext/experiments/trash/run.py
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
    BPE_VOCAB_SIZE, BPE_CHARACTER_COVERAGE,
    BPE_TFIDF_MIN_DF, BPE_TFIDF_MAX_DF, SUBLINEAR_TF, NORM,
    experiment_dirs,
)
from analysis.fasttext.core.data_loader import (
    load_wiki_for_training, load_flores_parallel, load_oldi_parallel,
)
from analysis.fasttext.core.embed import (
    train_bpe, fit_transform_bpe, transform_bpe, per_sentence_bpe,
    top_bpe_features_per_variety,
)
from analysis.fasttext.core.evaluate import variety_eval, parallel_eval


METHOD = "fasttext"
EXPERIMENT = "trash_bpe"


def _save_variety_vectors(X, codes, out_dir: Path) -> None:
    dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    np.savez_compressed(
        out_dir / "variety_vectors.npz",
        matrix=dense.astype(np.float32), labels=np.asarray(codes),
    )
    pd.DataFrame(dense, index=codes).to_csv(
        out_dir / "variety_vectors.csv", float_format="%.6f",
    )


def _save_top_features(top_dict: dict, out_dir: Path) -> None:
    rows = []
    for code, feats in top_dict.items():
        for rank, (feat, weight) in enumerate(feats, start=1):
            rows.append({"code": code, "rank": rank, "feature": feat, "weight": weight})
    pd.DataFrame(rows).to_csv(out_dir / "top_features.csv", index=False)


def evaluate_on(test_name: str, test_data, sp, vectorizer, codes):
    print(f"\n--- Eval on {test_name.upper()} ---")
    mo_root = SCRIPT_DIR / "method_outputs" / test_name
    mo_root.mkdir(parents=True, exist_ok=True)

    # Centroid eval
    X, codes_out = transform_bpe(sp, vectorizer, test_data, codes)
    _save_variety_vectors(X, codes_out, mo_root)
    _save_top_features(top_bpe_features_per_variety(X, vectorizer, codes_out, k=30), mo_root)

    _, er_centroid = experiment_dirs(SCRIPT_DIR, f"{test_name}/centroid")
    centroid_report = variety_eval(
        X, codes_out, out_dir=er_centroid,
        method_label=f"BPE+TFIDF ({test_name.upper()} centroid)",
    )

    # Parallel eval
    per_sent = per_sentence_bpe(sp, vectorizer, test_data, codes)
    _, er_parallel = experiment_dirs(SCRIPT_DIR, f"{test_name}/parallel")
    parallel_eval(
        per_sent, out_dir=er_parallel,
        method_label=f"BPE+TFIDF ({test_name.upper()} parallel)",
    )

    print(f"  centroid: sil_family={centroid_report['silhouette_family']:+.4f}  "
          f"sil_romance={centroid_report['silhouette_romance_vs_rest']:+.4f}")


def main():
    parser = argparse.ArgumentParser(description="BPE+TFIDF: train Wiki, eval FLORES + OLDI")
    parser.add_argument("--sample-size",  type=int, default=SAMPLE_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    print(f"{METHOD} — {EXPERIMENT}  (this folder is intentionally throwaway)")
    print("=" * 60)
    print(f"  sample_size  = {args.sample_size}")
    print(f"  random_state = {args.random_state}")
    print(f"  varieties    = {VARIETY_CODES}")
    print()

    print("Loading Wiki ...")
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

    print("\n--- Training BPE + fitting TF-IDF on Wiki ---")
    sp = train_bpe(wiki_data, VARIETY_CODES, model_prefix=mo_root / "models" / "bpe_model")
    X_wiki, vec, _ = fit_transform_bpe(sp, wiki_data, VARIETY_CODES)

    write_run_meta(
        out_dir=mo_root,
        method=METHOD,
        experiment=EXPERIMENT,
        params={
            "sample_size":            args.sample_size,
            "random_state":           args.random_state,
            "bpe_vocab_size":         BPE_VOCAB_SIZE,
            "bpe_character_coverage": BPE_CHARACTER_COVERAGE,
            "tfidf_min_df":           BPE_TFIDF_MIN_DF,
            "tfidf_max_df":           BPE_TFIDF_MAX_DF,
            "sublinear_tf":           SUBLINEAR_TF,
            "norm":                   NORM,
        },
    )

    evaluate_on("flores", flores_data, sp, vec, VARIETY_CODES)
    evaluate_on("oldi",   oldi_data,   sp, vec, VARIETY_CODES)

    print("\nDone.")


if __name__ == "__main__":
    main()
