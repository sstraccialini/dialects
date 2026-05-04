"""
TF-IDF on Wikipedia (in-domain, no cross-corpus).

Both sub-pipelines (word and char n-grams) are fitted on the Wiki
super-documents, one per variety, then evaluated as centroid distances
between the 13 varieties (6 dialects + 7 languages).

Output layout (relative to this experiment's folder):

  method_outputs/
  ├── word/
  │   ├── variety_vectors.{npz,csv}
  │   └── top_features.csv
  ├── char/
  │   └── ...
  ├── run_stats.csv
  └── run_meta.json
  evaluation_results/
  ├── word/                          (run_evaluation outputs: distances,
  └── char/                          dendrogram, projections, silhouette)

Launch (from repo root, venv active):
    python analysis/tfidf/experiments/wiki_only/run.py
    python analysis/tfidf/experiments/wiki_only/run.py --pipeline word
    python analysis/tfidf/experiments/wiki_only/run.py --pipeline char
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
from analysis.tfidf.core.config import (
    VARIETY_CODES, SAMPLE_SIZE, RANDOM_STATE,
    WORD_NGRAM_RANGE, CHAR_NGRAM_RANGE, CHAR_ANALYZER,
    SUBLINEAR_TF, NORM,
    experiment_dirs,
)
from analysis.tfidf.core.data_loader import (
    load_wiki_for_training, build_variety_documents,
)
from analysis.tfidf.core.vectorize import (
    fit_transform_word, fit_transform_char, top_features_per_variety,
)
from analysis.tfidf.core.evaluate import variety_eval


METHOD = "tfidf"
EXPERIMENT = "wiki_only"


def _save_top_features(top_dict: dict, out_dir: Path) -> None:
    rows = []
    for code, feats in top_dict.items():
        for rank, (feature, weight) in enumerate(feats, start=1):
            rows.append({"code": code, "rank": rank, "feature": feature, "weight": weight})
    pd.DataFrame(rows).to_csv(out_dir / "top_features.csv", index=False)


def _save_variety_vectors(X, codes, out_dir: Path) -> None:
    """Both .npz and .csv so it's easy to load and easy to inspect."""
    dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    np.savez_compressed(
        out_dir / "variety_vectors.npz",
        matrix=dense.astype(np.float32),
        labels=np.asarray(codes),
    )
    pd.DataFrame(dense, index=codes).to_csv(
        out_dir / "variety_vectors.csv", float_format="%.6f",
    )


def run_pipeline(variant: str, fit_fn, variety_docs, codes):
    print(f"\n=== Pipeline: {variant} ===")
    mo, er = experiment_dirs(SCRIPT_DIR, variant)

    X, vectorizer = fit_fn(variety_docs)
    print(f"  TF-IDF shape: {X.shape}")

    _save_variety_vectors(X, codes, mo)
    _save_top_features(top_features_per_variety(X, vectorizer, codes, k=30), mo)

    report = variety_eval(X, codes, out_dir=er, method_label=f"TF-IDF {variant} (Wiki)")
    return {
        "variant": variant,
        "tfidf_shape": tuple(X.shape),
        "silhouette_family":           report["silhouette_family"],
        "silhouette_romance_vs_rest":  report["silhouette_romance_vs_rest"],
    }


def main():
    parser = argparse.ArgumentParser(description="TF-IDF on Wikipedia (centroid eval)")
    parser.add_argument("--sample-size",   type=int, default=SAMPLE_SIZE)
    parser.add_argument("--random-state",  type=int, default=RANDOM_STATE)
    parser.add_argument("--pipeline",      choices=["word", "char", "both"], default="both")
    args = parser.parse_args()

    print(f"{METHOD} — {EXPERIMENT}")
    print("=" * 60)
    print(f"  pipeline      = {args.pipeline}")
    print(f"  sample_size   = {args.sample_size}")
    print(f"  random_state  = {args.random_state}")
    print(f"  varieties     = {VARIETY_CODES}")
    print()

    print("Loading + sampling Wiki ...")
    data, stats = load_wiki_for_training(
        sample_size=args.sample_size, random_state=args.random_state,
    )
    stats["sample_size_param"] = args.sample_size
    stats["random_state"]       = args.random_state

    docs, codes = build_variety_documents(data)

    mo_root = SCRIPT_DIR / "method_outputs"
    mo_root.mkdir(parents=True, exist_ok=True)
    stats.to_csv(mo_root / "run_stats.csv", index=False)

    write_run_meta(
        out_dir=mo_root,
        method=METHOD,
        experiment=EXPERIMENT,
        params={
            "sample_size":         args.sample_size,
            "random_state":        args.random_state,
            "pipeline":            args.pipeline,
            "word_ngram_range":    list(WORD_NGRAM_RANGE),
            "char_ngram_range":    list(CHAR_NGRAM_RANGE),
            "char_analyzer":       CHAR_ANALYZER,
            "sublinear_tf":        SUBLINEAR_TF,
            "norm":                NORM,
        },
    )

    reports = []
    if args.pipeline in ("word", "both"):
        reports.append(run_pipeline("word", fit_transform_word, docs, codes))
    if args.pipeline in ("char", "both"):
        reports.append(run_pipeline("char", fit_transform_char, docs, codes))

    print("\n" + "=" * 60)
    print("Done. Summary:")
    for r in reports:
        sf = r["silhouette_family"]
        sr = r["silhouette_romance_vs_rest"]
        print(f"  {r['variant']:>5}: tfidf_shape={r['tfidf_shape']}  "
              f"sil_family={sf:+.4f}  sil_romance={sr:+.4f}")


if __name__ == "__main__":
    main()
