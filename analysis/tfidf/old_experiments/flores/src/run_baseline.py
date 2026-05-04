"""
End-to-end orchestrator for the TF-IDF baseline on FLORES+.

Launch (from repo root, with venv active):
    python analysis/tfidf/flores/src/run_baseline.py

Two sub-pipelines are run by default ("word" and "char" n-grams). Each one
produces its own evaluation artefacts under
``analysis/tfidf/flores/evaluation_results/<word|char>/`` via the central
evaluation module; method-specific artefacts (top_features.csv, run_stats.csv)
live under ``method_outputs/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
for p in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import (
    VARIETY_CODES, VARIETY_GROUP, VARIETY_NAMES,
    GROUP_NAMES, GROUP_COLORS,
    outputs_subdir, evaluation_subdir,
)
from data_loader import load_all_varieties, build_variety_documents
from vectorize import (
    fit_transform_word, fit_transform_char, top_features_per_variety,
)

from evaluation.evaluation import run_evaluation


ROMANCE_FAMILIES = {"italo_romance", "italian", "romance"}


def save_top_features(top_dict: dict, variant: str) -> Path:
    """Long-form CSV: code, rank, feature, weight."""
    rows = []
    for code, feats in top_dict.items():
        for rank, (f, w) in enumerate(feats, start=1):
            rows.append({"code": code, "rank": rank, "feature": f, "weight": w})
    out = outputs_subdir(variant) / "top_features.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def run_pipeline(variant: str, fit_fn, variety_docs, codes) -> dict:
    print(f"\n=== Pipeline: {variant} ===")
    X, vectorizer = fit_fn(variety_docs)
    print(f"  TF-IDF shape: {X.shape}")

    save_top_features(
        top_features_per_variety(X, vectorizer, codes, k=30), variant=variant,
    )

    report = run_evaluation(
        variety_vectors=X.toarray() if hasattr(X, "toarray") else X,
        variety_codes=codes,
        out_dir=evaluation_subdir(variant),
        method_label=f"TF-IDF ({variant})",
        family_groups=VARIETY_GROUP,
        family_colors=GROUP_COLORS,
        family_display_names=GROUP_NAMES,
        display_names=VARIETY_NAMES,
        romance_families=ROMANCE_FAMILIES,
    )
    return {
        "variant": variant,
        "tfidf_shape": X.shape,
        "silhouette_family": report["silhouette_family"],
        "silhouette_romance_vs_rest": report["silhouette_romance_vs_rest"],
    }


def main():
    parser = argparse.ArgumentParser(description="TF-IDF baseline orchestrator (FLORES+)")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--pipeline", choices=["word", "char", "both"], default="both")
    parser.add_argument("--random-state", type=int, default=None)
    args = parser.parse_args()

    from config import SAMPLE_SIZE, RANDOM_STATE
    sample_size = args.sample_size or SAMPLE_SIZE
    random_state = args.random_state if args.random_state is not None else RANDOM_STATE

    print("TF-IDF baseline on FLORES+")
    print("=" * 60)
    print(f"  sample_size   = {sample_size}")
    print(f"  random_state  = {random_state}")
    print(f"  pipeline      = {args.pipeline}")
    print(f"  varieties     = {VARIETY_CODES}")
    print()

    print("Loading + sampling...")
    data, stats = load_all_varieties(
        sample_size=sample_size,
        random_state=random_state,
    )
    stats["sample_size_param"] = sample_size
    stats["random_state"] = random_state
    stats.to_csv(outputs_subdir() / "run_stats.csv", index=False)

    docs, codes = build_variety_documents(data)

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
    print(f"\nMethod outputs:       {outputs_subdir()}")
    print(f"Evaluation artefacts: {evaluation_subdir()}")


if __name__ == "__main__":
    main()
