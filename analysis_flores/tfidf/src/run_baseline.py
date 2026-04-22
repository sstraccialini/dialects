"""
End-to-end orchestrator for the TF-IDF baseline on FLORES+.

Launch with (from repo root, with venv active):
    python analysis_flores/tfidf/src/run_baseline.py

Options:
    --sample-size N    sentences per variety (default: config.SAMPLE_SIZE = 2009)
    --pipeline {word,char,both}  which pipeline(s) to run (default both)
    --random-state N   random seed (default: config.RANDOM_STATE)

Outputs in `analysis_flores/tfidf/results/`:
    word/   distances.csv, top_features.csv, nearest_neighbors.csv,
            dendrogram.png, projection_mds.png, projection_tsne.png
    char/   same as word/
    shared/ silhouette_report.txt, run_stats.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sibling modules importable when running the file directly.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import VARIETY_CODES, results_subdir
from data_loader import load_all_varieties, build_variety_documents
from vectorize import (
    fit_transform_word, fit_transform_char, top_features_per_variety,
)
from similarity import (
    cosine_distance_matrix, save_distance_matrix, nearest_neighbors,
    save_nearest_neighbors, save_top_features,
)
from cluster import cluster_pipeline, save_silhouette_report
from visualize import visualize_pipeline


def run_pipeline(pipeline_name: str, fit_fn, variety_docs, codes) -> dict:
    """Run TF-IDF fit + distance + cluster + visualization for one pipeline."""
    print(f"\n=== Pipeline: {pipeline_name} ===")
    X, vectorizer = fit_fn(variety_docs)
    print(f"  TF-IDF shape: {X.shape}")
    dist = cosine_distance_matrix(X)

    save_distance_matrix(dist, codes, pipeline=pipeline_name)
    top = top_features_per_variety(X, vectorizer, codes, k=30)
    save_top_features(top, pipeline=pipeline_name)
    save_nearest_neighbors(nearest_neighbors(dist, codes, k=3), pipeline=pipeline_name)

    cluster_report = cluster_pipeline(dist, codes, pipeline_name)
    viz_paths = visualize_pipeline(dist, codes, pipeline_name)

    return {
        "pipeline": pipeline_name,
        "tfidf_shape": X.shape,
        "silhouette_family": cluster_report["silhouette_family"],
        "silhouette_romance_vs_rest": cluster_report["silhouette_romance_vs_rest"],
        "dendrogram": cluster_report["dendrogram_path"],
        "mds": viz_paths["mds"],
        "tsne": viz_paths["tsne"],
    }


def main():
    parser = argparse.ArgumentParser(description="TF-IDF baseline orchestrator (FLORES+)")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="sentences per variety (default: config.SAMPLE_SIZE)")
    parser.add_argument("--pipeline", choices=["word", "char", "both"], default="both",
                        help="which pipeline(s) to run")
    parser.add_argument("--random-state", type=int, default=None,
                        help="seed for sub-sampling")
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
    stats.to_csv(results_subdir("shared") / "run_stats.csv", index=False)

    docs, codes = build_variety_documents(data)

    reports = []
    if args.pipeline in ("word", "both"):
        reports.append(run_pipeline("word", fit_transform_word, docs, codes))
    if args.pipeline in ("char", "both"):
        reports.append(run_pipeline("char", fit_transform_char, docs, codes))

    raw = [
        {
            "pipeline": r["pipeline"],
            "n_varieties": len(codes),
            "silhouette_family": r["silhouette_family"],
            "silhouette_romance_vs_rest": r["silhouette_romance_vs_rest"],
            "dendrogram_path": r["dendrogram"],
        }
        for r in reports
    ]
    save_silhouette_report(raw)

    print("\n" + "=" * 60)
    print("Done. Summary:")
    for r in reports:
        print(f"  {r['pipeline']:>5}: tfidf_shape={r['tfidf_shape']}  "
              f"sil_family={r['silhouette_family']:+.4f}  "
              f"sil_romance={r['silhouette_romance_vs_rest']:+.4f}")
    print(f"\nResults directory: {results_subdir('').parent}")


if __name__ == "__main__":
    main()
