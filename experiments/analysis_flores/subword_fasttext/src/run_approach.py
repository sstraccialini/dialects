"""
End-to-end orchestrator for the Subword / FastText approach on FLORES+.

Runs one or both sub-pipelines:
    fasttext   -> shared FastText + mean-pooled variety vectors
    bpe        -> SentencePiece BPE + TF-IDF on BPE pieces

For each sub-pipeline it produces (mirroring the TF-IDF baseline
structure):
    results/<pipeline>/distances.csv
    results/<pipeline>/nearest_neighbors.csv
    results/<pipeline>/dendrogram.png
    results/<pipeline>/projection_mds.png
    results/<pipeline>/projection_tsne.png
Plus pipeline-specific extras:
    fasttext/variety_vectors.csv, sentence_vectors.npz
    bpe/top_features.csv   (top BPE pieces per variety)

Shared outputs:
    results/shared/silhouette_report.txt
    results/shared/run_stats.csv

Launch (from repo root, with venv active):
    python analysis_flores/subword_fasttext/src/run_approach.py

Options:
    --pipeline {fasttext,bpe,both}   which sub-pipeline(s) to run
    --sample-size N                  sentences per variety (default: config.SAMPLE_SIZE = 2009)
    --random-state N                 seed (default: config.RANDOM_STATE)
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
from data_loader import load_all_varieties
from embed_fasttext import run_fasttext_pipeline
from embed_bpe import run_bpe_pipeline, top_pieces_per_variety
from similarity import (
    cosine_distance_matrix, save_distance_matrix,
    nearest_neighbors, save_nearest_neighbors, save_top_features,
)
from cluster import cluster_pipeline, save_silhouette_report
from visualize import visualize_pipeline


def _report_from(pipeline: str, X_shape, codes, cluster_report, viz_paths) -> dict:
    return {
        "pipeline": pipeline,
        "shape": tuple(X_shape),
        "n_varieties": len(codes),
        "silhouette_family": cluster_report["silhouette_family"],
        "silhouette_romance_vs_rest": cluster_report["silhouette_romance_vs_rest"],
        "dendrogram_path": cluster_report["dendrogram_path"],
        "mds": viz_paths["mds"],
        "tsne": viz_paths["tsne"],
    }


def run_fasttext(data) -> dict:
    X, codes, _ = run_fasttext_pipeline(data)
    dist = cosine_distance_matrix(X)
    save_distance_matrix(dist, codes, pipeline="fasttext")
    save_nearest_neighbors(nearest_neighbors(dist, codes, k=3), pipeline="fasttext")
    cluster_report = cluster_pipeline(dist, codes, "fasttext")
    viz_paths = visualize_pipeline(dist, codes, "fasttext")
    return _report_from("fasttext", X.shape, codes, cluster_report, viz_paths)


def run_bpe(data) -> dict:
    X, vec, codes, _sp = run_bpe_pipeline(data)
    dist = cosine_distance_matrix(X)
    save_distance_matrix(dist, codes, pipeline="bpe")
    save_nearest_neighbors(nearest_neighbors(dist, codes, k=3), pipeline="bpe")
    top = top_pieces_per_variety(X, vec, codes, k=30)
    save_top_features(top, pipeline="bpe")
    cluster_report = cluster_pipeline(dist, codes, "bpe")
    viz_paths = visualize_pipeline(dist, codes, "bpe")
    return _report_from("bpe", X.shape, codes, cluster_report, viz_paths)


def main():
    parser = argparse.ArgumentParser(
        description="Subword / FastText approach orchestrator (FLORES+)"
    )
    parser.add_argument(
        "--pipeline", choices=["fasttext", "bpe", "both"], default="both",
        help="which sub-pipeline(s) to run",
    )
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="sentences per variety (default: config.SAMPLE_SIZE)",
    )
    parser.add_argument(
        "--random-state", type=int, default=None,
        help="seed (default: config.RANDOM_STATE)",
    )
    args = parser.parse_args()

    from config import SAMPLE_SIZE, RANDOM_STATE
    sample_size = args.sample_size or SAMPLE_SIZE
    random_state = args.random_state if args.random_state is not None else RANDOM_STATE

    print("Subword / FastText approach on FLORES+")
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

    reports = []
    if args.pipeline in ("fasttext", "both"):
        reports.append(run_fasttext(data))
    if args.pipeline in ("bpe", "both"):
        reports.append(run_bpe(data))

    raw = [
        {
            "pipeline": r["pipeline"],
            "n_varieties": r["n_varieties"],
            "silhouette_family": r["silhouette_family"],
            "silhouette_romance_vs_rest": r["silhouette_romance_vs_rest"],
            "dendrogram_path": r["dendrogram_path"],
        }
        for r in reports
    ]
    save_silhouette_report(raw)

    print("\n" + "=" * 60)
    print("Done. Summary:")
    for r in reports:
        print(f"  {r['pipeline']:>8}: shape={r['shape']}  "
              f"sil_family={r['silhouette_family']:+.4f}  "
              f"sil_romance={r['silhouette_romance_vs_rest']:+.4f}")
    print(f"\nResults directory: {results_subdir('').parent}")


if __name__ == "__main__":
    main()
