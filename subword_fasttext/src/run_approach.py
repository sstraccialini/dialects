"""
End-to-end orchestrator for the Subword / FastText approach (Person 5).

Two pipelines:
  fasttext — gensim FastText subword embeddings (skip-gram, char n-grams)
  bpe      — SentencePiece BPE tokenization + TF-IDF on BPE pieces

Launch with (from repo root):
    python subword_fasttext/src/run_approach.py

Options:
    --sample-size N           sentences per variety (default config.SAMPLE_SIZE)
    --pipeline {fasttext,bpe,both}   which pipeline(s) to run (default both)
    --random-state N          random seed (default config.RANDOM_STATE)

Outputs in `subword_fasttext/results/`:
    fasttext/  distances.csv, nearest_neighbors.csv, variety_vectors.csv,
               dendrogram.png, projection_mds.png, projection_tsne.png
    bpe/       distances.csv, nearest_neighbors.csv, top_features.csv,
               dendrogram.png, projection_mds.png, projection_tsne.png
    shared/    silhouette_report.txt, run_stats.csv
    models/    fasttext_model.bin, bpe_model.model, bpe_model.vocab
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import VARIETY_CODES, results_subdir
from data_loader import load_all_varieties
from embed_fasttext import train_fasttext, variety_embeddings
from embed_bpe import train_bpe, fit_transform_bpe, top_bpe_features_per_variety
from similarity import (
    cosine_distance_matrix, save_distance_matrix,
    nearest_neighbors, save_nearest_neighbors, save_top_features,
)
from cluster import cluster_pipeline, save_silhouette_report
from visualize import visualize_pipeline


def run_fasttext_pipeline(data: dict, codes: list, random_state: int) -> dict:
    print("\n=== Pipeline: fasttext ===")
    model = train_fasttext(data, codes, save=True)
    X, _ = variety_embeddings(model, data, codes, save=True)
    print(f"  Embedding matrix shape: {X.shape}")

    dist = cosine_distance_matrix(X)
    save_distance_matrix(dist, codes, pipeline="fasttext")
    save_nearest_neighbors(nearest_neighbors(dist, codes, k=3), pipeline="fasttext")
    # Note: FastText dense embeddings have no interpretable "top features";
    # variety_vectors.csv is saved by variety_embeddings() instead.

    cluster_report = cluster_pipeline(dist, codes, "fasttext")
    viz_paths = visualize_pipeline(dist, codes, "fasttext")

    return {
        "pipeline": "fasttext",
        "embedding_shape": X.shape,
        "silhouette_family": cluster_report["silhouette_family"],
        "silhouette_romance_vs_rest": cluster_report["silhouette_romance_vs_rest"],
        "dendrogram": cluster_report["dendrogram_path"],
        "mds": viz_paths["mds"],
        "tsne": viz_paths["tsne"],
    }


def run_bpe_pipeline(data: dict, codes: list, random_state: int) -> dict:
    print("\n=== Pipeline: bpe ===")
    sp = train_bpe(data, codes)
    X, vectorizer = fit_transform_bpe(sp, data, codes)
    print(f"  BPE TF-IDF shape: {X.shape}")

    dist = cosine_distance_matrix(X)
    save_distance_matrix(dist, codes, pipeline="bpe")
    top = top_bpe_features_per_variety(X, vectorizer, codes, k=30)
    save_top_features(top, pipeline="bpe")
    save_nearest_neighbors(nearest_neighbors(dist, codes, k=3), pipeline="bpe")

    cluster_report = cluster_pipeline(dist, codes, "bpe")
    viz_paths = visualize_pipeline(dist, codes, "bpe")

    return {
        "pipeline": "bpe",
        "tfidf_shape": X.shape,
        "silhouette_family": cluster_report["silhouette_family"],
        "silhouette_romance_vs_rest": cluster_report["silhouette_romance_vs_rest"],
        "dendrogram": cluster_report["dendrogram_path"],
        "mds": viz_paths["mds"],
        "tsne": viz_paths["tsne"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Subword / FastText approach orchestrator (Person 5)"
    )
    parser.add_argument("--sample-size", type=int, default=None,
                        help="sentences per variety (default: config.SAMPLE_SIZE)")
    parser.add_argument("--pipeline", choices=["fasttext", "bpe", "both"], default="both",
                        help="which pipeline(s) to run")
    parser.add_argument("--random-state", type=int, default=None,
                        help="seed for sub-sampling")
    args = parser.parse_args()

    from config import SAMPLE_SIZE, RANDOM_STATE
    sample_size = args.sample_size or SAMPLE_SIZE
    random_state = args.random_state or RANDOM_STATE

    print("Subword / FastText approach (Person 5)")
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
        reports.append(run_fasttext_pipeline(data, VARIETY_CODES, random_state))
    if args.pipeline in ("bpe", "both"):
        reports.append(run_bpe_pipeline(data, VARIETY_CODES, random_state))

    raw = [
        {
            "pipeline": r["pipeline"],
            "n_varieties": len(VARIETY_CODES),
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
        shape_key = "embedding_shape" if r["pipeline"] == "fasttext" else "tfidf_shape"
        print(f"  {r['pipeline']:>9}: shape={r[shape_key]}  "
              f"sil_family={r['silhouette_family']:+.4f}  "
              f"sil_romance={r['silhouette_romance_vs_rest']:+.4f}")
    print(f"\nResults directory: {results_subdir('').parent}")


if __name__ == "__main__":
    main()
