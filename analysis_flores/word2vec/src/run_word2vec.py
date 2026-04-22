"""
End-to-end orchestrator for the Word2Vec approach on FLORES+.

Launch (from repo root, with venv active):
    python analysis_flores/word2vec/src/run_word2vec.py

Options:
    --sample-size N     sentences per variety (default: config.SAMPLE_SIZE = 2009)
    --random-state N    seed (default: config.RANDOM_STATE)

Produces (in analysis_flores/word2vec/results/):
    models/word2vec.model            trained gensim model
    sentence_vectors.npz             (n_sent x D) + aligned codes
    variety_vectors.csv  / .npz      (16 x D), variety-level vectors
    distances.csv                    (16 x 16 cosine distance matrix)
    nearest_neighbors.csv            top-3 neighbours per variety
    dendrogram.png                   hierarchical clustering
    projection_mds.png / tsne.png    2D projections
    silhouette_report.txt            silhouette scores (family + romance)
    run_stats.csv                    per-variety sentence counts
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
from train import build_tokenised_corpus, train_word2vec, save_word2vec
from build_vectors import (
    embed_corpus, aggregate_variety_vectors,
    save_sentence_vectors, save_variety_vectors,
)
from similarity import (
    cosine_distance_matrix, save_distance_matrix,
    nearest_neighbors, save_nearest_neighbors,
)
from cluster import cluster_pipeline, save_silhouette_report
from visualize import visualize_pipeline


def main():
    parser = argparse.ArgumentParser(description="Word2Vec approach orchestrator (FLORES+)")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="sentences per variety (default: config.SAMPLE_SIZE)")
    parser.add_argument("--random-state", type=int, default=None,
                        help="seed (default: config.RANDOM_STATE)")
    args = parser.parse_args()

    from config import SAMPLE_SIZE, RANDOM_STATE
    sample_size = args.sample_size or SAMPLE_SIZE
    random_state = args.random_state if args.random_state is not None else RANDOM_STATE

    print("Word2Vec approach on FLORES+")
    print("=" * 60)
    print(f"  sample_size   = {sample_size}")
    print(f"  random_state  = {random_state}")
    print(f"  varieties     = {VARIETY_CODES}")
    print()

    print("Loading + sampling...")
    data, stats = load_all_varieties(
        sample_size=sample_size,
        random_state=random_state,
    )
    stats["sample_size_param"] = sample_size
    stats["random_state"] = random_state
    stats.to_csv(results_subdir("") / "run_stats.csv", index=False)

    print("\n--- Building tokenised corpus ---")
    tokenised, sentence_codes = build_tokenised_corpus(data)
    total_tokens = sum(len(s) for s in tokenised)
    print(f"  {len(tokenised)} sentences, {total_tokens:,} tokens")

    print("\n--- Training Word2Vec (shared, skip-gram) ---")
    model = train_word2vec(tokenised)
    model_path = save_word2vec(model)
    print(f"  Saved: {model_path}")

    print("\n--- Embedding sentences + aggregating per variety ---")
    sent_vecs, sent_codes_out = embed_corpus(model, tokenised, sentence_codes)
    print(f"  sentence_vectors: shape={sent_vecs.shape} "
          f"(dropped {len(tokenised) - sent_vecs.shape[0]} fully-OOV sentences)")

    X, codes = aggregate_variety_vectors(sent_vecs, sent_codes_out)
    print(f"  variety_vectors : shape={X.shape}, codes={codes}")

    sv_path = save_sentence_vectors(sent_vecs, sent_codes_out)
    vv_paths = save_variety_vectors(X, codes)
    print(f"  Saved: {sv_path}")
    print(f"  Saved: {vv_paths['csv']}")
    print(f"  Saved: {vv_paths['npz']}")

    print("\n--- Cosine distance + clustering + projections ---")
    dist = cosine_distance_matrix(X)
    save_distance_matrix(dist, codes)
    save_nearest_neighbors(nearest_neighbors(dist, codes, k=3))
    cluster_report = cluster_pipeline(dist, codes)
    viz_paths = visualize_pipeline(dist, codes)

    save_silhouette_report([{
        "pipeline": "word2vec",
        "n_varieties": cluster_report["n_varieties"],
        "silhouette_family": cluster_report["silhouette_family"],
        "silhouette_romance_vs_rest": cluster_report["silhouette_romance_vs_rest"],
        "dendrogram_path": cluster_report["dendrogram_path"],
    }])

    print("\n" + "=" * 60)
    print("Done. Summary:")
    print(f"  word2vec: X.shape={X.shape}  "
          f"sil_family={cluster_report['silhouette_family']:+.4f}  "
          f"sil_romance={cluster_report['silhouette_romance_vs_rest']:+.4f}")
    print(f"\nResults directory: {results_subdir('')}")


if __name__ == "__main__":
    main()
