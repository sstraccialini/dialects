"""
End-to-end orchestrator for the multilingual embeddings approach on
FLORES+.

Launch (from repo root, with venv active):
    python analysis_flores/multilingual/src/run_pipeline.py

Options:
    --model-name NAME    any AutoModel-compatible HF id (default: config.DEFAULT_MODEL_NAME = xlm-roberta-base)
    --sample-size N      sentences per variety (default: config.SAMPLE_SIZE = 2009)
    --random-state N     seed (default: config.RANDOM_STATE)
    --batch-size N       batch size for inference (default: config.BATCH_SIZE = 32)
    --device DEVICE      'cuda', 'mps', or 'cpu' (default: auto)

Produces (in analysis_flores/multilingual/results/):
    run_stats.csv                    per-variety sentence counts
    sentence_vectors.npz             (n_sent x D) + aligned codes + model name
    variety_vectors.csv  / .npz      (16 x D), variety-level vectors
    distances.csv                    (16 x 16) cosine distance matrix
    nearest_neighbors.csv            top-3 neighbours per variety
    dendrogram.png                   hierarchical clustering
    projection_mds.png / tsne.png    2D projections
    silhouette_report.txt            silhouette scores (family + romance)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make sibling modules importable when running the file directly.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import (
    VARIETY_CODES, DEFAULT_MODEL_NAME, MAX_LENGTH, BATCH_SIZE, results_subdir,
)
from data_loader import load_all_varieties, iter_labeled_sentences
from embedder import MultilingualEmbedder
from similarity import (
    cosine_distance_matrix, save_distance_matrix,
    nearest_neighbors, save_nearest_neighbors,
)
from cluster import cluster_pipeline, save_silhouette_report
from visualize import visualize_pipeline


def _l2_normalise(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return v / norm


def aggregate_variety_vectors(
    sentence_vectors: np.ndarray,
    sentence_codes,
    codes = VARIETY_CODES,
):
    """
    Per-variety vector = mean of its (unit-length) sentence vectors,
    then re-L2-normalised. Order follows the canonical variety order.
    """
    codes_ordered = []
    rows = []
    arr_codes = np.asarray(sentence_codes)
    for slug in codes:
        mask = (arr_codes == slug)
        if mask.sum() == 0:
            continue
        rows.append(sentence_vectors[mask].mean(axis=0))
        codes_ordered.append(slug)
    X = np.vstack(rows).astype(np.float32)
    X = _l2_normalise(X)
    return X, codes_ordered


def save_sentence_vectors(
    sentence_vectors: np.ndarray,
    sentence_codes,
    model_name: str,
) -> str:
    out = results_subdir("") / "sentence_vectors.npz"
    np.savez_compressed(
        out,
        vectors=sentence_vectors.astype(np.float32),
        codes=np.asarray(sentence_codes),
        model_name=np.asarray(model_name),
    )
    return str(out)


def save_variety_vectors(X: np.ndarray, codes, model_name: str) -> dict:
    out_csv = results_subdir("") / "variety_vectors.csv"
    pd.DataFrame(X, index=codes).to_csv(out_csv, float_format="%.6f")
    out_npz = results_subdir("") / "variety_vectors.npz"
    np.savez_compressed(
        out_npz,
        matrix=X.astype(np.float32),
        labels=np.asarray(codes),
        model_name=np.asarray(model_name),
    )
    return {"csv": str(out_csv), "npz": str(out_npz)}


def main():
    parser = argparse.ArgumentParser(
        description="Multilingual embeddings orchestrator (FLORES+)"
    )
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME,
                        help="HuggingFace model id (default: xlm-roberta-base)")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="sentences per variety (default: config.SAMPLE_SIZE)")
    parser.add_argument("--random-state", type=int, default=None,
                        help="seed (default: config.RANDOM_STATE)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"inference batch size (default: {BATCH_SIZE})")
    parser.add_argument("--device", type=str, default=None,
                        help="'cuda', 'mps', or 'cpu' (default: auto)")
    args = parser.parse_args()

    from config import SAMPLE_SIZE, RANDOM_STATE
    sample_size = args.sample_size or SAMPLE_SIZE
    random_state = args.random_state if args.random_state is not None else RANDOM_STATE

    print("Multilingual embeddings on FLORES+")
    print("=" * 60)
    print(f"  model_name    = {args.model_name}")
    print(f"  sample_size   = {sample_size}")
    print(f"  random_state  = {random_state}")
    print(f"  batch_size    = {args.batch_size}")
    print(f"  max_length    = {MAX_LENGTH}")
    print(f"  varieties     = {VARIETY_CODES}")
    print()

    print("Loading + sampling...")
    data, stats = load_all_varieties(
        sample_size=sample_size,
        random_state=random_state,
    )
    stats["sample_size_param"] = sample_size
    stats["random_state"] = random_state
    stats["model_name"] = args.model_name
    stats.to_csv(results_subdir("") / "run_stats.csv", index=False)

    sents, sent_codes = iter_labeled_sentences(data)
    print(f"  {len(sents)} sentences to embed")

    print("\n--- Loading model ---")
    embedder = MultilingualEmbedder(
        model_name=args.model_name,
        device=args.device,
        max_length=MAX_LENGTH,
    )

    print("\n--- Computing sentence embeddings ---")
    sent_vecs = embedder.encode(sents, batch_size=args.batch_size)
    print(f"  sentence_vectors: shape={sent_vecs.shape}")

    print("\n--- Aggregating per variety ---")
    X, codes = aggregate_variety_vectors(sent_vecs, sent_codes)
    print(f"  variety_vectors : shape={X.shape}, codes={codes}")

    sv_path = save_sentence_vectors(sent_vecs, sent_codes, args.model_name)
    vv_paths = save_variety_vectors(X, codes, args.model_name)
    print(f"  Saved: {sv_path}")
    print(f"  Saved: {vv_paths['csv']}")
    print(f"  Saved: {vv_paths['npz']}")

    print("\n--- Cosine distance + clustering + projections ---")
    dist = cosine_distance_matrix(X)
    save_distance_matrix(dist, codes)
    save_nearest_neighbors(nearest_neighbors(dist, codes, k=3))
    cluster_report = cluster_pipeline(dist, codes, model_name=args.model_name)
    visualize_pipeline(dist, codes, model_name=args.model_name)

    save_silhouette_report([{
        "pipeline": "multilingual",
        "model_name": args.model_name,
        "n_varieties": cluster_report["n_varieties"],
        "silhouette_family": cluster_report["silhouette_family"],
        "silhouette_romance_vs_rest": cluster_report["silhouette_romance_vs_rest"],
        "dendrogram_path": cluster_report["dendrogram_path"],
    }])

    print("\n" + "=" * 60)
    print("Done. Summary:")
    print(f"  multilingual ({args.model_name}): X.shape={X.shape}  "
          f"sil_family={cluster_report['silhouette_family']:+.4f}  "
          f"sil_romance={cluster_report['silhouette_romance_vs_rest']:+.4f}")
    print(f"\nResults directory: {results_subdir('')}")


if __name__ == "__main__":
    main()
