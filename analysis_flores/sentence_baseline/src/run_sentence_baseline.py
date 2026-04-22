"""
Sentence-embedding baseline on FLORES+.

End-to-end pipeline:
    1. load per-variety .txt files from flores_data/flores_plus/
    2. encode sentences with a multilingual sentence-transformer
    3. average sentence embeddings to produce one vector per variety
    4. compute cosine distance matrix across varieties
    5. cluster (hierarchical, average linkage) + silhouette
    6. 2D projections (MDS, t-SNE)
    7. save a dialect-vs-modern-language similarity table + rankings

Outputs land in `results/sentence/` (plus `results/shared/` for the
silhouette report).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    VARIETY_CODES,
    DIALECT_CODES,
    MODERN_LANGUAGE_CODES,
    VARIETY_NAMES,
    SAMPLE_SIZE,
    SENTENCE_MODEL,
    SENTENCE_BATCH_SIZE,
    VARIETY_AGGREGATION,
    results_subdir,
)
from data_loader import load_all_varieties
from sentence_vectorize import fit_transform_sentence, matrix_to_dict
from similarity import (
    cosine_distance_matrix,
    save_distance_matrix,
    nearest_neighbors,
    save_nearest_neighbors,
)
from cluster import cluster_pipeline, save_silhouette_report
from visualize import visualize_pipeline


def dialect_vs_language_similarity_table(X: np.ndarray, codes: list) -> pd.DataFrame:
    """
    Build a wide table where rows = Italo-Romance dialects and
    columns = modern languages. Values are cosine similarities between
    variety-level sentence embeddings.
    """
    vecs = matrix_to_dict(X, codes)

    rows = []
    for d in DIALECT_CODES:
        if d not in vecs:
            continue

        row = {"dialect_code": d, "dialect_name": VARIETY_NAMES[d]}
        dvec = vecs[d].reshape(1, -1)

        for lang in MODERN_LANGUAGE_CODES:
            if lang not in vecs:
                continue
            lvec = vecs[lang].reshape(1, -1)
            sim = cosine_similarity(dvec, lvec)[0, 0]
            row[lang] = float(sim)

        rows.append(row)

    return pd.DataFrame(rows)


def dialect_language_rankings(sim_df: pd.DataFrame) -> pd.DataFrame:
    """Convert the wide similarity table into a ranked long-format table."""
    long_rows = []

    for _, row in sim_df.iterrows():
        dialect_code = row["dialect_code"]
        dialect_name = row["dialect_name"]

        scores = []
        for lang in MODERN_LANGUAGE_CODES:
            if lang in row and pd.notna(row[lang]):
                scores.append((lang, VARIETY_NAMES[lang], float(row[lang])))

        scores = sorted(scores, key=lambda x: x[2], reverse=True)

        for rank, (lang_code, lang_name, score) in enumerate(scores, start=1):
            long_rows.append({
                "dialect_code": dialect_code,
                "dialect_name": dialect_name,
                "rank": rank,
                "language_code": lang_code,
                "language_name": lang_name,
                "cosine_similarity": score,
            })

    return pd.DataFrame(long_rows)


def main():
    print(f"Loading {len(VARIETY_CODES)} varieties from FLORES+ "
          f"(sample_size={SAMPLE_SIZE})...")
    data, _ = load_all_varieties(
        codes=VARIETY_CODES,
        sample_size=SAMPLE_SIZE,
        verbose=True,
    )

    codes = [c for c in VARIETY_CODES if c in data]

    print(f"\nEncoding with {SENTENCE_MODEL}...")
    X, _ = fit_transform_sentence(
        data, codes,
        model_name=SENTENCE_MODEL,
        aggregation=VARIETY_AGGREGATION,
        batch_size=SENTENCE_BATCH_SIZE,
    )

    dist = cosine_distance_matrix(X)

    sim_df = dialect_vs_language_similarity_table(X, codes)
    sim_out = results_subdir("sentence") / "dialect_vs_language_similarity.csv"
    sim_df.to_csv(sim_out, index=False)

    rank_df = dialect_language_rankings(sim_df)
    rank_out = results_subdir("sentence") / "dialect_vs_language_rankings.csv"
    rank_df.to_csv(rank_out, index=False)

    save_distance_matrix(dist, codes, pipeline="sentence")
    save_nearest_neighbors(nearest_neighbors(dist, codes, k=3), pipeline="sentence")

    rep = cluster_pipeline(dist, codes, "sentence")
    visualize_pipeline(dist, codes, "sentence")

    save_silhouette_report([{
        "pipeline": "sentence",
        "n_varieties": len(codes),
        "silhouette_family": rep["silhouette_family"],
        "silhouette_romance_vs_rest": rep["silhouette_romance_vs_rest"],
        "dendrogram_path": rep["dendrogram_path"],
    }])

    print(f"\nSaved similarity table: {sim_out}")
    print(f"Saved ranking table:    {rank_out}")
    print("Sentence baseline completed.")


if __name__ == "__main__":
    main()
