"""
Sentence-embedding baseline on Wikipedia data (multilingual MiniLM).

Pipeline:
    1. load per-variety CSVs (text, article_id) from Dataset/wiki/
    2. encode sentences with a multilingual sentence-transformer
    3. average sentence embeddings per article, then per variety
    4. dialect-vs-modern-language similarity table + rankings (method-specific)
    5. central evaluation: distances, dendrogram, projections, silhouette, ...
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
for p in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import (
    VARIETY_CODES, VARIETY_GROUP, VARIETY_NAMES,
    GROUP_NAMES, GROUP_COLORS,
    DIALECT_CODES, MODERN_LANGUAGE_CODES,
    SAMPLE_SIZE, SENTENCE_MODEL, SENTENCE_BATCH_SIZE,
    ARTICLE_AGGREGATION, VARIETY_AGGREGATION,
    outputs_subdir, evaluation_subdir,
)
from data_loader import load_all_varieties_with_article_ids
from sentence_vectorize import fit_transform_sentence_by_article, matrix_to_dict

from evaluation.evaluation import run_evaluation


VARIANT = "sentence"
ROMANCE_FAMILIES = {"italo_romance", "italian", "romance"}


def dialect_vs_language_similarity_table(X: np.ndarray, codes: list) -> pd.DataFrame:
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
            row[lang] = float(cosine_similarity(dvec, lvec)[0, 0])
        rows.append(row)
    return pd.DataFrame(rows)


def dialect_language_rankings(sim_df: pd.DataFrame) -> pd.DataFrame:
    long_rows = []
    for _, row in sim_df.iterrows():
        scores = []
        for lang in MODERN_LANGUAGE_CODES:
            if lang in row and pd.notna(row[lang]):
                scores.append((lang, VARIETY_NAMES[lang], float(row[lang])))
        scores.sort(key=lambda x: x[2], reverse=True)
        for rank, (lang_code, lang_name, score) in enumerate(scores, start=1):
            long_rows.append({
                "dialect_code": row["dialect_code"],
                "dialect_name": row["dialect_name"],
                "rank": rank,
                "language_code": lang_code,
                "language_name": lang_name,
                "cosine_similarity": score,
            })
    return pd.DataFrame(long_rows)


def main():
    target_dialects = DIALECT_CODES
    target_languages = MODERN_LANGUAGE_CODES
    quick_codes = list(target_dialects) + list(target_languages)

    print(f"Loading {len(quick_codes)} varieties (sample_size={SAMPLE_SIZE})...")
    data, _ = load_all_varieties_with_article_ids(
        codes=quick_codes,
        sample_size=SAMPLE_SIZE,
        verbose=True,
    )

    codes = [c for c in quick_codes if c in data]

    print(f"\nEncoding with {SENTENCE_MODEL}...")
    X, _ = fit_transform_sentence_by_article(
        data, codes,
        model_name=SENTENCE_MODEL,
        article_aggregation=ARTICLE_AGGREGATION,
        variety_aggregation=VARIETY_AGGREGATION,
        batch_size=SENTENCE_BATCH_SIZE,
    )

    out_dir = outputs_subdir(VARIANT)
    np.savez_compressed(
        out_dir / "variety_vectors.npz",
        matrix=X.astype(np.float32),
        labels=np.asarray(codes),
        model_name=np.asarray(SENTENCE_MODEL),
    )
    pd.DataFrame(X, index=codes).to_csv(out_dir / "variety_vectors.csv",
                                        float_format="%.6f")

    sim_df = dialect_vs_language_similarity_table(X, codes)
    eval_dir = evaluation_subdir(VARIANT)
    sim_out = eval_dir / "dialect_vs_language_similarity.csv"
    rank_out = eval_dir / "dialect_vs_language_rankings.csv"
    sim_df.to_csv(sim_out, index=False)
    dialect_language_rankings(sim_df).to_csv(rank_out, index=False)

    print("\n--- Central evaluation ---")
    report = run_evaluation(
        variety_vectors=X,
        variety_codes=codes,
        out_dir=eval_dir,
        method_label=f"sentence ({SENTENCE_MODEL})",
        family_groups=VARIETY_GROUP,
        family_colors=GROUP_COLORS,
        family_display_names=GROUP_NAMES,
        display_names=VARIETY_NAMES,
        romance_families=ROMANCE_FAMILIES,
    )

    print(f"\nSaved similarity table: {sim_out}")
    print(f"Saved ranking table:    {rank_out}")
    sf = report["silhouette_family"]
    sr = report["silhouette_romance_vs_rest"]
    print(f"sil_family={sf:+.4f}  sil_romance={sr:+.4f}")
    print(f"\nMethod outputs:       {outputs_subdir()}")
    print(f"Evaluation artefacts: {evaluation_subdir()}")


if __name__ == "__main__":
    main()
