"""
Sentence-embedding vectorizer.

Two end-to-end pipelines are provided:

- fit_transform_sentence
    Encode every sampled sentence through a multilingual
    sentence-transformer, then average the embeddings per variety.
    Useful when we do not have any document structure (e.g. FLORES+).

- fit_transform_sentence_by_article
    Encode every sampled sentence, then
    (1) average sentence embeddings inside each `article_id`,
    (2) average article embeddings across the variety.
    Two-stage aggregation gives each article equal weight regardless
    of the number of sentences it produced, which is more robust
    on Wikipedia data where article length varies widely.

Both functions return (X, model) where X has shape
(n_varieties, hidden_size) and rows follow `codes` order.

The default backbone is paraphrase-multilingual-MiniLM-L12-v2:
compact (~118M params), fast on CPU, covers 50+ languages with a
contrastive bi-encoder objective.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import SENTENCE_MODEL as DEFAULT_MODEL


def load_sentence_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """Load a pretrained sentence-transformer."""
    return SentenceTransformer(model_name)


def encode_sentences(
    sentences: List[str],
    model: SentenceTransformer,
    batch_size: int = 32,
    show_progress_bar: bool = True,
) -> np.ndarray:
    """
    Encode a list of sentences into a dense matrix:
        shape = (n_sentences, hidden_size)

    Embeddings are l2-normalized so downstream cosine similarity is
    well-defined and insensitive to sentence length.
    """
    return model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def aggregate_sentence_embeddings(
    sentence_embeddings: np.ndarray,
    method: str = "mean",
) -> np.ndarray:
    """Aggregate a stack of sentence embeddings into a single vector."""
    if sentence_embeddings.ndim != 2:
        raise ValueError("sentence_embeddings must be 2D: (n_sentences, hidden_size)")

    if method == "mean":
        return sentence_embeddings.mean(axis=0)
    if method == "median":
        return np.median(sentence_embeddings, axis=0)

    raise ValueError(f"Unknown aggregation method: {method}")


def fit_transform_sentence(
    data: Dict[str, List[str]],
    codes: List[str],
    model_name: str = DEFAULT_MODEL,
    aggregation: str = "mean",
    batch_size: int = 32,
) -> Tuple[np.ndarray, SentenceTransformer]:
    """
    End-to-end flat sentence-embedding pipeline.

    Input:
        data  = dict {code: list_of_sentences}
        codes = ordered list of variety codes
    Output:
        X     = dense matrix (n_varieties, hidden_size)
        model = loaded sentence-transformer
    """
    model = load_sentence_model(model_name)

    variety_vectors = []
    for code in tqdm(codes, desc="Encoding varieties"):
        if code not in data:
            raise ValueError(f"Missing variety '{code}' in data")
        sentences = data[code]
        if not sentences:
            raise ValueError(f"No sentences found for variety '{code}'")

        sent_emb = encode_sentences(
            sentences, model=model,
            batch_size=batch_size, show_progress_bar=False,
        )
        variety_vec = aggregate_sentence_embeddings(sent_emb, method=aggregation)
        variety_vectors.append(variety_vec)

    X = np.vstack(variety_vectors)
    return X, model


def fit_transform_sentence_by_article(
    data: Dict[str, "pd.DataFrame"],
    codes: List[str],
    model_name: str = DEFAULT_MODEL,
    article_aggregation: str = "mean",
    variety_aggregation: str = "mean",
    batch_size: int = 32,
) -> Tuple[np.ndarray, SentenceTransformer]:
    """
    Article-aware end-to-end sentence-embedding pipeline.

    Input:
        data  = dict {code: DataFrame(text, article_id)}
        codes = ordered list of variety codes

    For each variety:
        1. encode all sentences
        2. average sentence embeddings within each `article_id`
        3. average article embeddings across the variety
    """
    model = load_sentence_model(model_name)

    variety_vectors = []
    for code in tqdm(codes, desc="Encoding varieties"):
        if code not in data:
            raise ValueError(f"Missing variety '{code}' in data")
        df = data[code]
        if df.empty:
            raise ValueError(f"No rows found for variety '{code}'")

        sentences = df["text"].tolist()
        sent_emb = encode_sentences(
            sentences, model=model,
            batch_size=batch_size, show_progress_bar=False,
        )

        tmp = df.copy()
        tmp["embedding"] = list(sent_emb)

        article_vectors = []
        for _, group in tmp.groupby("article_id"):
            arr = np.vstack(group["embedding"].to_list())
            article_vec = aggregate_sentence_embeddings(arr, method=article_aggregation)
            article_vectors.append(article_vec)

        article_matrix = np.vstack(article_vectors)
        variety_vec = aggregate_sentence_embeddings(
            article_matrix, method=variety_aggregation,
        )
        variety_vectors.append(variety_vec)

    X = np.vstack(variety_vectors)
    return X, model


def matrix_to_dict(X: np.ndarray, codes: List[str]) -> Dict[str, np.ndarray]:
    """Map each variety code to its embedding vector."""
    return {code: X[i] for i, code in enumerate(codes)}


if __name__ == "__main__":
    from data_loader import load_all_varieties_with_article_ids
    from config import VARIETY_CODES

    print("Loading sampled rows with article_id...")
    data, stats = load_all_varieties_with_article_ids(verbose=True)

    codes = [c for c in VARIETY_CODES if c in data]

    print("\nBuilding sentence-embedding baseline (by article)...")
    X, model = fit_transform_sentence_by_article(
        data=data,
        codes=codes,
        model_name=DEFAULT_MODEL,
        article_aggregation="mean",
        variety_aggregation="mean",
        batch_size=32,
    )

    print("\nDone.")
    print("Matrix shape:", X.shape)
