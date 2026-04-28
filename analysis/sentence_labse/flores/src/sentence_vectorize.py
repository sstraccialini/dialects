"""
Sentence-embedding vectorizer (FLORES+).

End-to-end pipeline:
    1. encode every sampled sentence through a multilingual
       sentence-transformer;
    2. average the (L2-normalised) embeddings to obtain one vector
       per variety.

There is no article-level aggregation here because FLORES+ has no
article structure — every row is an independent parallel sentence.
FLORES+ parallelism is itself a strong anchor across varieties.

The default backbone is paraphrase-multilingual-MiniLM-L12-v2:
compact (~118M params), fast on CPU, covers 50+ languages with a
contrastive bi-encoder objective.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
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
    End-to-end sentence-embedding pipeline.

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


def matrix_to_dict(X: np.ndarray, codes: List[str]) -> Dict[str, np.ndarray]:
    """Map each variety code to its embedding vector."""
    return {code: X[i] for i, code in enumerate(codes)}


if __name__ == "__main__":
    from data_loader import load_all_varieties
    from config import VARIETY_CODES

    print("Loading FLORES+ sentences...")
    data, stats = load_all_varieties(verbose=True)

    codes = [c for c in VARIETY_CODES if c in data]

    print("\nBuilding sentence-embedding baseline...")
    X, model = fit_transform_sentence(
        data=data,
        codes=codes,
        model_name=DEFAULT_MODEL,
        aggregation="mean",
        batch_size=32,
    )

    print("\nDone.")
    print("Matrix shape:", X.shape)
