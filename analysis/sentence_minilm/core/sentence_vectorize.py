"""
Sentence-embedding vectorizer.

Two pipelines:

  - fit_transform_sentence:           sentence → variety mean. Used on
                                      flat corpora (FLORES, OLDI).

  - fit_transform_sentence_by_article: sentence → article mean → variety
                                      mean. Two-stage; used on Wiki where
                                      article structure matters.

Returns (X, model) for both. Backbone is parameterised via `model_name`
so subclasses (e.g. LaBSE) can reuse this module unchanged.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import SENTENCE_MODEL as DEFAULT_MODEL


def load_sentence_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def encode_sentences(
    sentences: List[str],
    model: SentenceTransformer,
    batch_size: int = 32,
    show_progress_bar: bool = False,
) -> np.ndarray:
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
    if sentence_embeddings.ndim != 2:
        raise ValueError("sentence_embeddings must be 2D")
    if method == "mean":
        return sentence_embeddings.mean(axis=0)
    if method == "median":
        return np.median(sentence_embeddings, axis=0)
    raise ValueError(f"Unknown aggregation method: {method}")


def fit_transform_sentence(
    data: Dict[str, List[str]],
    codes: List[str],
    model: SentenceTransformer = None,
    model_name: str = DEFAULT_MODEL,
    aggregation: str = "mean",
    batch_size: int = 32,
) -> Tuple[np.ndarray, SentenceTransformer]:
    """Flat: sentence → variety mean."""
    if model is None:
        model = load_sentence_model(model_name)
    variety_vectors = []
    out_codes = []
    for code in tqdm(codes, desc="Encoding varieties"):
        if code not in data:
            continue
        sentences = data[code]
        if not sentences:
            continue
        sent_emb = encode_sentences(sentences, model=model, batch_size=batch_size)
        variety_vectors.append(aggregate_sentence_embeddings(sent_emb, method=aggregation))
        out_codes.append(code)
    return np.vstack(variety_vectors), model


def fit_transform_sentence_by_article(
    data: Dict[str, "pd.DataFrame"],
    codes: List[str],
    model: SentenceTransformer = None,
    model_name: str = DEFAULT_MODEL,
    article_aggregation: str = "mean",
    variety_aggregation: str = "mean",
    batch_size: int = 32,
) -> Tuple[np.ndarray, SentenceTransformer]:
    """Two-stage: sentence → article → variety mean."""
    if model is None:
        model = load_sentence_model(model_name)
    variety_vectors = []
    out_codes = []
    for code in tqdm(codes, desc="Encoding varieties"):
        if code not in data:
            continue
        df = data[code]
        if df.empty:
            continue
        sentences = df["text"].tolist()
        sent_emb = encode_sentences(sentences, model=model, batch_size=batch_size)
        tmp = df.copy()
        tmp["embedding"] = list(sent_emb)
        article_vectors = []
        for _, group in tmp.groupby("article_id"):
            arr = np.vstack(group["embedding"].to_list())
            article_vectors.append(aggregate_sentence_embeddings(arr, method=article_aggregation))
        article_matrix = np.vstack(article_vectors)
        variety_vectors.append(aggregate_sentence_embeddings(article_matrix, method=variety_aggregation))
        out_codes.append(code)
    return np.vstack(variety_vectors), model


def encode_per_sentence(
    data: Dict[str, List[str]],
    codes: List[str],
    model: SentenceTransformer = None,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
) -> Tuple[Dict[str, np.ndarray], SentenceTransformer]:
    """For parallel-alignment eval: {code: (N, D)} embeddings."""
    if model is None:
        model = load_sentence_model(model_name)
    out: Dict[str, np.ndarray] = {}
    for code in tqdm(codes, desc="Encoding (per sentence)"):
        if code not in data:
            continue
        sents = data[code]
        emb = encode_sentences(sents, model=model, batch_size=batch_size)
        out[code] = emb.astype(np.float32)
    return out, model


def matrix_to_dict(X: np.ndarray, codes: List[str]) -> Dict[str, np.ndarray]:
    return {code: X[i] for i, code in enumerate(codes)}
