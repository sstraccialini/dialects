"""
Build sentence- and variety-level embeddings from a trained Word2Vec.

- Sentence embedding = mean of in-vocab token vectors (None if all OOV).
- Variety embedding = L2-normalised mean of sentence vectors for that variety.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from gensim.models import Word2Vec

from .config import VARIETY_CODES
from .preprocess import tokenize


def embed_sentence(model: Word2Vec, tokens: List[str]) -> np.ndarray | None:
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def embed_corpus(
    model: Word2Vec,
    tokenised_sents: List[List[str]],
    sentence_codes: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """Embed every sentence; drops sentences with zero in-vocab tokens."""
    D = model.vector_size
    rows: List[np.ndarray] = []
    codes_out: List[str] = []
    for toks, code in zip(tokenised_sents, sentence_codes):
        v = embed_sentence(model, toks)
        if v is None:
            continue
        rows.append(v)
        codes_out.append(code)
    if not rows:
        return np.zeros((0, D), dtype=np.float32), []
    return np.vstack(rows).astype(np.float32), codes_out


def _l2_normalise(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return v / norm


def aggregate_variety_vectors(
    sentence_vectors: np.ndarray,
    sentence_codes: List[str],
    codes: List[str] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Per-variety vector = L2-normalised mean of its sentence vectors."""
    if codes is None:
        codes = VARIETY_CODES
    codes_ordered: List[str] = []
    rows = []
    arr_codes = np.asarray(sentence_codes)
    for code in codes:
        mask = (arr_codes == code)
        if mask.sum() == 0:
            continue
        rows.append(sentence_vectors[mask].mean(axis=0))
        codes_ordered.append(code)
    X = np.vstack(rows).astype(np.float32)
    X = _l2_normalise(X)
    return X, codes_ordered


def embed_data_per_sentence(
    model: Word2Vec,
    data: Dict[str, List[str]],
    codes: List[str] = None,
) -> Dict[str, np.ndarray]:
    """For parallel-alignment eval: return {code: (N, D) sentence vectors}.

    Sentences with zero in-vocab tokens become a zero vector so alignment
    indexing is preserved.
    """
    if codes is None:
        codes = VARIETY_CODES
    D = model.vector_size
    out: Dict[str, np.ndarray] = {}
    for code in codes:
        if code not in data:
            continue
        rows = []
        for s in data[code]:
            toks = tokenize(s)
            v = embed_sentence(model, toks) if toks else None
            if v is None:
                v = np.zeros(D, dtype=np.float32)
            rows.append(v)
        out[code] = np.vstack(rows).astype(np.float32) if rows else np.zeros((0, D), dtype=np.float32)
    return out
