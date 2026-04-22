"""
Build sentence- and variety-level embeddings from a trained Word2Vec.

Sentence embedding = mean of in-vocabulary token vectors.
    - A sentence whose tokens are ALL OOV produces no vector and is
      skipped. This is the expected Word2Vec behaviour (FastText would
      still get a vector via subwords).

Variety embedding = mean of sentence embeddings for that variety,
L2-normalised (so cosine distance becomes numerically well-behaved).

Outputs saved to:
    results/sentence_vectors.npz   (vectors + aligned variety codes)
    results/variety_vectors.csv    (16 x D, row-indexed by code)
    results/variety_vectors.npz    (same content in npz for downstream)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from config import VARIETY_CODES, results_subdir
from preprocess import tokenize


def embed_sentence(model: Word2Vec, tokens: List[str]) -> np.ndarray | None:
    """Mean of in-vocab token vectors; None if no token is in vocab."""
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def embed_corpus(
    model: Word2Vec,
    tokenised_sents: List[List[str]],
    sentence_codes: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Embed every sentence. Drops sentences with zero in-vocab tokens.

    Returns (vectors[n_valid x D], aligned_codes[n_valid]).
    """
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
    codes: List[str] = VARIETY_CODES,
) -> Tuple[np.ndarray, List[str]]:
    """
    Per-variety vector = mean of its sentence vectors, then L2-normalised.

    Keeps the canonical variety order (drops codes with 0 valid sentences).
    """
    codes_ordered: List[str] = []
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
    sentence_codes: List[str],
) -> str:
    out = results_subdir("") / "sentence_vectors.npz"
    np.savez_compressed(
        out,
        vectors=sentence_vectors.astype(np.float32),
        codes=np.asarray(sentence_codes),
    )
    return str(out)


def save_variety_vectors(X: np.ndarray, codes: List[str]) -> Dict[str, str]:
    """Save as both CSV (human-readable) and NPZ (downstream)."""
    out_csv = results_subdir("") / "variety_vectors.csv"
    pd.DataFrame(X, index=codes).to_csv(out_csv, float_format="%.6f")

    out_npz = results_subdir("") / "variety_vectors.npz"
    np.savez_compressed(out_npz, matrix=X.astype(np.float32), labels=np.asarray(codes))
    return {"csv": str(out_csv), "npz": str(out_npz)}
