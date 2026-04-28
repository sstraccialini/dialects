"""
FastText subword embedding pipeline (Person 5).

Training strategy: one shared model on all 14 varieties.

Why a shared space?
  Each variety has ~16k sentences. A per-variety model would be
  severely undertrained, especially for low-resource dialects.
  A shared model lets subword n-grams generalize across morphologically
  related forms (e.g. "parlare" / "parrà" / "parlari").

Variety representation:
  sentence vector  = mean of its token vectors
  variety vector   = mean of its sentence vectors
  → one (FT_VECTOR_SIZE,) vector per variety

FastText handles OOV tokens automatically via the subword hash table,
so even hapax legomena in a dialect receive a reasonable embedding.

Saved artefacts:
  results/models/fasttext_model.bin   — gensim FastText model
  results/fasttext/variety_vectors.csv  — 14 × FT_VECTOR_SIZE matrix
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

from gensim.models import FastText

from config import (
    FT_VECTOR_SIZE, FT_WINDOW, FT_MIN_COUNT,
    FT_MIN_N, FT_MAX_N, FT_EPOCHS, FT_SG, FT_WORKERS,
    RANDOM_STATE, MODELS_DIR, outputs_subdir,
)
from preprocess import preprocess_for_subword


def _tokenize(text: str) -> List[str]:
    """Whitespace tokenizer applied after preprocessing."""
    return text.split()


def _prepare_sentences(
    data: Dict[str, List[str]],
    codes: List[str],
) -> List[List[str]]:
    """
    Flatten all varieties into a single list of tokenized sentences
    for FastText training.
    """
    all_sentences: List[List[str]] = []
    for code in codes:
        for sent in data[code]:
            tokens = _tokenize(preprocess_for_subword(sent))
            if tokens:
                all_sentences.append(tokens)
    return all_sentences


def train_fasttext(
    data: Dict[str, List[str]],
    codes: List[str],
    save: bool = True,
) -> FastText:
    """
    Train a gensim FastText model on all variety sentences.

    Parameters
    ----------
    data : {code: list[str]} sampled sentences
    codes: ordered variety codes
    save : if True, persist the model to MODELS_DIR/fasttext_model.bin

    Returns
    -------
    Trained gensim FastText model.
    """
    print("  Preparing sentences for FastText training...")
    sentences = _prepare_sentences(data, codes)
    print(f"  Total sentences: {len(sentences):,}")

    print("  Training FastText model (this may take a few minutes)...")
    model = FastText(
        sentences=sentences,
        vector_size=FT_VECTOR_SIZE,
        window=FT_WINDOW,
        min_count=FT_MIN_COUNT,
        min_n=FT_MIN_N,
        max_n=FT_MAX_N,
        sg=FT_SG,
        workers=FT_WORKERS,
        epochs=FT_EPOCHS,
        seed=RANDOM_STATE,
    )
    print(f"  Vocab size: {len(model.wv):,}  |  Vector dim: {model.vector_size}")

    if save:
        path = Path(MODELS_DIR) / "fasttext_model.bin"
        model.save(str(path))
        print(f"  Model saved → {path}")

    return model


def _embed_sentence(model: FastText, tokens: List[str]) -> np.ndarray:
    """Average word vectors for a tokenized sentence (handles OOV via subwords)."""
    if not tokens:
        return np.zeros(model.vector_size, dtype=np.float32)
    # gensim FastText: model.wv[token] returns a subword-composed vector
    # even for out-of-vocabulary tokens.
    vecs = np.array([model.wv[tok] for tok in tokens], dtype=np.float32)
    return vecs.mean(axis=0)


def variety_embeddings(
    model: FastText,
    data: Dict[str, List[str]],
    codes: List[str],
    save: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute one embedding per variety by mean-pooling sentence vectors.

    Returns
    -------
    X     : (n_varieties, FT_VECTOR_SIZE) float32 array
    codes : same order as X rows
    """
    print("  Computing variety embeddings (mean-pooling sentences)...")
    embeddings = []
    for code in codes:
        sent_vecs = []
        for sent in data[code]:
            tokens = _tokenize(preprocess_for_subword(sent))
            if tokens:
                sent_vecs.append(_embed_sentence(model, tokens))
        if sent_vecs:
            variety_vec = np.mean(sent_vecs, axis=0).astype(np.float32)
        else:
            variety_vec = np.zeros(model.vector_size, dtype=np.float32)
        embeddings.append(variety_vec)

    X = np.array(embeddings, dtype=np.float32)
    print(f"  Variety embedding matrix: {X.shape}")

    if save:
        from config import VARIETY_NAMES
        df = pd.DataFrame(X, index=codes)
        out = outputs_subdir("fasttext") / "variety_vectors.csv"
        df.to_csv(out, float_format="%.6f")
        print(f"  Variety vectors saved → {out}")

    return X, list(codes)
