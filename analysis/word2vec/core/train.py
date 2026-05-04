"""
Train a single shared Word2Vec model on tokenised sentences.

Usage:
    model = train_word2vec(tokenised_sents)
    save_word2vec(model, path)
    model = load_word2vec(path)
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from gensim.models import Word2Vec

from .config import (
    W2V_VECTOR_SIZE,
    W2V_WINDOW,
    W2V_MIN_COUNT,
    W2V_SG,
    W2V_EPOCHS,
    W2V_WORKERS,
    W2V_SEED,
)


def train_word2vec(
    tokenised_sents: List[List[str]],
    vector_size: int = W2V_VECTOR_SIZE,
    window: int = W2V_WINDOW,
    min_count: int = W2V_MIN_COUNT,
    sg: int = W2V_SG,
    epochs: int = W2V_EPOCHS,
    workers: int = W2V_WORKERS,
    seed: int = W2V_SEED,
    verbose: bool = True,
) -> Word2Vec:
    if verbose:
        print(f"  Training Word2Vec (sg={sg}) on {len(tokenised_sents)} sentences")
        print(f"    vector_size={vector_size} window={window} min_count={min_count} "
              f"epochs={epochs} workers={workers}")
    model = Word2Vec(
        sentences=tokenised_sents,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        epochs=epochs,
        workers=workers,
        seed=seed,
    )
    if verbose:
        print(f"    Vocabulary: {len(model.wv.key_to_index):,} words")
    return model


def save_word2vec(model: Word2Vec, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    return path


def load_word2vec(path: Path) -> Word2Vec:
    return Word2Vec.load(str(path))
