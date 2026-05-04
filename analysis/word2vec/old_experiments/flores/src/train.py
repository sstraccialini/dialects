"""
Train a single shared Word2Vec model on all 16 FLORES+ varieties.

We train on the full list of tokenised sentences, preserving the
canonical order of varieties (determinism) but otherwise shuffling is
handled internally by gensim.

Hyperparameters come from config.W2V_*.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from gensim.models import Word2Vec

from config import (
    W2V_VECTOR_SIZE,
    W2V_WINDOW,
    W2V_MIN_COUNT,
    W2V_SG,
    W2V_EPOCHS,
    W2V_WORKERS,
    W2V_SEED,
    VARIETY_CODES,
    MODELS_DIR,
)
from preprocess import tokenize


def build_tokenised_corpus(
    data: Dict[str, List[str]],
    codes: List[str] = VARIETY_CODES,
) -> Tuple[List[List[str]], List[str]]:
    """
    Flatten {slug: sentences} into (tokenised_sents, sentence_codes)
    aligned lists. Empty sentences are dropped.
    """
    tokenised: List[List[str]] = []
    sentence_codes: List[str] = []
    for slug in codes:
        if slug not in data:
            continue
        for s in data[slug]:
            toks = tokenize(s)
            if toks:
                tokenised.append(toks)
                sentence_codes.append(slug)
    return tokenised, sentence_codes


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
    """Train a Word2Vec (skip-gram) model on a list of tokenised sentences."""
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


def save_word2vec(model: Word2Vec, path: Path = None) -> Path:
    """Save the trained model (gensim native format)."""
    out = path or (MODELS_DIR / "word2vec.model")
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    return out


def load_word2vec(path: Path = None) -> Word2Vec:
    """Load a previously trained Word2Vec model."""
    src = path or (MODELS_DIR / "word2vec.model")
    return Word2Vec.load(str(src))
