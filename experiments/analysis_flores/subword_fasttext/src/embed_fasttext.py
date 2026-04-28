"""
FastText pipeline: shared model + per-variety mean-pooled sentence vectors.

Why a SHARED model (not per-variety):
    Each variety only contributes ~2009 sentences. Training a separate
    FastText per variety would give flabby, noisy embeddings. A single
    shared model sees all varieties at once, so subword n-grams
    generalise across morphologically-related forms (e.g. Italian
    "parlare" vs Sicilian "parrari" share "-are"/"-ari" suffixes).

Pipeline:
    1) Tokenise every sentence of every variety with
       `preprocess.tokenize_for_fasttext`.
    2) Train gensim.models.FastText on the flat list of tokenised
       sentences (config.FT_* hyperparameters).
    3) For each sentence, compute its embedding as the mean of its
       token vectors (OOV words still get a vector via subword n-grams).
    4) For each variety, aggregate sentence embeddings into a single
       mean vector (L2-normalised at the end so cosine distances behave
       well).

Outputs saved to results/fasttext/:
    - fasttext_model.bin               (via gensim's native save)
    - sentence_vectors.npz             (all sentences + their variety code)
    - variety_vectors.csv              (n_varieties x FT_VECTOR_SIZE)

The variety_vectors CSV is what similarity/cluster/visualize consume.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from gensim.models import FastText

from config import (
    FT_VECTOR_SIZE,
    FT_WINDOW,
    FT_MIN_COUNT,
    FT_MIN_N,
    FT_MAX_N,
    FT_EPOCHS,
    FT_SG,
    FT_WORKERS,
    MODELS_DIR,
    RANDOM_STATE,
    VARIETY_CODES,
    results_subdir,
)
from preprocess import tokenize_for_fasttext


# ---------- Corpus preparation ----------

def build_tokenised_corpus(
    data: Dict[str, List[str]],
    codes: List[str] = VARIETY_CODES,
) -> Tuple[List[List[str]], List[str]]:
    """
    Flatten {slug: sentences} into two aligned lists:
        tokenised_sents[i] = list[str] tokens of sentence i
        sentence_codes[i]  = variety slug of sentence i

    We keep the order: all sentences of the first code, then the second,
    etc. This is deterministic because the data dict is ordered by
    codes.
    """
    tokenised: List[List[str]] = []
    sentence_codes: List[str] = []
    for slug in codes:
        if slug not in data:
            continue
        for s in data[slug]:
            toks = tokenize_for_fasttext(s)
            if not toks:
                continue
            tokenised.append(toks)
            sentence_codes.append(slug)
    return tokenised, sentence_codes


# ---------- Training ----------

def train_fasttext(
    tokenised_sents: List[List[str]],
    seed: int = RANDOM_STATE,
    verbose: bool = True,
) -> FastText:
    """Train a gensim FastText model on the full tokenised corpus."""
    if verbose:
        print(f"  Training FastText (skip-gram) on {len(tokenised_sents)} sentences")
        print(f"    vector_size={FT_VECTOR_SIZE} window={FT_WINDOW} "
              f"min_count={FT_MIN_COUNT} min_n={FT_MIN_N} max_n={FT_MAX_N} "
              f"epochs={FT_EPOCHS} sg={FT_SG}")

    model = FastText(
        vector_size=FT_VECTOR_SIZE,
        window=FT_WINDOW,
        min_count=FT_MIN_COUNT,
        min_n=FT_MIN_N,
        max_n=FT_MAX_N,
        sg=FT_SG,
        workers=FT_WORKERS,
        seed=seed,
    )
    model.build_vocab(corpus_iterable=tokenised_sents)
    model.train(
        corpus_iterable=tokenised_sents,
        total_examples=model.corpus_count,
        epochs=FT_EPOCHS,
    )
    if verbose:
        print(f"    Vocabulary: {len(model.wv.key_to_index):,} words")
    return model


def save_fasttext(model: FastText, path: Path) -> Path:
    """Save a trained FastText model (gensim native format)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    return path


def load_fasttext(path: Path) -> FastText:
    """Load a trained FastText model previously saved with save_fasttext."""
    return FastText.load(str(path))


# ---------- Embedding ----------

def embed_sentence(model: FastText, tokens: List[str]) -> np.ndarray:
    """
    Mean-pool token vectors. OOV words still get a vector via subwords.
    Returns a zero vector if the sentence is empty.
    """
    if not tokens:
        return np.zeros(model.vector_size, dtype=np.float32)
    vecs = [model.wv[t] for t in tokens]  # FastText always returns a vector
    return np.mean(vecs, axis=0)


def embed_corpus(
    model: FastText,
    tokenised_sents: List[List[str]],
) -> np.ndarray:
    """Embed every sentence. Returns (n_sentences, vector_size)."""
    mat = np.zeros((len(tokenised_sents), model.vector_size), dtype=np.float32)
    for i, toks in enumerate(tokenised_sents):
        mat[i] = embed_sentence(model, toks)
    return mat


def _l2_normalise(v: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation; safe for zero rows."""
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

    Returns (X, codes_ordered) aligned, so X[i] is the vector of
    codes_ordered[i] (only codes with >=1 sentence are kept).
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


# ---------- Save helpers ----------

def save_sentence_vectors(
    sentence_vectors: np.ndarray,
    sentence_codes: List[str],
    pipeline: str = "fasttext",
) -> Path:
    """Persist sentence-level embeddings (compressed npz)."""
    out = results_subdir(pipeline) / "sentence_vectors.npz"
    np.savez_compressed(
        out,
        vectors=sentence_vectors.astype(np.float32),
        codes=np.asarray(sentence_codes),
    )
    return out


def save_variety_vectors(
    X: np.ndarray,
    codes: List[str],
    pipeline: str = "fasttext",
) -> Path:
    """Persist per-variety vectors as a labeled CSV (one row per variety)."""
    df = pd.DataFrame(X, index=codes)
    out = results_subdir(pipeline) / "variety_vectors.csv"
    df.to_csv(out, float_format="%.6f")
    return out


# ---------- End-to-end ----------

def run_fasttext_pipeline(
    data: Dict[str, List[str]],
    codes: List[str] = VARIETY_CODES,
    seed: int = RANDOM_STATE,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str], FastText]:
    """
    Full FastText sub-pipeline:
        tokenise -> train -> embed -> aggregate per variety -> save.

    Returns (X, codes_ordered, model).
    """
    print("\n--- Building tokenised corpus ---")
    tokenised, sentence_codes = build_tokenised_corpus(data, codes)
    print(f"  {len(tokenised)} sentences, {sum(len(s) for s in tokenised):,} tokens")

    print("\n--- Training shared FastText ---")
    model = train_fasttext(tokenised, seed=seed, verbose=verbose)
    model_path = MODELS_DIR / "fasttext_model.bin"
    save_fasttext(model, model_path)
    print(f"  Saved: {model_path}")

    print("\n--- Computing sentence + variety embeddings ---")
    sent_vecs = embed_corpus(model, tokenised)
    X, codes_ordered = aggregate_variety_vectors(sent_vecs, sentence_codes, codes)
    print(f"  Variety vectors: shape={X.shape}, codes={codes_ordered}")

    sv_path = save_sentence_vectors(sent_vecs, sentence_codes)
    vv_path = save_variety_vectors(X, codes_ordered)
    print(f"  Saved: {sv_path}")
    print(f"  Saved: {vv_path}")

    return X, codes_ordered, model


if __name__ == "__main__":
    # Smoke test: tiny run on 30 sentences per variety.
    from data_loader import load_all_varieties
    data, _ = load_all_varieties(sample_size=30)
    X, codes, _ = run_fasttext_pipeline(data)
    print(f"\nSmoke test: X.shape={X.shape}, codes={codes}")
