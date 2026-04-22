"""
BPE-based embedding pipeline (Person 5).

Steps:
  1. Train a SentencePiece BPE model on all variety sentences combined.
  2. Tokenize each variety's sentences into BPE pieces.
  3. Aggregate each variety into one "BPE super-document" (pieces joined
     by spaces).
  4. Fit TF-IDF on BPE pieces → 14 × V_bpe sparse matrix.

Why BPE + TF-IDF?
  - BPE is language-agnostic: discovers shared subword units without
    prior morphological knowledge.
  - For dialects, many words share roots with standard Italian but
    differ in suffixes/vowels; BPE captures shared pieces (e.g. "parl")
    while still encoding dialectal endings.
  - Using TF-IDF on BPE pieces (same hyperparameters as Person 1's word
    pipeline) isolates the contribution of the tokenization unit: the
    only variable vs. the baseline is word → BPE piece.
  - BPE is also the tokenization used by mBERT and XLM-R (Person 4),
    making a BPE-TF-IDF ↔ mBERT comparison interpretable.

Saved artefacts:
  results/models/bpe_model.model + .vocab  — SentencePiece model
  results/bpe/top_features.csv             — top-30 BPE pieces per variety
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import sentencepiece as spm
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from config import (
    BPE_VOCAB_SIZE,
    BPE_CHARACTER_COVERAGE,
    BPE_MODEL_PREFIX,
    BPE_TFIDF_MIN_DF,
    BPE_TFIDF_MAX_DF,
    BPE_TFIDF_MAX_FEATURES,
    SUBLINEAR_TF,
    NORM,
    MODELS_DIR,
    results_subdir,
)
from preprocess import preprocess_for_subword


def train_bpe(
    data: Dict[str, List[str]],
    codes: List[str],
) -> spm.SentencePieceProcessor:
    """
    Train a SentencePiece BPE model on all sentences from all varieties.

    Writes BPE_MODEL_PREFIX + '.model' and '.vocab' to MODELS_DIR.
    Returns the loaded SentencePieceProcessor ready for inference.
    """
    print("  Building BPE training corpus...")
    corpus_lines: List[str] = []
    for code in codes:
        for sent in data[code]:
            processed = preprocess_for_subword(sent)
            if processed.strip():
                corpus_lines.append(processed)

    # sentencepiece requires a file on disk.
    corpus_path = Path(MODELS_DIR) / "_bpe_train_corpus.txt"
    corpus_path.write_text("\n".join(corpus_lines), encoding="utf-8")
    print(f"  Corpus: {len(corpus_lines):,} sentences → {corpus_path}")

    print(f"  Training BPE (vocab_size={BPE_VOCAB_SIZE})...")
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=BPE_MODEL_PREFIX,
        vocab_size=BPE_VOCAB_SIZE,
        model_type="bpe",
        character_coverage=BPE_CHARACTER_COVERAGE,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        # Keep diacritics: do not apply Unicode normalization.
        normalization_rule_name="identity",
    )

    # Clean up temporary corpus file.
    corpus_path.unlink(missing_ok=True)

    sp = spm.SentencePieceProcessor()
    sp.Load(BPE_MODEL_PREFIX + ".model")
    print(f"  BPE model saved → {BPE_MODEL_PREFIX}.model")
    return sp


def build_bpe_documents(
    sp: spm.SentencePieceProcessor,
    data: Dict[str, List[str]],
    codes: List[str],
) -> List[str]:
    """
    Tokenize each variety with BPE and join pieces into a
    super-document string (one per variety, space-separated pieces).
    """
    docs = []
    for code in codes:
        pieces: List[str] = []
        for sent in data[code]:
            processed = preprocess_for_subword(sent)
            pieces.extend(sp.EncodeAsPieces(processed))
        docs.append(" ".join(pieces))
    return docs


def fit_transform_bpe(
    sp: spm.SentencePieceProcessor,
    data: Dict[str, List[str]],
    codes: List[str],
) -> Tuple["scipy.sparse.csr_matrix", TfidfVectorizer]:
    """
    Build BPE super-documents and fit TF-IDF on BPE pieces.

    Returns (X, vectorizer) with X shape (n_varieties, V_bpe).
    """
    print("  Tokenizing varieties with BPE...")
    docs = build_bpe_documents(sp, data, codes)

    vectorizer = TfidfVectorizer(
        analyzer="word",
        min_df=BPE_TFIDF_MIN_DF,
        max_df=BPE_TFIDF_MAX_DF,
        max_features=BPE_TFIDF_MAX_FEATURES,
        sublinear_tf=SUBLINEAR_TF,
        norm=NORM,
        lowercase=False,
        stop_words=None,
        strip_accents=None,
        # BPE pieces may start with "▁" (SentencePiece word-start marker);
        # use a pattern that matches any non-whitespace token.
        token_pattern=r"(?u)\S+",
    )
    X = vectorizer.fit_transform(docs)
    print(f"  BPE TF-IDF matrix: {X.shape}")
    return X, vectorizer


def top_bpe_features_per_variety(
    X,
    vectorizer: TfidfVectorizer,
    codes: List[str],
    k: int = 30,
) -> dict:
    """
    For each variety, return the top-k BPE pieces by TF-IDF weight.

    Returns {code: [(piece, weight), ...]} sorted descending.
    """
    feat_names = vectorizer.get_feature_names_out()
    result = {}
    for i, code in enumerate(codes):
        row = X[i].toarray().ravel()
        top_idx = row.argsort()[::-1][:k]
        result[code] = [(feat_names[j], float(row[j])) for j in top_idx if row[j] > 0]
    return result
