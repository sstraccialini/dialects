"""
TF-IDF vectorization of varieties.

Two independent pipelines:

- WORD: TfidfVectorizer(analyzer='word', ngram_range=(1,2))
- CHAR: TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5))

Both operate on the "super-document per variety" produced by
`data_loader.build_variety_documents`. Final output: a TF-IDF matrix
of shape (n_varieties, V).

We return both the matrix and the vectorizer so we can:
- inspect the vocabulary (top features per variety);
- reuse the same fit on new data (transform-only) for cross-domain experiments.
"""
from __future__ import annotations

from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer

from .config import (
    WORD_NGRAM_RANGE, WORD_MIN_DF, WORD_MAX_DF, WORD_MAX_FEATURES,
    CHAR_NGRAM_RANGE, CHAR_ANALYZER, CHAR_MIN_DF, CHAR_MAX_DF, CHAR_MAX_FEATURES,
    SUBLINEAR_TF, NORM,
)
from .preprocess import preprocess_for_word, preprocess_for_char


def build_word_vectorizer() -> TfidfVectorizer:
    """Factory for the WORD n-gram vectorizer."""
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=WORD_NGRAM_RANGE,
        min_df=WORD_MIN_DF,
        max_df=WORD_MAX_DF,
        max_features=WORD_MAX_FEATURES,
        sublinear_tf=SUBLINEAR_TF,
        norm=NORM,
        lowercase=False,            # already done in preprocess
        stop_words=None,            # language-agnostic, IDF handles common terms
        strip_accents=None,         # keep diacritics
    )


def build_char_vectorizer() -> TfidfVectorizer:
    """Factory for the CHAR n-gram vectorizer (char_wb)."""
    return TfidfVectorizer(
        analyzer=CHAR_ANALYZER,     # 'char_wb' respects word boundaries
        ngram_range=CHAR_NGRAM_RANGE,
        min_df=CHAR_MIN_DF,
        max_df=CHAR_MAX_DF,
        max_features=CHAR_MAX_FEATURES,
        sublinear_tf=SUBLINEAR_TF,
        norm=NORM,
        lowercase=False,
        stop_words=None,
        strip_accents=None,
    )


def fit_transform_word(
    variety_documents: List[str],
) -> Tuple["scipy.sparse.csr_matrix", TfidfVectorizer]:
    """Fit a fresh word TfidfVectorizer on the given super-documents and
    return (X, vectorizer). X has shape (n_varieties, V_word)."""
    processed = [preprocess_for_word(doc) for doc in variety_documents]
    vec = build_word_vectorizer()
    X = vec.fit_transform(processed)
    return X, vec


def fit_transform_char(
    variety_documents: List[str],
) -> Tuple["scipy.sparse.csr_matrix", TfidfVectorizer]:
    """Same as fit_transform_word, but for char n-grams."""
    processed = [preprocess_for_char(doc) for doc in variety_documents]
    vec = build_char_vectorizer()
    X = vec.fit_transform(processed)
    return X, vec


def transform_word(
    vectorizer: TfidfVectorizer,
    variety_documents: List[str],
) -> "scipy.sparse.csr_matrix":
    """Transform-only: reuse a vectorizer fitted on Wiki to vectorize a
    new set of documents (e.g. OLDI/FLORES) without refitting. Used for
    cross-domain experiments."""
    processed = [preprocess_for_word(doc) for doc in variety_documents]
    return vectorizer.transform(processed)


def transform_char(
    vectorizer: TfidfVectorizer,
    variety_documents: List[str],
) -> "scipy.sparse.csr_matrix":
    """Char counterpart to transform_word."""
    processed = [preprocess_for_char(doc) for doc in variety_documents]
    return vectorizer.transform(processed)


def top_features_per_variety(
    X,
    vectorizer: TfidfVectorizer,
    codes: List[str],
    k: int = 20,
) -> dict:
    """For each variety, return top-k features by TF-IDF weight.

    Output: dict {code: list[(feature, weight)]}. Useful for sanity checks
    and interpretability.
    """
    feature_names = vectorizer.get_feature_names_out()
    out = {}
    for i, code in enumerate(codes):
        row = X[i].toarray().ravel()
        top_idx = row.argsort()[::-1][:k]
        out[code] = [
            (feature_names[j], float(row[j])) for j in top_idx if row[j] > 0
        ]
    return out
