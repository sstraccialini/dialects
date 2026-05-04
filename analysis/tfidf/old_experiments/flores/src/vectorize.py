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
- reuse the same fit on new data in future experiments.
"""

from __future__ import annotations

from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer

from config import (
    WORD_NGRAM_RANGE, WORD_MIN_DF, WORD_MAX_DF, WORD_MAX_FEATURES,
    CHAR_NGRAM_RANGE, CHAR_ANALYZER, CHAR_MIN_DF, CHAR_MAX_DF, CHAR_MAX_FEATURES,
    SUBLINEAR_TF, NORM,
)
from preprocess import preprocess_for_word, preprocess_for_char


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
        # IMPORTANT: no lowercase here (already done by preprocess).
        lowercase=False,
        # IMPORTANT: no stop_words. sklearn's built-in list only covers
        # English and would bias the comparison; we let IDF handle
        # common terms in a language-agnostic way.
        stop_words=None,
        # NB: do NOT normalize unicode (keep distinctive diacritics).
        strip_accents=None,
    )


def build_char_vectorizer() -> TfidfVectorizer:
    """Factory for the CHAR n-gram vectorizer (char_wb)."""
    return TfidfVectorizer(
        analyzer=CHAR_ANALYZER,            # 'char_wb' respects word boundaries
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
    """
    Apply word preprocessing + TfidfVectorizer(word) to a list of
    super-documents (one per variety, in a fixed order).

    Returns (X, vectorizer) where X has shape (n_varieties, V_word).
    """
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


def top_features_per_variety(
    X,
    vectorizer: TfidfVectorizer,
    codes: List[str],
    k: int = 20,
) -> dict:
    """
    For each variety, return the top-k features by TF-IDF weight.

    Output: dict {code: list[(feature, weight)]}.
    Used for sanity checks and interpretability.
    """
    feature_names = vectorizer.get_feature_names_out()
    out = {}
    for i, code in enumerate(codes):
        row = X[i].toarray().ravel()
        top_idx = row.argsort()[::-1][:k]
        out[code] = [(feature_names[j], float(row[j])) for j in top_idx if row[j] > 0]
    return out


if __name__ == "__main__":
    # Smoke test: load data, fit both pipelines, print shapes and top-10
    # features per variety.
    from data_loader import load_all_varieties, build_variety_documents

    print("Loading...")
    data, _ = load_all_varieties(verbose=False)
    docs, codes = build_variety_documents(data)

    print("\n--- WORD pipeline ---")
    Xw, vw = fit_transform_word(docs)
    print(f"  X shape: {Xw.shape}")
    print(f"  avg nnz per doc: {Xw.nnz / Xw.shape[0]:.0f}")
    topw = top_features_per_variety(Xw, vw, codes, k=10)

    print("\n--- CHAR pipeline ---")
    Xc, vc = fit_transform_char(docs)
    print(f"  X shape: {Xc.shape}")
    print(f"  avg nnz per doc: {Xc.nnz / Xc.shape[0]:.0f}")
    topc = top_features_per_variety(Xc, vc, codes, k=10)

    print("\n--- Top-10 WORD features per variety ---")
    for c in codes:
        feats = ", ".join(f"{f}({w:.3f})" for f, w in topw[c][:10])
        print(f"  [{c:>10}] {feats}")

    print("\n--- Top-10 CHAR features per variety ---")
    for c in codes:
        feats = ", ".join(f"{f!r}({w:.3f})" for f, w in topc[c][:10])
        print(f"  [{c:>10}] {feats}")
