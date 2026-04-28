"""
BPE + TF-IDF pipeline.

Flow:
    1) Train a SentencePiece BPE model on ALL sentences across the 16
       varieties (character_coverage=0.9995 is enough even with Arabic
       and Greek since we don't lowercase-only — but we've already
       lowercased in preprocessing).
    2) Encode every variety's super-document into BPE pieces.
    3) Feed the BPE-piece sequences to TF-IDF with the SAME
       hyperparameters as Person 1's word pipeline (sublinear_tf,
       L2-norm, analyzer over pre-tokenised strings).
    4) Compute the cosine distance matrix (handled in similarity.py).

The single variable vs the TF-IDF word baseline is the tokenisation
unit (BPE pieces instead of whitespace-split words). That keeps the
comparison clean.

Outputs saved to:
    MODELS_DIR/bpe_model.model (+ .vocab)          (SentencePiece)
    results/bpe/variety_vectors.npz                (optional cache)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import sentencepiece as spm
from scipy.sparse import csr_matrix
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
    VARIETY_CODES,
    results_subdir,
)
from preprocess import preprocess_for_subword


# ---------- SentencePiece BPE training ----------

def train_bpe(
    data: Dict[str, List[str]],
    codes: List[str] = VARIETY_CODES,
    vocab_size: int = BPE_VOCAB_SIZE,
    character_coverage: float = BPE_CHARACTER_COVERAGE,
    model_prefix: str = BPE_MODEL_PREFIX,
    verbose: bool = True,
) -> spm.SentencePieceProcessor:
    """
    Train a SentencePiece BPE model on all varieties combined.

    We dump the preprocessed sentences into a temp .txt file and pass it
    to SentencePiece, which requires a file input.
    """
    # Ensure the models dir exists (BPE_MODEL_PREFIX lives under MODELS_DIR).
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    all_sents: List[str] = []
    for slug in codes:
        if slug not in data:
            continue
        for s in data[slug]:
            cleaned = preprocess_for_subword(s)
            if cleaned:
                all_sents.append(cleaned)

    if verbose:
        print(f"  Training BPE on {len(all_sents):,} preprocessed sentences "
              f"(vocab={vocab_size}, cov={character_coverage})")

    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".txt", delete=False
    ) as tmp:
        for s in all_sents:
            tmp.write(s + "\n")
        tmp_path = tmp.name

    try:
        spm.SentencePieceTrainer.Train(
            input=tmp_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            model_type="bpe",
            # Reasonable defaults
            input_sentence_size=0,        # use all
            shuffle_input_sentence=True,
            pad_id=0, unk_id=1, bos_id=-1, eos_id=-1,
            normalization_rule_name="identity",  # we already normalised
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
    if verbose:
        print(f"  BPE vocab size: {sp.get_piece_size():,}")
    return sp


def load_bpe(model_prefix: str = BPE_MODEL_PREFIX) -> spm.SentencePieceProcessor:
    """Load a previously trained SentencePiece BPE model."""
    return spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")


# ---------- Encoding ----------

def encode_sentence_to_pieces(sp: spm.SentencePieceProcessor, text: str) -> List[str]:
    """Preprocess + encode one sentence into BPE pieces."""
    cleaned = preprocess_for_subword(text)
    if not cleaned:
        return []
    return sp.encode(cleaned, out_type=str)


def encode_variety_document(
    sp: spm.SentencePieceProcessor,
    sentences: List[str],
) -> str:
    """
    Encode every sentence into BPE pieces and return them joined with
    spaces. This is the string we hand to TF-IDF.

    NOTE: SentencePiece pieces begin with '▁' to mark a word boundary.
    TfidfVectorizer with a simple split-on-whitespace tokenizer will
    treat each piece as an atomic token, which is what we want.
    """
    pieces_per_sentence: List[str] = []
    for s in sentences:
        pieces = encode_sentence_to_pieces(sp, s)
        if pieces:
            pieces_per_sentence.append(" ".join(pieces))
    return " ".join(pieces_per_sentence)


def build_bpe_documents(
    sp: spm.SentencePieceProcessor,
    data: Dict[str, List[str]],
    codes: List[str] = VARIETY_CODES,
) -> Tuple[List[str], List[str]]:
    """
    Build one BPE-piece super-document per variety.

    Returns (documents, codes_ordered) aligned to VARIETY_CODES order.
    """
    codes_ordered: List[str] = []
    documents: List[str] = []
    for slug in codes:
        if slug not in data:
            continue
        doc = encode_variety_document(sp, data[slug])
        codes_ordered.append(slug)
        documents.append(doc)
    return documents, codes_ordered


# ---------- TF-IDF over BPE pieces ----------

def _whitespace_tokenizer(text: str) -> List[str]:
    """Simple split: each BPE piece is a whole token."""
    return text.split()


def build_bpe_tfidf_vectorizer() -> TfidfVectorizer:
    """
    TfidfVectorizer over BPE pieces.

    - analyzer='word' with a whitespace tokenizer (no preprocessing;
      pieces are already tokens, including the '▁' prefix).
    - ngram_range=(1,1): BPE pieces already carry sub-word info, so we
      keep it simple and avoid feature-space blow-up.
    - sublinear_tf + l2 norm for stability (same as Person 1's baseline).
    - min_df=1, max_df=1.0 to preserve discriminative rare pieces.
    """
    return TfidfVectorizer(
        analyzer="word",
        tokenizer=_whitespace_tokenizer,
        preprocessor=None,
        lowercase=False,                 # already lowercased, and we
                                         # don't want to mangle '▁'
        token_pattern=None,
        ngram_range=(1, 1),
        min_df=BPE_TFIDF_MIN_DF,
        max_df=BPE_TFIDF_MAX_DF,
        max_features=BPE_TFIDF_MAX_FEATURES,
        sublinear_tf=SUBLINEAR_TF,
        norm=NORM,
    )


def fit_transform_bpe_tfidf(
    documents: List[str],
) -> Tuple[csr_matrix, TfidfVectorizer]:
    """Fit TF-IDF over BPE-piece documents and return (X, vectorizer)."""
    vec = build_bpe_tfidf_vectorizer()
    X = vec.fit_transform(documents)
    return X, vec


# ---------- Inspection helper ----------

def top_pieces_per_variety(
    X: csr_matrix,
    vectorizer: TfidfVectorizer,
    codes: List[str],
    k: int = 30,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Top-k BPE pieces per variety (by TF-IDF weight) for interpretability.
    """
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    top: Dict[str, List[Tuple[str, float]]] = {}
    arr = X.toarray()
    for i, code in enumerate(codes):
        row = arr[i]
        if row.size == 0:
            top[code] = []
            continue
        idx = np.argsort(row)[::-1][:k]
        top[code] = [(feature_names[j], float(row[j])) for j in idx if row[j] > 0]
    return top


# ---------- End-to-end ----------

def run_bpe_pipeline(
    data: Dict[str, List[str]],
    codes: List[str] = VARIETY_CODES,
    verbose: bool = True,
) -> Tuple[csr_matrix, TfidfVectorizer, List[str], spm.SentencePieceProcessor]:
    """
    Full BPE + TF-IDF sub-pipeline.

    Returns (X, vectorizer, codes_ordered, sp_model).
    """
    print("\n--- Training SentencePiece BPE ---")
    sp = train_bpe(data, codes=codes, verbose=verbose)

    print("\n--- Encoding variety super-documents ---")
    documents, codes_ordered = build_bpe_documents(sp, data, codes)
    print(f"  {len(documents)} documents, avg len = "
          f"{np.mean([len(d) for d in documents]):.0f} chars")

    print("\n--- TF-IDF over BPE pieces ---")
    X, vec = fit_transform_bpe_tfidf(documents)
    print(f"  X shape: {X.shape}  (nnz={X.nnz:,})")

    return X, vec, codes_ordered, sp


if __name__ == "__main__":
    from data_loader import load_all_varieties
    data, _ = load_all_varieties(sample_size=50)
    X, vec, codes, sp = run_bpe_pipeline(data)
    print(f"\nSmoke test: X.shape={X.shape}, codes={codes}")
    print(f"  first 5 features: {vec.get_feature_names_out()[:5]}")
