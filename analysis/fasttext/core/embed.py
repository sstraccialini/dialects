"""
FastText (gensim) and BPE+TF-IDF (sentencepiece) embedding logic.

Two independent sub-pipelines:

  - FastText: train on (tokenised) sentences → mean-pool token vectors
              per sentence → mean-pool sentence vectors per variety.
              Subwords let OOV tokens still receive embeddings.

  - BPE     : train SentencePiece BPE over all sentences → tokenize each
              variety's sentences into BPE pieces → fit TF-IDF on BPE
              pieces (one super-document per variety).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import sentencepiece as spm
from gensim.models import FastText
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import (
    FT_VECTOR_SIZE, FT_WINDOW, FT_MIN_COUNT,
    FT_MIN_N, FT_MAX_N, FT_EPOCHS, FT_SG, FT_WORKERS,
    RANDOM_STATE,
    BPE_VOCAB_SIZE, BPE_CHARACTER_COVERAGE,
    BPE_TFIDF_MIN_DF, BPE_TFIDF_MAX_DF, BPE_TFIDF_MAX_FEATURES,
    SUBLINEAR_TF, NORM,
)
from .preprocess import preprocess_for_subword


# --------------------------------------------------------------------------- #
# FastText
# --------------------------------------------------------------------------- #
def _tokenize(text: str) -> List[str]:
    return text.split()


def _prepare_sentences(
    data: Dict[str, List[str]],
    codes: List[str],
) -> List[List[str]]:
    sentences: List[List[str]] = []
    for code in codes:
        if code not in data:
            continue
        for sent in data[code]:
            tokens = _tokenize(preprocess_for_subword(sent))
            if tokens:
                sentences.append(tokens)
    return sentences


def train_fasttext(
    data: Dict[str, List[str]],
    codes: List[str],
    save_to: Path = None,
    verbose: bool = True,
) -> FastText:
    if verbose:
        print("  Preparing sentences for FastText training...")
    sentences = _prepare_sentences(data, codes)
    if verbose:
        print(f"  Total sentences: {len(sentences):,}")
        print("  Training FastText model...")
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
    if verbose:
        print(f"  Vocab size: {len(model.wv):,}  |  Vector dim: {model.vector_size}")
    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_to))
        if verbose:
            print(f"  Model saved → {save_to}")
    return model


def _embed_sentence(model: FastText, tokens: List[str]) -> np.ndarray:
    if not tokens:
        return np.zeros(model.vector_size, dtype=np.float32)
    vecs = np.array([model.wv[tok] for tok in tokens], dtype=np.float32)
    return vecs.mean(axis=0)


def variety_embeddings_fasttext(
    model: FastText,
    data: Dict[str, List[str]],
    codes: List[str],
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    if verbose:
        print("  Computing variety embeddings (mean-pooling sentences)...")
    embeddings = []
    out_codes = []
    for code in codes:
        if code not in data:
            continue
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
        out_codes.append(code)
    X = np.array(embeddings, dtype=np.float32)
    if verbose:
        print(f"  Variety embedding matrix: {X.shape}")
    return X, out_codes


def per_sentence_fasttext(
    model: FastText,
    data: Dict[str, List[str]],
    codes: List[str],
) -> Dict[str, np.ndarray]:
    """For parallel-alignment eval: {code: (N, D) sentence vectors}."""
    out: Dict[str, np.ndarray] = {}
    for code in codes:
        if code not in data:
            continue
        rows = []
        for sent in data[code]:
            tokens = _tokenize(preprocess_for_subword(sent))
            rows.append(_embed_sentence(model, tokens))
        if rows:
            out[code] = np.vstack(rows).astype(np.float32)
        else:
            out[code] = np.zeros((0, model.vector_size), dtype=np.float32)
    return out


# --------------------------------------------------------------------------- #
# BPE + TF-IDF
# --------------------------------------------------------------------------- #
def train_bpe(
    data: Dict[str, List[str]],
    codes: List[str],
    model_prefix: Path,
    verbose: bool = True,
) -> spm.SentencePieceProcessor:
    """Train a SentencePiece BPE model on all sentences from all varieties."""
    model_prefix = Path(model_prefix)
    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    corpus_lines: List[str] = []
    for code in codes:
        if code not in data:
            continue
        for sent in data[code]:
            processed = preprocess_for_subword(sent)
            if processed.strip():
                corpus_lines.append(processed)

    corpus_path = model_prefix.parent / "_bpe_train_corpus.txt"
    corpus_path.write_text("\n".join(corpus_lines), encoding="utf-8")
    if verbose:
        print(f"  Corpus: {len(corpus_lines):,} sentences → {corpus_path}")
        print(f"  Training BPE (vocab_size={BPE_VOCAB_SIZE})...")

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=BPE_VOCAB_SIZE,
        model_type="bpe",
        character_coverage=BPE_CHARACTER_COVERAGE,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        normalization_rule_name="identity",
    )
    corpus_path.unlink(missing_ok=True)

    sp = spm.SentencePieceProcessor()
    sp.Load(str(model_prefix) + ".model")
    if verbose:
        print(f"  BPE model saved → {model_prefix}.model")
    return sp


def build_bpe_documents(
    sp: spm.SentencePieceProcessor,
    data: Dict[str, List[str]],
    codes: List[str],
) -> Tuple[List[str], List[str]]:
    docs = []
    out_codes = []
    for code in codes:
        if code not in data:
            continue
        pieces: List[str] = []
        for sent in data[code]:
            processed = preprocess_for_subword(sent)
            pieces.extend(sp.EncodeAsPieces(processed))
        docs.append(" ".join(pieces))
        out_codes.append(code)
    return docs, out_codes


def build_bpe_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        min_df=BPE_TFIDF_MIN_DF,
        max_df=BPE_TFIDF_MAX_DF,
        max_features=BPE_TFIDF_MAX_FEATURES,
        sublinear_tf=SUBLINEAR_TF,
        norm=NORM,
        lowercase=False,
        stop_words=None,
        strip_accents=None,
        token_pattern=r"(?u)\S+",
    )


def fit_transform_bpe(
    sp: spm.SentencePieceProcessor,
    data: Dict[str, List[str]],
    codes: List[str],
    verbose: bool = True,
) -> Tuple["scipy.sparse.csr_matrix", TfidfVectorizer, List[str]]:
    if verbose:
        print("  Tokenizing varieties with BPE...")
    docs, out_codes = build_bpe_documents(sp, data, codes)
    vectorizer = build_bpe_vectorizer()
    X = vectorizer.fit_transform(docs)
    if verbose:
        print(f"  BPE TF-IDF matrix: {X.shape}")
    return X, vectorizer, out_codes


def transform_bpe(
    sp: spm.SentencePieceProcessor,
    vectorizer: TfidfVectorizer,
    data: Dict[str, List[str]],
    codes: List[str],
) -> Tuple["scipy.sparse.csr_matrix", List[str]]:
    docs, out_codes = build_bpe_documents(sp, data, codes)
    X = vectorizer.transform(docs)
    return X, out_codes


def per_sentence_bpe(
    sp: spm.SentencePieceProcessor,
    vectorizer: TfidfVectorizer,
    data: Dict[str, List[str]],
    codes: List[str],
) -> Dict[str, np.ndarray]:
    """For parallel-alignment eval: BPE-piece TF-IDF per sentence."""
    out: Dict[str, np.ndarray] = {}
    for code in codes:
        if code not in data:
            continue
        sents_as_pieces = []
        for sent in data[code]:
            processed = preprocess_for_subword(sent)
            pieces = sp.EncodeAsPieces(processed)
            sents_as_pieces.append(" ".join(pieces))
        Xs = vectorizer.transform(sents_as_pieces)
        out[code] = Xs.toarray().astype(np.float32)
    return out


def top_bpe_features_per_variety(
    X,
    vectorizer: TfidfVectorizer,
    codes: List[str],
    k: int = 30,
) -> dict:
    feat_names = vectorizer.get_feature_names_out()
    result = {}
    for i, code in enumerate(codes):
        row = X[i].toarray().ravel()
        top_idx = row.argsort()[::-1][:k]
        result[code] = [(feat_names[j], float(row[j])) for j in top_idx if row[j] > 0]
    return result
