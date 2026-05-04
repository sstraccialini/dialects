"""
Shared configuration for FastText / BPE experiments.

Two sub-pipelines:
    fasttext — gensim FastText (subword char n-grams, skip-gram)
    bpe      — SentencePiece BPE tokenization + TF-IDF on BPE pieces

Path/variety/sampling/distance constants come from
`analysis._shared.varieties`; this file only adds FastText/BPE-specific knobs.
"""
from __future__ import annotations

from analysis._shared.varieties import (  # noqa: F401
    REPO_ROOT, DATASET_DIR,
    WIKI_DIR, WIKI_GROUP_A_DIR, WIKI_LANGUAGES_DIR,
    FLORES_DIR, OLDI_DIR,
    VARIETIES, VARIETY_CODES, VARIETY_GROUP,
    DIALECT_CODES, MODERN_LANGUAGE_CODES,
    WIKI_VARIETY_DIR, FLORES_SLUG, OLDI_PARQUET,
    VARIETY_NAMES, GROUP_NAMES, GROUP_COLORS, ROMANCE_FAMILIES,
    SAMPLE_SIZE, RANDOM_STATE,
    DISTANCE_METRIC, LINKAGE_METHODS, PROJECTION_METHODS,
    experiment_dirs,
)


# --------------------------------------------------------------------------- #
# Preprocessing — text already aggressive-normalized at extraction time.
# --------------------------------------------------------------------------- #
LOWERCASE = True
MASK_NUMBERS = False
NUMBER_TOKEN = " "
STRIP_PUNCT_FOR_WORD = False
KEEP_DIACRITICS = False


# --------------------------------------------------------------------------- #
# FastText hyperparameters
# --------------------------------------------------------------------------- #
FT_VECTOR_SIZE = 200
FT_WINDOW = 5
FT_MIN_COUNT = 2
FT_MIN_N = 3
FT_MAX_N = 6
FT_EPOCHS = 10
FT_SG = 1
FT_WORKERS = 4


# --------------------------------------------------------------------------- #
# BPE hyperparameters
# --------------------------------------------------------------------------- #
BPE_VOCAB_SIZE = 8000
BPE_CHARACTER_COVERAGE = 0.9995


# --------------------------------------------------------------------------- #
# TF-IDF on BPE tokens
# --------------------------------------------------------------------------- #
BPE_TFIDF_MIN_DF = 1
BPE_TFIDF_MAX_DF = 1.0
BPE_TFIDF_MAX_FEATURES = None
SUBLINEAR_TF = True
NORM = "l2"
