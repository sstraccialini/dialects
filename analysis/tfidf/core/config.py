"""
Shared configuration for TF-IDF experiments.

Path/variety/sampling/distance constants come from the central
`analysis._shared.varieties` module — this file only adds TF-IDF
specific knobs (preprocessing flags, n-gram ranges).
"""
from __future__ import annotations

from analysis._shared.varieties import (  # noqa: F401
    REPO_ROOT, DATASET_DIR,
    WIKI_DIR, WIKI_GROUP_A_DIR, WIKI_LANGUAGES_DIR,
    VARIETIES, VARIETY_CODES, VARIETY_GROUP,
    DIALECT_CODES, MODERN_LANGUAGE_CODES,
    WIKI_VARIETY_DIR, FLORES_SLUG, OLDI_PARQUET,
    VARIETY_NAMES, GROUP_NAMES, GROUP_COLORS, ROMANCE_FAMILIES,
    SAMPLE_SIZE, RANDOM_STATE,
    DISTANCE_METRIC, LINKAGE_METHODS, PROJECTION_METHODS,
    experiment_dirs,
)


# --------------------------------------------------------------------------- #
# Preprocessing — already done by aggressive_normalize at extraction time,
# kept here for documentation / fallback if a script disables it.
# --------------------------------------------------------------------------- #
LOWERCASE = True               # idempotent on aggressive-normalized text
MASK_NUMBERS = False           # already stripped at extraction
NUMBER_TOKEN = " "
STRIP_PUNCT_FOR_WORD = False   # already stripped
KEEP_DIACRITICS = False        # already stripped


# --------------------------------------------------------------------------- #
# TF-IDF hyperparameters (shared for every experiment)
# --------------------------------------------------------------------------- #
# Word pipeline
WORD_NGRAM_RANGE = (1, 2)
WORD_MIN_DF = 1
WORD_MAX_DF = 1.0
WORD_MAX_FEATURES = None

# Char pipeline (char_wb keeps n-grams within word boundaries)
CHAR_NGRAM_RANGE = (3, 5)
CHAR_ANALYZER = "char_wb"
CHAR_MIN_DF = 1
CHAR_MAX_DF = 1.0
CHAR_MAX_FEATURES = None

# Common
SUBLINEAR_TF = True
NORM = "l2"
