"""
Shared configuration for Word2Vec experiments.

Path/variety/sampling/distance constants come from `analysis._shared.varieties`;
this file only adds the Word2Vec-specific knobs.
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
# Preprocessing
# --------------------------------------------------------------------------- #
LOWERCASE = True               # idempotent on aggressive-normalized text
MASK_NUMBERS = False           # already stripped at extraction
NUMBER_TOKEN = " "


# --------------------------------------------------------------------------- #
# Word2Vec hyperparameters
# --------------------------------------------------------------------------- #
W2V_VECTOR_SIZE = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 2
W2V_SG = 1                     # skip-gram
W2V_EPOCHS = 15
W2V_WORKERS = 4
W2V_SEED = 42
