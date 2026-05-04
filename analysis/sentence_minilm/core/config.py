"""
Shared configuration for the sentence-MiniLM experiments.
Backbone: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2.

Path/variety/sampling/distance constants come from `analysis._shared.varieties`;
this file only adds the backbone-specific knobs.
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
# Sentence-transformer
# --------------------------------------------------------------------------- #
SENTENCE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SENTENCE_BATCH_SIZE = 32
ARTICLE_AGGREGATION = "mean"
VARIETY_AGGREGATION = "mean"

# Inference parameters (used by both baseline and fine-tuned experiments)
MAX_LENGTH = 128
BATCH_SIZE = 64
