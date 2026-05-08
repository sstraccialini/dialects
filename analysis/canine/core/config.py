"""
Shared configuration for CANINE (character-level encoder) experiments.
Backbone (default): google/canine-c.

Path/variety/sampling/distance constants come from `analysis._shared.varieties`;
this file only adds CANINE-specific knobs.
"""
from __future__ import annotations

from analysis._shared.varieties import (  # noqa: F401
    REPO_ROOT, DATASET_DIR,
    WIKI_DIR,
    WIKI_NATIVE_GROUP_A_DIR    as WIKI_GROUP_A_DIR,    # not_normalized for pretrained
    WIKI_NATIVE_LANGUAGES_DIR  as WIKI_LANGUAGES_DIR,  # idem
    FLORES_NATIVE_DIR          as FLORES_DIR,          # idem
    OLDI_NATIVE_DIR            as OLDI_DIR,            # idem
    WIKI_VARIETY_DIR_NATIVE    as WIKI_VARIETY_DIR,    # idem
    VARIETIES, VARIETY_CODES, VARIETY_GROUP,
    DIALECT_CODES, MODERN_LANGUAGE_CODES,
    FLORES_SLUG, OLDI_PARQUET,
    OLDI_PAIR_DIALECTS, OLDI_PAIR_SLUG,
    VARIETY_NAMES, GROUP_NAMES, GROUP_COLORS, ROMANCE_FAMILIES,
    SAMPLE_SIZE, RANDOM_STATE,
    DISTANCE_METRIC, LINKAGE_METHODS, PROJECTION_METHODS,
    experiment_dirs,
)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
DEFAULT_MODEL_NAME = "google/canine-c"
MAX_LENGTH = 512   # characters
BATCH_SIZE = 16    # heavier per-sample than XLM-R
