"""
Configuration for XPhoneBERT phoneme-level experiments.

Variety set is **extended to 15** compared to the orthographic experiments:
the standard `_shared.varieties` registry (13) is augmented with Arabic
(arb) and Greek (ell). They were dropped from the orthographic side
because of script differences (Latin vs Arabic vs Greek alphabets), but
in IPA space all scripts collapse to the same Unicode IPA alphabet, so
they're meaningful here.
"""
from __future__ import annotations

from analysis._shared.varieties import (  # noqa: F401
    REPO_ROOT, DATASET_DIR,
    DIALECT_CODES,
    SAMPLE_SIZE, RANDOM_STATE,
    DISTANCE_METRIC, LINKAGE_METHODS, PROJECTION_METHODS,
    experiment_dirs,
)


# --------------------------------------------------------------------------- #
# Manzini-Savoia phonetic data (for the 6 dialects)
# --------------------------------------------------------------------------- #
MS_DIR = DATASET_DIR / "manzini_savoia" / "by_region"
MS_REGION_FILE = {
    "fur": "friuli_venezia_giulia.csv",
    "lij": "liguria.csv",
    "lmo": "lombardia.csv",
    "sc":  "sardegna.csv",
    "scn": "sicilia.csv",
    "vec": "veneto.csv",
}


# --------------------------------------------------------------------------- #
# Variety registry — 15 codes (13 original + arb + ell)
# --------------------------------------------------------------------------- #
VARIETIES = [
    # Italo-Romance dialects
    ("fur", "italo_romance"),
    ("lij", "italo_romance"),
    ("lmo", "italo_romance"),
    ("sc",  "italo_romance"),
    ("scn", "italo_romance"),
    ("vec", "italo_romance"),
    # Standard Italian
    ("ita", "italian"),
    # Other Romance
    ("spa", "romance"),
    ("fra", "romance"),
    ("cat", "romance"),
    # Non-Romance European
    ("deu", "germanic"),
    ("slv", "slavic"),
    ("eng", "english"),
    # Newly re-added thanks to IPA (script-independent)
    ("ell", "greek"),
    ("arb", "semitic"),
]
VARIETY_CODES = [c for c, _ in VARIETIES]
VARIETY_GROUP = dict(VARIETIES)

STANDARD_LANGUAGE_CODES = [
    "ita", "spa", "fra", "cat", "deu", "slv", "eng", "ell", "arb",
]


# --------------------------------------------------------------------------- #
# Display labels
# --------------------------------------------------------------------------- #
VARIETY_NAMES = {
    "fur": "Friulian", "lij": "Ligurian", "lmo": "Lombard",
    "sc":  "Sardinian", "scn": "Sicilian", "vec": "Venetian",
    "ita": "Italian", "spa": "Spanish", "fra": "French", "cat": "Catalan",
    "deu": "German", "slv": "Slovenian", "eng": "English",
    "ell": "Greek", "arb": "Arabic",
}

GROUP_NAMES = {
    "italo_romance": "Italo-Romance",
    "italian":       "Italian (standard)",
    "romance":       "Romance (ES, FR, CA)",
    "germanic":      "Germanic (German)",
    "english":       "English",
    "slavic":        "Slovenian",
    "greek":         "Greek",
    "semitic":       "Arabic",
}

GROUP_COLORS = {
    "italo_romance": "#d62728",
    "italian":       "#ff7f0e",
    "romance":       "#2ca02c",
    "germanic":      "#1f77b4",
    "english":       "#17becf",
    "slavic":        "#e377c2",
    "greek":         "#9467bd",
    "semitic":       "#8c564b",
}

ROMANCE_FAMILIES = {"italo_romance", "italian", "romance"}


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
MODEL_NAME = "vinai/xphonebert-base"
MAX_LENGTH = 512   # tokens (phonemes) — IPA strings are short, plenty
BATCH_SIZE = 32
