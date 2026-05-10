"""
Single source of truth for the variety registry (codes, families, names,
colors), the dataset paths, and a couple of cross-method utilities.

Every method's `core/config.py` re-exports these names and adds only
its method-specific knobs (model name, training hyperparameters, ...).
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Repository / dataset paths
# --------------------------------------------------------------------------- #
# This file lives at <repo>/analysis/_shared/varieties.py
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "Dataset"

WIKI_DIR = DATASET_DIR / "wiki"

# Wiki has two parallel variants (see Dataset/wiki/PIPELINE.md):
#   - normalized/      lowercase ASCII only (matches FLORES/OLDI normalized)
#   - not_normalized/  native text with diacritics/case/punct/digits/non-ASCII
# Default constants point to the normalized variant for backward compat;
# methods that consume pretrained-tokenizer-friendly text (XLM-R, CANINE,
# Sentence-MiniLM, LaBSE) should opt-in to *_NATIVE_*.
WIKI_NORMALIZED_DIR     = WIKI_DIR / "normalized"
WIKI_NATIVE_DIR         = WIKI_DIR / "not_normalized"

WIKI_GROUP_A_DIR        = WIKI_NORMALIZED_DIR / "dialects_in_both_OLDI_and_Flores"
WIKI_LANGUAGES_DIR      = WIKI_NORMALIZED_DIR / "languages"
WIKI_OTHERS_DIR         = WIKI_NORMALIZED_DIR / "others_dialects"

WIKI_NATIVE_GROUP_A_DIR   = WIKI_NATIVE_DIR / "dialects_in_both_OLDI_and_Flores"
WIKI_NATIVE_LANGUAGES_DIR = WIKI_NATIVE_DIR / "languages"
WIKI_NATIVE_OTHERS_DIR    = WIKI_NATIVE_DIR / "others_dialects"

# --------------------------------------------------------------------------- #
# Cleaned single-CSV format (FINAL experiments — 2026-05-10).
# Each CSV has 3 metadata cols (unique_id, dataset, original_id) + one column
# per variety using the FLORES_SLUG name (italiano, friulano, ...).
#   FLORES cleaned:            1827 × 20  (3 meta + 17 lang)
#   FLORES cleaned_normalized: same (aggressive_normalize applied)
#   OLDI   cleaned:            5167 × 10  (3 meta + ita + 6 dialects)
#   OLDI   cleaned_normalized: same
# Use these for the FINAL experiments. The old per-language tree
# (Dataset/flores/normalized/, Dataset/oldi/normalized/, etc.) was archived
# under Dataset_archive/ on 2026-05-10.
# --------------------------------------------------------------------------- #
FLORES_CLEANED_DIR       = DATASET_DIR / "flores" / "cleaned"
FLORES_CLEANED_NORM_DIR  = DATASET_DIR / "flores" / "cleaned_normalized"
OLDI_CLEANED_DIR         = DATASET_DIR / "oldi"   / "cleaned"
OLDI_CLEANED_NORM_DIR    = DATASET_DIR / "oldi"   / "cleaned_normalized"

FLORES_CLEANED_CSV       = FLORES_CLEANED_DIR      / "flores.csv"
FLORES_CLEANED_NORM_CSV  = FLORES_CLEANED_NORM_DIR / "flores.csv"
OLDI_CLEANED_CSV         = OLDI_CLEANED_DIR        / "oldi.csv"
OLDI_CLEANED_NORM_CSV    = OLDI_CLEANED_NORM_DIR   / "oldi.csv"

# Legacy constants — point to the archived locations so any leftover code
# fails LOUDLY rather than silently picking up partial data.
FLORES_DIR        = REPO_ROOT / "Dataset_archive" / "flores" / "normalized"
FLORES_NATIVE_DIR = REPO_ROOT / "Dataset_archive" / "flores" / "not_normalized"
OLDI_DIR          = REPO_ROOT / "Dataset_archive" / "oldi"   / "normalized"
OLDI_NATIVE_DIR   = REPO_ROOT / "Dataset_archive" / "oldi"   / "not_normalized"


# --------------------------------------------------------------------------- #
# Variety registry — 13 varieties (ISO 639-3).
# --------------------------------------------------------------------------- #
VARIETIES = [
    # Italo-Romance dialects (Group A — OLDI ∩ FLORES)
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
    ("por", "romance"),
    ("oci", "romance"),
    # Germanic
    ("deu", "germanic"),
    ("eng", "english"),
    # Slavic
    ("slv", "slavic"),
    ("hrv", "slavic"),
    # Non-Indo-European
    ("hun", "uralic"),
]

VARIETY_CODES = [code for code, _ in VARIETIES]
VARIETY_GROUP = dict(VARIETIES)

# Convenience subsets
DIALECT_CODES = ["fur", "lij", "lmo", "sc", "scn", "vec"]
MODERN_LANGUAGE_CODES = [
    "ita", "spa", "fra", "cat", "por", "oci",   # Romance
    "deu", "eng",                                # Germanic
    "slv", "hrv",                                # Slavic
    "hun",                                       # Uralic
]


# --------------------------------------------------------------------------- #
# Per-corpus path resolvers
# --------------------------------------------------------------------------- #
# Languages routed to wiki/{normalized,not_normalized}/languages/.
_LANG_CODES = ["ita", "spa", "fra", "cat", "por", "oci",
               "deu", "eng", "slv", "hrv", "hun"]

WIKI_VARIETY_DIR = {
    **{c: WIKI_GROUP_A_DIR for c in DIALECT_CODES},
    **{c: WIKI_LANGUAGES_DIR for c in _LANG_CODES},
}

# Same mapping for the native (not_normalized) variant.  Use this for
# pretrained-encoder methods (XLM-R, CANINE, Sentence-MiniLM, LaBSE).
WIKI_VARIETY_DIR_NATIVE = {
    **{c: WIKI_NATIVE_GROUP_A_DIR for c in DIALECT_CODES},
    **{c: WIKI_NATIVE_LANGUAGES_DIR for c in _LANG_CODES},
}

# FLORES filenames use the Italian-name slug.
FLORES_SLUG = {
    "fur": "friulano", "lij": "ligure", "lmo": "lombardo", "sc": "sardo",
    "scn": "siciliano", "vec": "veneto",
    "ita": "italiano", "spa": "spagnolo", "fra": "francese", "cat": "catalano",
    "deu": "tedesco", "slv": "sloveno", "eng": "inglese",
    # NEW (FLORES+ downloaded May 2026)
    "por": "portoghese", "oci": "occitano",
    "hrv": "croato", "hun": "ungherese",
}

# OLDI parquet filenames use BCP47 ("<iso>_<script>"); Sardinian uses "srd".
# NOTE: the 7 new standards (por/ron/oci/glg/hrv/sqi/hun) are NOT in OLDI.
# They get OLDI-eval skipped via the missing-key check in data loaders.
OLDI_PARQUET = {
    "fur": "fur_Latn", "lij": "lij_Latn", "lmo": "lmo_Latn",
    "sc":  "srd_Latn", "scn": "scn_Latn", "vec": "vec_Latn",
    "ita": "ita_Latn", "spa": "spa_Latn", "fra": "fra_Latn",
    "cat": "cat_Latn", "deu": "deu_Latn", "slv": "slv_Latn", "eng": "eng_Latn",
}

# OLDI Italian↔dialect parallel pairs — only the 6 Italo-Romance dialects.
# Files live at Dataset/oldi/normalized/pairs_ita_<slug>.tsv with columns
# {"id", "italiano", "<slug>"}.
OLDI_PAIR_DIALECTS = ["fur", "lij", "lmo", "sc", "scn", "vec"]
OLDI_PAIR_SLUG = {
    "fur": "friulano", "lij": "ligure", "lmo": "lombardo",
    "sc":  "sardo",    "scn": "siciliano", "vec": "veneto",
}


# --------------------------------------------------------------------------- #
# Display labels (for plots and tables)
# --------------------------------------------------------------------------- #
VARIETY_NAMES = {
    "fur": "Friulian",
    "lij": "Ligurian",
    "lmo": "Lombard",
    "sc":  "Sardinian",
    "scn": "Sicilian",
    "vec": "Venetian",
    "ita": "Italian",
    "spa": "Spanish",
    "fra": "French",
    "cat": "Catalan",
    "por": "Portuguese",
    "oci": "Occitan",
    "deu": "German",
    "eng": "English",
    "slv": "Slovenian",
    "hrv": "Croatian",
    "hun": "Hungarian",
}

GROUP_NAMES = {
    "italo_romance": "Italo-Romance",
    "italian":       "Italian (standard)",
    "romance":       "Romance (other)",
    "germanic":      "Germanic",
    "english":       "English",
    "slavic":        "Slavic",
    "uralic":        "Uralic (non-IE)",
}

GROUP_COLORS = {
    "italo_romance": "#d62728",  # red
    "italian":       "#ff7f0e",  # orange
    "romance":       "#2ca02c",  # green
    "germanic":      "#1f77b4",  # blue
    "english":       "#17becf",  # cyan
    "slavic":        "#e377c2",  # pink
    "uralic":        "#8c564b",  # brown
}

ROMANCE_FAMILIES = {"italo_romance", "italian", "romance"}
DIALECT_FAMILIES = {"italo_romance"}


# --------------------------------------------------------------------------- #
# Sampling defaults
# --------------------------------------------------------------------------- #
SAMPLE_SIZE = 100_000          # max sentences per variety from Wiki
RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Distance / clustering / visualization defaults
# --------------------------------------------------------------------------- #
DISTANCE_METRIC = "cosine"
LINKAGE_METHODS = ["average", "ward"]
PROJECTION_METHODS = ["pca", "tsne"]


# --------------------------------------------------------------------------- #
# Helper used by every experiment's run.py
# --------------------------------------------------------------------------- #
def experiment_dirs(experiment_dir: Path, variant: str = "") -> tuple[Path, Path]:
    """Return (method_outputs, evaluation_results) under an experiment folder."""
    mo = experiment_dir / "method_outputs"
    er = experiment_dir / "evaluation_results"
    if variant:
        mo = mo / variant
        er = er / variant
    mo.mkdir(parents=True, exist_ok=True)
    er.mkdir(parents=True, exist_ok=True)
    return mo, er
