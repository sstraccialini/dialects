"""
Central configuration for the CANINE (character-level encoder) approach
on FLORES+.

We embed every sentence with CANINE — a tokenizer-free transformer that
operates directly on Unicode characters. Mean-pool its character-level
hidden states, then aggregate per variety.

Why CANINE for this project:
  XLM-R / LaBSE / MiniLM all rely on a SentencePiece tokenizer trained on
  major languages only. Italo-Romance dialects (veneto, friulano, sardo,
  ...) are NOT in their training corpora, so dialect-specific characters
  (xè, ł, ç, ...) get fragmented into noisy subwords. CANINE removes
  that bottleneck: every Unicode character is a primitive of the model,
  so the dialectal orthography is preserved at the input.

Mirrors the structure of analysis/multilingual_xlmr/flores so the two
methods can be compared apples-to-apples (same 16 varieties, same
evaluation pipeline).
"""

from __future__ import annotations

from pathlib import Path

# ---------- Paths ----------
REPO_ROOT = Path(__file__).resolve().parents[4]
DATASET_DIR = REPO_ROOT / "Dataset"
FLORES_DIR = DATASET_DIR / "flores" / "normalized"

METHOD_DIR = Path(__file__).resolve().parents[1]
METHOD_OUTPUTS_DIR = METHOD_DIR / "method_outputs"
EVALUATION_RESULTS_DIR = METHOD_DIR / "evaluation_results"
METHOD_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def outputs_subdir(variant: str = "") -> Path:
    p = METHOD_OUTPUTS_DIR if not variant else METHOD_OUTPUTS_DIR / variant
    p.mkdir(parents=True, exist_ok=True)
    return p


def evaluation_subdir(variant: str = "") -> Path:
    p = EVALUATION_RESULTS_DIR if not variant else EVALUATION_RESULTS_DIR / variant
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------- Varieties (aligned with the rest of analysis/) ----------
VARIETIES = [
    ("veneto",     "italo_romance"),
    ("siciliano",  "italo_romance"),
    ("lombardo",   "italo_romance"),
    ("sardo",      "italo_romance"),
    ("ligure",     "italo_romance"),
    ("friulano",   "italo_romance"),
    ("ladino",     "italo_romance"),
    ("italiano",   "italian"),
    ("spagnolo",   "romance"),
    ("francese",   "romance"),
    ("catalano",   "romance"),
    ("tedesco",    "germanic"),
    ("inglese",    "english"),
    ("sloveno",    "slavic"),
]

VARIETY_CODES = [code for code, _ in VARIETIES]
VARIETY_GROUP = dict(VARIETIES)

VARIETY_NAMES = {
    "veneto":     "Venetian",
    "siciliano":  "Sicilian",
    "lombardo":   "Lombard",
    "sardo":      "Sardinian",
    "ligure":     "Ligurian",
    "friulano":   "Friulian",
    "ladino":     "Ladin",
    "italiano":   "Italian",
    "spagnolo":   "Spanish",
    "francese":   "French",
    "catalano":   "Catalan",
    "tedesco":    "German",
    "inglese":    "English",
    "sloveno":    "Slovenian",
}

GROUP_NAMES = {
    "italo_romance": "Italo-Romance",
    "italian":       "Italian (standard)",
    "romance":       "Romance (ES, FR, CA)",
    "germanic":      "Germanic (German)",
    "english":       "English",
    "slavic":        "Slovenian",
}

GROUP_COLORS = {
    "italo_romance": "#d62728",
    "italian":       "#ff7f0e",
    "romance":       "#2ca02c",
    "germanic":      "#1f77b4",
    "english":       "#17becf",
    "slavic":        "#e377c2",
}

# ---------- Sampling ----------
SAMPLE_SIZE = 2009
RANDOM_STATE = 42

# ---------- Model ----------
# CANINE-c: autoregressive char-LM pretraining variant. CANINE-s exists
# too (with subword loss) but -c is the closer analogue to XLM-R MLM
# for our purposes.
DEFAULT_MODEL_NAME = "google/canine-c"

# CANINE consumes raw Unicode characters. FLORES+ sentences are ~150-300
# chars on average; 512 covers ~99% with safety margin.
MAX_LENGTH = 512

# Smaller batch than XLM-R because CANINE has longer effective sequences
# and a heavier downsampling+upsampling encoder.
BATCH_SIZE = 16

# ---------- Distances / clustering ----------
DISTANCE_METRIC = "cosine"
LINKAGE_METHOD = "average"
