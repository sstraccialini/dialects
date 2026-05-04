"""
Configuration for Task 3: XLM-R fine-tuning experiments on FLORES+.

Four conditions are compared:
  baseline      — xlm-roberta-base, no adaptation
  mlm_wiki      — continued MLM pretraining on Wikipedia dialects
  tlm_oldi      — Translation LM on OLDI parallel pairs (Italian ↔ dialect)
  mlm_then_tlm  — Sequential: MLM on wiki, then TLM on OLDI

Evaluation is always on FLORES+ devtest (never seen in training).
The 16-variety setup is identical to multilingual_xlmr so distance matrices
can be compared directly.
"""

from __future__ import annotations

from pathlib import Path

# ---------- Paths ----------
# config.py lives in analysis/xlmr_finetuned/flores/src/, so parents[4] = repo root
REPO_ROOT = Path(__file__).resolve().parents[4]
DATASET_DIR = REPO_ROOT / "Dataset"
FLORES_DIR = DATASET_DIR / "flores" / "normalized"
WIKI_DIR = DATASET_DIR / "wiki"
WIKI_GROUP_A_DIR  = WIKI_DIR / "dialects_in_both_OLDI_and_Flores"
WIKI_LANGUAGES_DIR = WIKI_DIR / "languages"
OLDI_DIR = DATASET_DIR / "oldi" / "normalized"

METHOD_DIR = Path(__file__).resolve().parents[1]
METHOD_OUTPUTS_DIR = METHOD_DIR / "method_outputs"
EVALUATION_RESULTS_DIR = METHOD_DIR / "evaluation_results"
MODELS_DIR = METHOD_DIR / "models"
METHOD_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def outputs_subdir(variant: str = "") -> Path:
    p = METHOD_OUTPUTS_DIR if not variant else METHOD_OUTPUTS_DIR / variant
    p.mkdir(parents=True, exist_ok=True)
    return p


def evaluation_subdir(variant: str = "") -> Path:
    p = EVALUATION_RESULTS_DIR if not variant else EVALUATION_RESULTS_DIR / variant
    p.mkdir(parents=True, exist_ok=True)
    return p


def model_dir(condition: str) -> Path:
    p = MODELS_DIR / condition
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------- Varieties (same as multilingual_xlmr for direct comparability) ----------
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

# ---------- OLDI: 6 dialects with parallel data ----------
OLDI_VARIETIES = ["veneto", "siciliano", "lombardo", "sardo", "ligure", "friulano"]

# Mapping from variety slug to (Wikipedia subfolder, ISO 639-3 code).
# Used by data_loader.load_wiki_texts to find each variety's CSV.
WIKI_CODES = {
    "veneto":    "vec",
    "siciliano": "scn",
    "lombardo":  "lmo",
    "sardo":     "sc",
    "ligure":    "lij",
    "friulano":  "fur",
    "italiano":  "ita",
}

# Subfolder routing for each Wikipedia variety.
WIKI_VARIETY_DIR = {
    "vec": WIKI_GROUP_A_DIR, "scn": WIKI_GROUP_A_DIR, "lmo": WIKI_GROUP_A_DIR,
    "sc":  WIKI_GROUP_A_DIR, "lij": WIKI_GROUP_A_DIR, "fur": WIKI_GROUP_A_DIR,
    "ita": WIKI_LANGUAGES_DIR,
}

# ---------- Sampling ----------
SAMPLE_SIZE = 2009
RANDOM_STATE = 42

# ---------- Inference ----------
BASE_MODEL = "xlm-roberta-base"
MAX_LENGTH = 128
MAX_LENGTH_TLM = 256
BATCH_SIZE = 32

# ---------- Training ----------
TRAIN_BATCH_SIZE = 16
GRAD_ACCUMULATION = 4        # effective batch = 64
MLM_EPOCHS = 3
TLM_EPOCHS = 5
MLM_LR = 3e-5
TLM_LR = 3e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_WIKI_SAMPLES = 10_000    # per dialect to keep training tractable

# ---------- Clustering ----------
DISTANCE_METRIC = "cosine"
LINKAGE_METHOD = "average"

# ---------- Experimental conditions ----------
CONDITIONS = ["baseline", "mlm_wiki", "tlm_oldi", "mlm_then_tlm"]
