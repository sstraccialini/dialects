"""
Configuration for Task 3 Sentence Baseline: Sentence Transformer fine-tuning experiments on FLORES+.

Four conditions are compared:
  baseline      — paraphrase-multilingual-MiniLM-L12-v2, no adaptation
  tsdae_wiki      — Unsupervised Denoising (TSDAE) on Wikipedia dialects
  mnrl_oldi      — Contrastive Learning (MNRL/TLM) on OLDI parallel pairs (Italian <-> dialect)
  tsdae_then_mnrl  — Sequential: unsupervised on wiki, then contrastive on OLDI

Evaluation is always on FLORES+ devtest (never seen in training).
"""

from __future__ import annotations

from pathlib import Path

# ---------- Paths ----------
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


# ---------- Varieties ----------
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
BASE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_LENGTH = 128
BATCH_SIZE = 64

# ---------- Training ----------
TRAIN_BATCH_SIZE = 16
TSDAE_EPOCHS = 3
MNRL_EPOCHS = 5
TSDAE_LR = 2e-5
MNRL_LR = 2e-5
WARMUP_RATIO = 0.1
MAX_WIKI_SAMPLES = 10_000

# ---------- Clustering ----------
DISTANCE_METRIC = "cosine"
LINKAGE_METHOD = "average"

# ---------- Experimental conditions ----------
CONDITIONS = ["baseline", "tsdae_wiki", "mnrl_oldi", "tsdae_then_mnrl"]
