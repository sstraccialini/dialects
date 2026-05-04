"""
Central configuration for the Sentence-embedding baseline on Wikipedia data.

Pipeline:
    sentence — pretrained multilingual sentence-transformer
               (paraphrase-multilingual-MiniLM-L12-v2 by default)

Design decisions:

1) One variety = one vector.
   For each variety we encode a large sample of sentences through the
   sentence-transformer, average them inside each Wikipedia article
   (via `article_id`), then average article vectors to obtain one
   variety-level embedding. This two-stage aggregation is more robust
   than a flat mean over ~16k unrelated sentences because each article
   gets equal weight regardless of how many sentences it produced.

2) Paraphrase-multilingual MiniLM.
   Compact (~118M params), fast on CPU, covers 50+ languages and is
   trained with a bi-encoder contrastive objective: semantically
   similar sentences end up close in the embedding space independent
   of surface form. Reuses the same transformer-backbone logic as
   Person 4's multilingual approach but at the sentence level.

3) Cosine distance + average linkage.
   Same distance / clustering conventions as the other four
   analysis_wiki methods so the results are directly comparable.

4) Preprocessing: none beyond the cleaning already performed in
   data/generation.py. Sentence-transformers do their own normalization
   and tokenization, so we do not lowercase, mask numbers or strip
   punctuation here.
"""

from __future__ import annotations

from pathlib import Path

# ---------- Paths ----------
# This file lives in <repo>/analysis/sentence_baseline/wiki/src/config.py
REPO_ROOT = Path(__file__).resolve().parents[4]
DATASET_DIR = REPO_ROOT / "Dataset"
WIKI_DIR = DATASET_DIR / "wiki"
GROUP_A_DIR   = WIKI_DIR / "dialects_in_both_OLDI_and_Flores"
LANGUAGES_DIR = WIKI_DIR / "languages"

METHOD_DIR = Path(__file__).resolve().parents[1]
METHOD_OUTPUTS_DIR = METHOD_DIR / "method_outputs"
EVALUATION_RESULTS_DIR = METHOD_DIR / "evaluation_results"
METHOD_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def outputs_subdir(variant: str = "") -> Path:
    """method_outputs/[<variant>/] - vectors, models, run_stats, top_features."""
    p = METHOD_OUTPUTS_DIR if not variant else METHOD_OUTPUTS_DIR / variant
    p.mkdir(parents=True, exist_ok=True)
    return p


def evaluation_subdir(variant: str = "") -> Path:
    """evaluation_results/[<variant>/] - written by the central evaluation module."""
    p = EVALUATION_RESULTS_DIR if not variant else EVALUATION_RESULTS_DIR / variant
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------- Varieties ----------
# 13 varieties: 6 Group A dialects + 7 comparison languages. ISO 639-3.
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
    # Non-Romance
    ("deu", "germanic"),
    ("slv", "slavic"),
    ("eng", "english"),
]

VARIETY_CODES = [code for code, _ in VARIETIES]
VARIETY_GROUP = dict(VARIETIES)

# Mapping variety → which subfolder its CSV lives in.
VARIETY_DIR = {
    "fur": GROUP_A_DIR, "lij": GROUP_A_DIR, "lmo": GROUP_A_DIR,
    "sc":  GROUP_A_DIR, "scn": GROUP_A_DIR, "vec": GROUP_A_DIR,
    "ita": LANGUAGES_DIR, "spa": LANGUAGES_DIR, "fra": LANGUAGES_DIR,
    "cat": LANGUAGES_DIR, "deu": LANGUAGES_DIR, "slv": LANGUAGES_DIR,
    "eng": LANGUAGES_DIR,
}

# Subsets used by the dialect-vs-modern-language analysis in
# run_sentence_baseline.py (similarity table + rankings).
DIALECT_CODES = ["fur", "lij", "lmo", "sc", "scn", "vec"]
MODERN_LANGUAGE_CODES = ["ita", "spa", "fra", "cat", "deu", "slv", "eng"]

# English full names used as readable labels in plots and report tables.
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
    "deu": "German",
    "slv": "Slovenian",
    "eng": "English",
}

# English labels for groups/families (used in legends).
GROUP_NAMES = {
    "italo_romance": "Italo-Romance",
    "italian":       "Italian (standard)",
    "romance":       "Romance (ES, FR, CA)",
    "germanic":      "Germanic (German)",
    "english":       "English",
    "slavic":        "Slovenian",
}


# ---------- Sampling ----------
SAMPLE_SIZE = 100_000          # max sentences per variety; smaller corpora take all
RANDOM_STATE = 42              # reproducibility


# ---------- Sentence-transformer ----------
SENTENCE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SENTENCE_BATCH_SIZE = 32
ARTICLE_AGGREGATION = "mean"   # sentence -> article
VARIETY_AGGREGATION = "mean"   # article  -> variety


# ---------- Distances / clustering ----------
DISTANCE_METRIC = "cosine"
LINKAGE_METHOD = "average"     # average linkage is sensible with cosine


# ---------- Visualization ----------
# Group colors (matplotlib-friendly). Shared across all analysis_wiki
# pipelines so plots are directly comparable.
GROUP_COLORS = {
    "italo_romance": "#d62728",  # red
    "italian":       "#ff7f0e",  # orange
    "romance":       "#2ca02c",  # green
    "germanic":      "#1f77b4",  # blue
    "english":       "#17becf",  # cyan
    "slavic":        "#e377c2",  # pink
}
