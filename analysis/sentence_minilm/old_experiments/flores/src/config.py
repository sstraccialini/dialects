"""
Central configuration for the Sentence-embedding baseline on FLORES+.

Pipeline:
    sentence — pretrained multilingual sentence-transformer
               (paraphrase-multilingual-MiniLM-L12-v2 by default)

Design decisions:

1) One variety = one vector.
   FLORES+ ships 2009 parallel sentences per variety (997 dev + 1012
   devtest). We encode every sentence through the sentence-transformer
   and average the (L2-normalised) embeddings to obtain one vector
   per variety. The parallel structure guarantees that every variety
   is represented by translations of the same underlying content, so
   the variety centroid is directly comparable across languages.

2) Paraphrase-multilingual MiniLM.
   Compact (~118M params), fast on CPU, covers 50+ languages and is
   trained with a bi-encoder contrastive objective: semantically
   similar sentences end up close in the embedding space independent
   of surface form. Same backbone as the sentence baseline on
   Wikipedia, so results are directly comparable across corpora.

3) Cosine distance + average linkage.
   Same distance / clustering conventions as the other four
   analysis_flores methods (TF-IDF, Word2Vec, subword/FastText,
   multilingual) so the 16x16 distance matrices can be compared
   directly.

4) Preprocessing: none.
   Sentence-transformers do their own normalization and tokenization,
   and FLORES+ text is already cleaned.

5) Parallelism replaces article aggregation.
   The Wikipedia variant averages sentence embeddings first inside
   each `article_id` and then across articles. FLORES+ has no article
   structure — every row is an independent, parallel sentence — so
   we use the single-stage flat mean here. Sentence parallelism is
   itself a much stronger anchor than article bucketing.
"""

from __future__ import annotations

from pathlib import Path

# ---------- Paths ----------
# This file lives in <repo>/analysis_flores/sentence_baseline/src/config.py
REPO_ROOT = Path(__file__).resolve().parents[4]
DATASET_DIR = REPO_ROOT / "Dataset"
FLORES_DIR = DATASET_DIR / "flores" / "normalized"

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
# (slug, family). Slugs match filenames (without .txt) in FLORES_DIR
# and column names in the parallel FLORES+ TSV. Family is used to
# color visualizations and compute silhouette scores.
VARIETIES = [
    # Italo-Romance dialects (7)
    ("veneto",     "italo_romance"),
    ("siciliano",  "italo_romance"),
    ("lombardo",   "italo_romance"),
    ("sardo",      "italo_romance"),
    ("ligure",     "italo_romance"),
    ("friulano",   "italo_romance"),
    ("ladino",     "italo_romance"),
    # Standard Italian
    ("italiano",   "italian"),
    # Other Romance
    ("spagnolo",   "romance"),
    ("francese",   "romance"),
    ("catalano",   "romance"),
    # Non-Romance
    ("tedesco",    "germanic"),
    ("inglese",    "english"),
    ("sloveno",    "slavic"),
]

VARIETY_CODES = [code for code, _ in VARIETIES]
VARIETY_GROUP = dict(VARIETIES)

# Subsets used in the dialect-vs-modern-language similarity table
# produced by run_sentence_baseline.py.
DIALECT_CODES = [
    "veneto", "siciliano", "lombardo", "sardo",
    "ligure", "friulano", "ladino",
]
MODERN_LANGUAGE_CODES = [
    "italiano", "spagnolo", "francese", "catalano",
    "tedesco", "inglese", "sloveno",
]

# English full names used as readable labels in plots and report tables.
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
# FLORES+ ships exactly 2009 sentences per variety (997 dev + 1012
# devtest). Default: use all of them. Lower only for sensitivity.
SAMPLE_SIZE = 2009
RANDOM_STATE = 42


# ---------- Sentence-transformer ----------
SENTENCE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SENTENCE_BATCH_SIZE = 32
VARIETY_AGGREGATION = "mean"   # sentence -> variety centroid


# ---------- Distances / clustering ----------
DISTANCE_METRIC = "cosine"
LINKAGE_METHOD = "average"     # average linkage is sensible with cosine


# ---------- Visualization ----------
# Group colors (matplotlib-friendly). Shared across all four
# analysis_flores pipelines so plots are directly comparable.
GROUP_COLORS = {
    "italo_romance": "#d62728",  # red
    "italian":       "#ff7f0e",  # orange
    "romance":       "#2ca02c",  # green
    "germanic":      "#1f77b4",  # blue
    "english":       "#17becf",  # cyan
    "slavic":        "#e377c2",  # pink
}
