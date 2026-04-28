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
# This file lives in <repo>/analysis_wiki/sentence_baseline/src/config.py
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATASETS_DIR = REPO_ROOT / "wiki_data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def results_subdir(pipeline: str) -> Path:
    """
    Return (and create) the results subfolder for a pipeline.

    Used to organize outputs as:
        results/sentence/  -> sentence-embedding pipeline outputs
        results/shared/    -> cross-pipeline outputs (silhouette, run stats)
    """
    p = RESULTS_DIR / pipeline
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------- Varieties ----------
# (code, group). The code is the prefix of the CSV file in wiki_data/.
# The group is used to color visualizations and compute silhouette
# against external labels.
VARIETIES = [
    # Italo-Romance dialects
    ("nap", "italo_romance"),
    ("scn", "italo_romance"),
    ("vec", "italo_romance"),
    ("lmo", "italo_romance"),
    ("sc",  "italo_romance"),
    # Standard Italian
    ("ita", "italian"),
    # Other Romance (direct contact + genealogy)
    ("es",  "romance"),
    ("fr",  "romance"),
    ("ca",  "romance"),
    # Non-Romance with potential historical contact
    ("de",  "germanic"),
    ("el",  "greek"),
    ("ar",  "semitic"),
    ("sl",  "slavic"),
    # Control
    ("en",  "english"),
]

VARIETY_CODES = [code for code, _ in VARIETIES]
VARIETY_GROUP = dict(VARIETIES)

# Subsets used by the dialect-vs-modern-language analysis in
# run_sentence_baseline.py (similarity table + rankings).
DIALECT_CODES = ["nap", "scn", "vec", "lmo", "sc"]
MODERN_LANGUAGE_CODES = ["ita", "es", "fr", "ca", "de", "el", "ar", "sl", "en"]

# English full names used as readable labels in plots and report tables.
VARIETY_NAMES = {
    "nap": "Neapolitan",
    "scn": "Sicilian",
    "vec": "Venetian",
    "lmo": "Lombard",
    "sc":  "Sardinian",
    "ita": "Italian",
    "es":  "Spanish",
    "fr":  "French",
    "ca":  "Catalan",
    "de":  "German",
    "el":  "Greek",
    "ar":  "Arabic",
    "sl":  "Slovenian",
    "en":  "English",
}

# English labels for groups/families (used in legends).
GROUP_NAMES = {
    "italo_romance": "Italo-Romance",
    "italian":       "Italian (standard)",
    "romance":       "Romance (ES, FR, CA)",
    "germanic":      "Germanic (German)",
    "english":       "English",
    "greek":         "Greek",
    "semitic":       "Arabic",
    "slavic":        "Slovenian",
}


# ---------- Sampling ----------
SAMPLE_SIZE = 16000            # sentences per variety (override with CLI)
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
    "greek":         "#9467bd",  # purple
    "semitic":       "#8c564b",  # brown
    "slavic":        "#e377c2",  # pink
}
