"""
Central configuration for the multilingual contextual-embeddings
approach on FLORES+.

We embed every sentence with a pretrained multilingual transformer
(XLM-R or mBERT), mean-pool its contextual token embeddings, then
aggregate per variety.

Design decisions:

1) Default model: XLM-R base (xlm-roberta-base).
   XLM-R has broader language coverage than mBERT, especially for
   low-resource scripts, and a cleaner objective (MLM only).
   mBERT is kept as an alternative via --model-name. Sentence-aligned
   models like
   "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
   produce better sentence-level embeddings out of the box but are
   distilled; we offer the raw encoders by default so distances
   reflect the model's own cross-lingual geometry.

2) Mean pooling with attention mask:
   The standard way to turn token embeddings into a sentence vector
   for XLM-R / mBERT (CLS works poorly without supervised fine-tuning).
   After pooling we L2-normalise each sentence, so cosine distance
   between variety centroids is numerically well-behaved.

3) Per-variety aggregation:
   Mean of the L2-normalised sentence vectors (a "mean of unit
   vectors") followed by another L2-normalisation. This is the
   standard "averaged embedding" centroid used in sentence-level
   clustering.

4) Same variety set + same labelling as the other analysis_flores modules,
   so the 16x16 distance matrices can be compared directly.

5) Linkage: average (coherent with cosine distance). The original
   multilingual baseline used Ward on cosine-derived distances, which
   is formally inconsistent; we fix that here.
"""

from __future__ import annotations

from pathlib import Path

# ---------- Paths ----------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FLORES_DIR = REPO_ROOT / "flores_data" / "flores_plus"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def results_subdir(pipeline: str) -> Path:
    """Return (and create) results/<pipeline>/. '' returns the root results dir."""
    p = RESULTS_DIR if pipeline == "" else RESULTS_DIR / pipeline
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------- Varieties (aligned with the other analysis_flores sub-projects) ----------
VARIETIES = [
    # Italo-Romance dialects
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
    ("greco",      "greek"),
    ("arabo",      "semitic"),
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
    "greco":      "Greek",
    "arabo":      "Arabic",
    "sloveno":    "Slovenian",
}

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

GROUP_COLORS = {
    "italo_romance": "#d62728",
    "italian":       "#ff7f0e",
    "romance":       "#2ca02c",
    "germanic":      "#1f77b4",
    "english":       "#17becf",  # cyan
    "greek":         "#9467bd",
    "semitic":       "#8c564b",
    "slavic":        "#e377c2",
}

# ---------- Sampling ----------
SAMPLE_SIZE = 2009
RANDOM_STATE = 42

# ---------- Model ----------
import os
# Adapted XLM-R after continued pretraining on Italian dialect Wikipedia.
# Saved by experiments/continued_pretraining/src/run_cp.py.
DEFAULT_MODEL_NAME = os.path.expanduser("~/xlmr-adapted-italian-dialects")
MAX_LENGTH = 128          # max tokens per sentence (FLORES+ ~30-80 tokens)
BATCH_SIZE = 32

# ---------- Distances / clustering ----------
DISTANCE_METRIC = "cosine"
LINKAGE_METHOD = "average"
