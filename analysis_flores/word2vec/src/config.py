"""
Central configuration for the Word2Vec approach on FLORES+.

Design decisions:

1) Shared Word2Vec model (not per-variety):
   Each variety has ~2009 sentences — too few to train a reliable
   separate model. A single shared skip-gram model sees all varieties
   at once; distributional similarity of translations reflects how
   each variety uses words in the same semantic slots.

2) Skip-gram (sg=1):
   Better for rare / morphologically diverse vocabulary than CBOW on
   small corpora.

3) vector_size=100:
   Kept identical to the original Wikipedia-based baseline for
   comparability. 100 is ample for ~32k sentences.

4) min_count=2:
   Drops words that occur only once (mostly typos / proper nouns).
   Words below this threshold do NOT get vectors in Word2Vec (unlike
   FastText which uses subwords), which is the intended behaviour for
   this approach.

5) epochs=15:
   Same as the original Wikipedia pipeline; enough for convergence at
   this corpus size.

6) Preprocessing:
   - lowercase + mask numbers (same as Person 1's TF-IDF baseline)
   - keep diacritics (distinctive for Romance languages)
   - tokenizer: a Unicode-aware regex that keeps letter runs joined
     across apostrophes (e.g. "l'università", "c'è") — important for
     dialectal tokens.
"""

from __future__ import annotations

from pathlib import Path

# ---------- Paths ----------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FLORES_DIR = REPO_ROOT / "flores_data" / "flores_plus"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = RESULTS_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def results_subdir(pipeline: str) -> Path:
    """Return (and create) results/<pipeline>/. Use '' for root results dir."""
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
    ("inglese",    "germanic"),
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
    "germanic":      "Germanic (EN, DE)",
    "greek":         "Greek",
    "semitic":       "Arabic",
    "slavic":        "Slovenian",
}

GROUP_COLORS = {
    "italo_romance": "#d62728",
    "italian":       "#ff7f0e",
    "romance":       "#2ca02c",
    "germanic":      "#1f77b4",
    "greek":         "#9467bd",
    "semitic":       "#8c564b",
    "slavic":        "#e377c2",
}

# ---------- Sampling ----------
SAMPLE_SIZE = 2009
RANDOM_STATE = 42

# ---------- Preprocessing ----------
LOWERCASE = True
MASK_NUMBERS = True
NUMBER_TOKEN = " NUM "

# ---------- Word2Vec hyperparameters ----------
W2V_VECTOR_SIZE = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 2
W2V_SG = 1                # skip-gram
W2V_EPOCHS = 15
W2V_WORKERS = 4
W2V_SEED = 42

# ---------- Distances / clustering ----------
DISTANCE_METRIC = "cosine"
LINKAGE_METHOD = "average"
