"""
Central configuration for the Subword / FastText approach (Person 5).

Two pipelines:
    fasttext — gensim FastText (subword char n-grams, skip-gram)
    bpe      — SentencePiece BPE tokenization + TF-IDF on BPE pieces

Design decisions:

1) Shared FastText model (not per-variety):
   Each variety has ~16k sentences — too few to train a reliable
   per-variety model. A single shared model lets the subword mechanism
   generalize across morphologically related forms across dialects.

2) FastText skip-gram (sg=1):
   Better for rare / morphologically diverse words; superior to CBOW
   on small, heterogeneous corpora (Bojanowski et al., 2017).

3) FastText min_n=3, max_n=6:
   Covers typical morpheme sizes in Italian and Romance languages.
   min_n=3 avoids trivially shared character pairs; max_n=6 captures
   longer suffixes and roots.

4) BPE vocab_size=8000:
   Large enough to cover the 14-variety vocabulary without exploding,
   small enough to stay interpretable. Standard for sentence-level BPE.

5) BPE + TF-IDF:
   BPE pieces are fed to TF-IDF (same hyperparameters as Person 1's
   word pipeline) to keep the comparison clean. The only variable is
   the tokenization unit.

6) Preprocessing: same as Person 1 (lowercase, mask numbers, keep
   diacritics, keep punctuation — important for subword models).
"""

from pathlib import Path

# ---------- Paths ----------
# This file lives in <repo>/analysis_wiki/subword_fasttext/src/config.py
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATASETS_DIR = REPO_ROOT / "wiki_data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Trained model artefacts (FastText .bin, BPE .model / .vocab)
MODELS_DIR = RESULTS_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def results_subdir(pipeline: str) -> Path:
    """Return (and create) results/<pipeline>/."""
    p = RESULTS_DIR / pipeline
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------- Varieties ----------
VARIETIES = [
    # Italo-Romance dialects
    ("nap", "italo_romance"),
    ("scn", "italo_romance"),
    ("vec", "italo_romance"),
    ("lmo", "italo_romance"),
    ("sc",  "italo_romance"),
    # Standard Italian
    ("ita", "italian"),
    # Other Romance
    ("es",  "romance"),
    ("fr",  "romance"),
    ("ca",  "romance"),
    # Non-Romance
    ("de",  "germanic"),
    ("el",  "greek"),
    ("ar",  "semitic"),
    ("sl",  "slavic"),
    # Control
    ("en",  "germanic"),
]

VARIETY_CODES = [code for code, _ in VARIETIES]
VARIETY_GROUP = dict(VARIETIES)

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
SAMPLE_SIZE = 16000
RANDOM_STATE = 42

# ---------- Preprocessing ----------
LOWERCASE = True
MASK_NUMBERS = True
NUMBER_TOKEN = " NUM "
STRIP_PUNCT_FOR_WORD = True
KEEP_DIACRITICS = True

# ---------- FastText hyperparameters ----------
FT_VECTOR_SIZE = 200        # embedding dimension
FT_WINDOW = 5               # context window
FT_MIN_COUNT = 2            # min frequency to add to vocabulary
                            # (OOV words still embedded via subwords)
FT_MIN_N = 3                # min char n-gram length (subword)
FT_MAX_N = 6                # max char n-gram length (subword)
FT_EPOCHS = 10
FT_SG = 1                   # 1 = skip-gram
FT_WORKERS = 4

# ---------- BPE hyperparameters ----------
BPE_VOCAB_SIZE = 8000
BPE_CHARACTER_COVERAGE = 0.9995
BPE_MODEL_PREFIX = str(MODELS_DIR / "bpe_model")

# ---------- TF-IDF on BPE tokens ----------
BPE_TFIDF_MIN_DF = 1
BPE_TFIDF_MAX_DF = 1.0
BPE_TFIDF_MAX_FEATURES = None
SUBLINEAR_TF = True
NORM = "l2"

# ---------- Distances / clustering ----------
DISTANCE_METRIC = "cosine"
LINKAGE_METHOD = "average"
