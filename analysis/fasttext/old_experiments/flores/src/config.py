"""
Central configuration for the Subword / FastText approach on FLORES+.

Two pipelines:
    fasttext — gensim FastText (subword char n-grams, skip-gram)
    bpe      — SentencePiece BPE tokenization + TF-IDF on BPE pieces

Design decisions:

1) Shared FastText model (not per-variety):
   Each variety has only 2009 sentences — too few to train a reliable
   per-variety model. A single shared model lets the subword mechanism
   generalize across morphologically related forms across dialects.

2) FastText skip-gram (sg=1):
   Better for rare / morphologically diverse words; superior to CBOW
   on small, heterogeneous corpora (Bojanowski et al., 2017).

3) FastText min_n=3, max_n=6:
   Covers typical morpheme sizes in Italian and Romance languages.
   min_n=3 avoids trivially shared character pairs; max_n=6 captures
   longer suffixes and roots.

4) FastText min_count=2:
   Words appearing only once are often typos/proper nouns; OOV words
   are still embedded via their subword n-grams, so nothing is lost.

5) BPE vocab_size=8000:
   Large enough to cover the 16-variety vocabulary without exploding,
   small enough to stay interpretable. Standard for sentence-level BPE.

6) BPE + TF-IDF:
   BPE pieces are fed to TF-IDF (same hyperparameters as Person 1's
   word pipeline) to keep the comparison clean. The only variable is
   the tokenization unit.

7) Preprocessing: same as Person 1 (lowercase, mask numbers, keep
   diacritics, keep punctuation — important for subword models).
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
    """method_outputs/[<variant>/] - vectors, models, run_stats, top_features."""
    p = METHOD_OUTPUTS_DIR if not variant else METHOD_OUTPUTS_DIR / variant
    p.mkdir(parents=True, exist_ok=True)
    return p


def evaluation_subdir(variant: str = "") -> Path:
    """evaluation_results/[<variant>/] - written by the central evaluation module."""
    p = EVALUATION_RESULTS_DIR if not variant else EVALUATION_RESULTS_DIR / variant
    p.mkdir(parents=True, exist_ok=True)
    return p


# Trained model artefacts (FastText .bin, BPE .model / .vocab) live under method_outputs/.
MODELS_DIR = METHOD_OUTPUTS_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Varieties ----------
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
    "english":       "#17becf",  # cyan
    "slavic":        "#e377c2",
}

# ---------- Sampling ----------
SAMPLE_SIZE = 2009
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
FT_MIN_COUNT = 2            # min frequency to add a word to the vocab
                            # (OOV words still embedded via subwords)
FT_MIN_N = 3                # min char n-gram length (subword)
FT_MAX_N = 6                # max char n-gram length (subword)
FT_EPOCHS = 15              # a bit higher than baseline Wikipedia since
                            # the corpus is much smaller (~32k vs ~224k)
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
