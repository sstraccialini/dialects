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
# This file lives in <repo>/analysis/fasttext/wiki/src/config.py
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


MODELS_DIR = METHOD_OUTPUTS_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Varieties ----------
# 13 varieties: 6 Group A dialects + 7 comparison languages.
# All codes are ISO 639-3.
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
SAMPLE_SIZE = 100_000           # max sentences per variety; smaller corpora take all
RANDOM_STATE = 42

# ---------- Preprocessing ----------
# Most flags now redundant — text is already aggressive-normalized
# at extraction time (lowercase ASCII letters + spaces only).
LOWERCASE = True
MASK_NUMBERS = False            # already stripped
NUMBER_TOKEN = " "
STRIP_PUNCT_FOR_WORD = False    # already stripped
KEEP_DIACRITICS = False         # already stripped

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
