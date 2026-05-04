"""
Central configuration for the TF-IDF baseline (Wikipedia corpus).

All design decisions are collected here so we have a single source of
truth and can run sensitivity experiments without touching the code.

Main decisions (taken together, documented for the report):

1) Aggregation:
   "One document per variety". We concatenate all sampled sentences for
   each language/dialect into a single super-document, obtaining a
   TF-IDF matrix of shape 13 x V. Standard formulation in computational
   dialectology and directly comparable via cosine distances.

2) Balancing:
   Sub-sampling to SAMPLE_SIZE sentences per variety (default 100,000).
   Varieties with fewer than that take all of their sentences. Fixed
   random_state for reproducibility. Parametrized via CLI.

3) Preprocessing:
   - lowercase: yes (standard for TF-IDF, reduces sparsity). Already
     applied at extraction time by `aggressive_normalize`, so this is
     idempotent.
   - mask numbers: not needed — aggressive normalize already strips
     digits at extraction time.
   - strip punctuation: not needed — same reason.
   - diacritics: already stripped at extraction time.

4) N-gram ranges:
   - char: (3, 5). Standard VarDial/language identification.
   - word: (1, 2). Unigrams + bigrams.

5) TF-IDF hyperparameters:
   - sublinear_tf=True: use 1+log(count), dampens hyper-frequent terms.
   - min_df=1, max_df=1.0: with only 13 aggregated documents, filtering
     would drop variety-unique features that are exactly the signal.
   - max_features=None: no cap.
   - norm='l2': normalized vectors, required for stable cosine.
"""

from pathlib import Path

# ---------- Paths ----------
# This file lives in <repo>/analysis/tfidf/wiki/src/config.py
REPO_ROOT = Path(__file__).resolve().parents[4]
DATASET_DIR = REPO_ROOT / "Dataset"
WIKI_DIR = DATASET_DIR / "wiki"

# Subfolder where each variety's CSV lives.
GROUP_A_DIR    = WIKI_DIR / "dialects_in_both_OLDI_and_Flores"
LANGUAGES_DIR  = WIKI_DIR / "languages"

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
# (code, family_group). All codes are ISO 639-3 (3 letters).
# 13 varieties: 6 dialects (Group A — OLDI ∩ FLORES) + 7 comparison languages.
VARIETIES = [
    # Italo-Romance dialects (Group A — present in both OLDI and FLORES+)
    ("fur", "italo_romance"),   # Friulian
    ("lij", "italo_romance"),   # Ligurian
    ("lmo", "italo_romance"),   # Lombard
    ("sc",  "italo_romance"),   # Sardinian
    ("scn", "italo_romance"),   # Sicilian
    ("vec", "italo_romance"),   # Venetian
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

# English full names for plot labels and report tables.
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

# Family/group display names for legends.
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


# ---------- Preprocessing ----------
# Most of these are now idempotent because aggressive_normalize already
# applied them at extraction time. Kept for documentation / safety.
LOWERCASE = True
MASK_NUMBERS = False           # already stripped at extraction
NUMBER_TOKEN = " "
STRIP_PUNCT_FOR_WORD = False   # already stripped
KEEP_DIACRITICS = False        # already stripped


# ---------- TF-IDF (word pipeline) ----------
WORD_NGRAM_RANGE = (1, 2)
WORD_MIN_DF = 1
WORD_MAX_DF = 1.0
WORD_MAX_FEATURES = None


# ---------- TF-IDF (char pipeline) ----------
CHAR_NGRAM_RANGE = (3, 5)
CHAR_ANALYZER = "char_wb"
CHAR_MIN_DF = 1
CHAR_MAX_DF = 1.0
CHAR_MAX_FEATURES = None


# ---------- TF-IDF (common) ----------
SUBLINEAR_TF = True
NORM = "l2"


# ---------- Distances / clustering ----------
DISTANCE_METRIC = "cosine"
LINKAGE_METHODS = ["average", "ward"]


# ---------- Visualization ----------
PROJECTION_METHODS = ["pca", "tsne"]


# Group colors (matplotlib-friendly).
GROUP_COLORS = {
    "italo_romance": "#d62728",   # red
    "italian":       "#ff7f0e",   # orange
    "romance":       "#2ca02c",   # green
    "germanic":      "#1f77b4",   # blue
    "english":       "#17becf",   # cyan
    "slavic":        "#e377c2",   # pink
}
