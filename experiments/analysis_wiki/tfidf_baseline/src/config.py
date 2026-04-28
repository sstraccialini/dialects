"""
Central configuration for the TF-IDF baseline (Person 1).

All design decisions are collected here so we have a single source of
truth and can run sensitivity experiments without touching the code.

Main decisions (taken together, documented for the report):

1) Aggregation:
   "One document per variety". We concatenate all sampled sentences for
   each language/dialect into a single super-document, obtaining a
   TF-IDF matrix of shape 14 x V. Standard formulation in computational
   dialectology and directly comparable via cosine distances.

2) Balancing:
   Sub-sampling to SAMPLE_SIZE sentences per variety (default 16000,
   equal to the observed minimum, ar ~16k). Fixed random_state for
   reproducibility. Parametrized via CLI to allow sensitivity checks
   (e.g. 50k).

3) Preprocessing:
   - lowercase: yes (standard for TF-IDF, reduces sparsity)
   - mask numbers: yes (replace digits with <NUM>, avoids Wikipedia
     dates dominating)
   - strip punctuation: ONLY for the word pipeline. For char n-grams
     punctuation is informative (apostrophes: "'a ", "l'e'", "c'e'"...)
   - diacritics: KEPT. Accents and spellings are distinctive traits
     of dialects (e.g. Venetian 'xe', Neapolitan 'e'/'e', Sicilian 'o').

4) N-gram ranges:
   - char: (3, 5). Standard VarDial/language identification.
   - word: (1, 2). Unigrams + bigrams.

5) TF-IDF hyperparameters:
   - sublinear_tf=True: use 1+log(count), dampens Wikipedia hyper-
     frequent terms.
   - min_df=1: we do NOT filter at corpus level. With only 14 aggregated
     documents, min_df=2 would drop variety-unique features, which are
     exactly the signal we're after.
   - max_df=1.0: same reason. Let IDF weight shared traits. Optionally
     revisited in sensitivity analyses.
   - max_features=None: no cap; if RAM becomes an issue set it to 100k.
   - norm='l2': normalized vectors, required for stable cosine.
"""

from pathlib import Path

# ---------- Paths ----------
# This file lives in <repo>/analysis_wiki/tfidf_baseline/src/config.py
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATASETS_DIR = REPO_ROOT / "wiki_data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def results_subdir(pipeline: str) -> Path:
    """
    Return (and create) the results subfolder for a pipeline.

    Used to organize outputs as:
        results/word/    -> word n-gram pipeline outputs
        results/char/    -> char n-gram pipeline outputs
        results/shared/  -> cross-pipeline outputs (silhouette, run stats)

    `pipeline` must be one of 'word', 'char', 'shared'.
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

# English full names used as readable labels in plots and report tables.
# CSV files keep the ISO codes since those are standard and compact.
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
# For single-language groups we show the language name directly so the
# legend matches the dot labels on the plots; for multi-language groups
# we list the members in parentheses.
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
SAMPLE_SIZE = 16000            # sentences per variety (override with --sample-size)
RANDOM_STATE = 42              # reproducibility


# ---------- Preprocessing ----------
LOWERCASE = True
MASK_NUMBERS = True            # replace \d+ with ' NUM '
NUMBER_TOKEN = " NUM "
STRIP_PUNCT_FOR_WORD = True    # strip punctuation only for word pipeline
KEEP_DIACRITICS = True         # do NOT normalize unicode


# ---------- TF-IDF (word pipeline) ----------
WORD_NGRAM_RANGE = (1, 2)
WORD_MIN_DF = 1
WORD_MAX_DF = 1.0
WORD_MAX_FEATURES = None


# ---------- TF-IDF (char pipeline) ----------
# analyzer='char_wb' means char n-grams do not cross word boundaries,
# useful to avoid spurious n-grams spanning whitespace + punctuation.
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
LINKAGE_METHODS = ["average", "ward"]   # ward requires euclidean; average is fine with cosine


# ---------- Visualization ----------
PROJECTION_METHODS = ["pca", "tsne"]    # UMAP optional if installed


# Group colors (matplotlib-friendly)
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
