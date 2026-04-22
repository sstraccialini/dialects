"""
Central configuration for the TF-IDF pipeline on FLORES+.

All design decisions are collected here so we have a single source of
truth and can run sensitivity experiments without touching the code.

Main decisions (documented for the report):

1) Aggregation:
   "One document per variety". We concatenate all sentences of each
   variety into a single super-document, obtaining a TF-IDF matrix of
   shape 16 x V. Standard in computational dialectology, directly
   comparable via cosine distances.

2) Balancing:
   FLORES+ gives us exactly 2009 sentences per variety (997 dev +
   1012 devtest). No sub-sampling needed: all varieties are naturally
   balanced. SAMPLE_SIZE is kept as a configurable upper bound and
   defaults to 2009 ("take everything"). Lower it only for sensitivity
   checks.

3) Preprocessing:
   - lowercase: yes (standard for TF-IDF, reduces sparsity)
   - mask numbers: yes (replace digits with <NUM>, reduces noise from
     dates and numeric literals)
   - strip punctuation: ONLY for the word pipeline. For char n-grams
     punctuation is informative (apostrophes: "'a ", "l'e'", "c'e'" ...).
   - diacritics: KEPT. Accents and spellings are distinctive traits
     of dialects (e.g. Venetian 'xe', Neapolitan 'e'/'e', Sicilian 'o').

4) N-gram ranges:
   - char: (3, 5). Standard VarDial / language identification.
   - word: (1, 2). Unigrams + bigrams.

5) TF-IDF hyperparameters:
   - sublinear_tf=True: 1 + log(count), dampens hyper-frequent terms.
   - min_df=1: we do NOT filter at corpus level. With only 16
     aggregated documents, min_df=2 would drop variety-unique features,
     which are precisely the signal we are after.
   - max_df=1.0: same reason. Let IDF weight shared traits.
   - max_features=None: no cap.
   - norm='l2': normalized vectors, required for stable cosine.
"""

from __future__ import annotations

from pathlib import Path

# ---------- Paths ----------
# This file lives in <repo>/analysis_flores/tfidf/src/config.py
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FLORES_DIR = REPO_ROOT / "flores_data" / "flores_plus"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def results_subdir(pipeline: str) -> Path:
    """
    Return (and create) the results subfolder for a pipeline.

    Used to organize outputs as:
        results/word/    -> word n-gram pipeline outputs
        results/char/    -> char n-gram pipeline outputs
        results/shared/  -> cross-pipeline outputs (silhouette, run stats)
    """
    p = RESULTS_DIR / pipeline
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------- Varieties ----------
# (slug, family). The slug is the filename (without .txt) inside
# FLORES_DIR and also the column name in the parallel TSV. The family
# is used to color visualizations and to compute silhouette scores.
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
    ("inglese",    "germanic"),
    ("greco",      "greek"),
    ("arabo",      "semitic"),
    ("sloveno",    "slavic"),
]

VARIETY_CODES = [code for code, _ in VARIETIES]
VARIETY_GROUP = dict(VARIETIES)

# English full names used as readable labels in plots and report tables.
# TXT files keep the Italian slugs since those match the FLORES+ column
# names and are easy to read in the data folder.
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

# English labels for families (used in legends).
GROUP_NAMES = {
    "italo_romance": "Italo-Romance",
    "italian":       "Italian (standard)",
    "romance":       "Romance (ES, FR, CA)",
    "germanic":      "Germanic (EN, DE)",
    "greek":         "Greek",
    "semitic":       "Arabic",
    "slavic":        "Slovenian",
}


# ---------- Sampling ----------
# FLORES+ ships exactly 2009 sentences per variety (997 dev + 1012
# devtest). Default: use all of them. Lower only for sensitivity.
SAMPLE_SIZE = 2009
RANDOM_STATE = 42


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
# analyzer='char_wb' means char n-grams do not cross word boundaries:
# avoids spurious n-grams spanning whitespace + punctuation.
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
LINKAGE_METHOD = "average"     # average linkage is sensible with cosine


# ---------- Visualization ----------
# Group colors (matplotlib-friendly). Shared across all four analysis_flores
# pipelines so plots are directly comparable.
GROUP_COLORS = {
    "italo_romance": "#d62728",  # red
    "italian":       "#ff7f0e",  # orange
    "romance":       "#2ca02c",  # green
    "germanic":      "#1f77b4",  # blue
    "greek":         "#9467bd",  # purple
    "semitic":       "#8c564b",  # brown
    "slavic":        "#e377c2",  # pink
}
