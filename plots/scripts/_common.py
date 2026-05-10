"""Shared loaders, taxonomy, and styling for the plots/ scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = REPO_ROOT / "plots" / "outputs"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Taxonomy — supports BOTH legacy Italian names and new ISO codes
# --------------------------------------------------------------------------- #

FAMILY_GROUPS: Dict[str, str] = {
    # ISO codes
    "fur": "italo_romance", "lij": "italo_romance", "lmo": "italo_romance",
    "sc":  "italo_romance", "scn": "italo_romance", "vec": "italo_romance",
    "nap": "italo_romance",
    "ita": "italian",
    "spa": "romance", "fra": "romance", "cat": "romance",
    "por": "romance", "ron": "romance", "oci": "romance", "glg": "romance",
    "es": "romance", "fr": "romance", "ca": "romance",
    "deu": "germanic", "de": "germanic",
    "eng": "english",  "en": "english",
    "slv": "slavic", "hrv": "slavic", "sl": "slavic",
    "sqi": "albanian",
    "hun": "uralic",
    "el": "greek", "ar": "semitic",
    # Italian display names
    "veneto": "italo_romance", "siciliano": "italo_romance",
    "lombardo": "italo_romance", "sardo": "italo_romance",
    "ligure": "italo_romance", "friulano": "italo_romance",
    "ladino": "italo_romance", "napolitano": "italo_romance",
    "italiano": "italian",
    "spagnolo": "romance", "francese": "romance", "catalano": "romance",
    "portoghese": "romance", "rumeno": "romance",
    "occitano": "romance", "galiziano": "romance",
    "tedesco": "germanic", "inglese": "english",
    "greco": "greek", "arabo": "semitic",
    "sloveno": "slavic", "croato": "slavic",
    "albanese": "albanian", "ungherese": "uralic",
}

FAMILY_COLORS: Dict[str, str] = {
    "italo_romance": "#d62728",  # red
    "italian":       "#ff7f0e",  # orange
    "romance":       "#2ca02c",  # green
    "germanic":      "#1f77b4",  # blue
    "english":       "#17becf",  # cyan
    "slavic":        "#e377c2",  # pink
    "albanian":      "#9467bd",  # purple
    "uralic":        "#8c564b",  # brown
    "greek":         "#bcbd22",  # olive
    "semitic":       "#7f7f7f",  # grey
}

FAMILY_DISPLAY: Dict[str, str] = {
    "italo_romance": "Italo-Romance dialect",
    "italian":       "Standard Italian",
    "romance":       "Other Romance",
    "germanic":      "Germanic",
    "english":       "English",
    "slavic":        "Slavic",
    "albanian":      "Albanian",
    "uralic":        "Uralic (non-IE)",
    "greek":         "Greek",
    "semitic":       "Semitic",
}

DISPLAY_NAMES: Dict[str, str] = {
    # ISO
    "fur": "Friulian", "lij": "Ligurian", "lmo": "Lombard",
    "sc":  "Sardinian", "scn": "Sicilian", "vec": "Venetian",
    "nap": "Neapolitan",
    "ita": "Italian",
    "spa": "Spanish", "fra": "French", "cat": "Catalan",
    "por": "Portuguese", "ron": "Romanian",
    "oci": "Occitan", "glg": "Galician",
    "es": "Spanish", "fr": "French", "ca": "Catalan",
    "deu": "German", "de": "German",
    "eng": "English", "en": "English",
    "slv": "Slovenian", "sl": "Slovenian", "hrv": "Croatian",
    "sqi": "Albanian", "hun": "Hungarian",
    "el": "Greek", "ar": "Arabic",
    # Italian
    "veneto": "Venetian", "siciliano": "Sicilian", "lombardo": "Lombard",
    "sardo": "Sardinian", "ligure": "Ligurian", "friulano": "Friulian",
    "ladino": "Ladin", "napolitano": "Neapolitan",
    "italiano": "Italian",
    "spagnolo": "Spanish", "francese": "French", "catalano": "Catalan",
    "tedesco": "German", "inglese": "English",
    "greco": "Greek", "arabo": "Arabic",
    "sloveno": "Slovenian",
}


# --------------------------------------------------------------------------- #
# Geographic centroids of the Italo-Romance dialect regions (approx)
#   Italian peninsula: ~36-47°N, 6-19°E
# --------------------------------------------------------------------------- #
DIALECT_COORDS: Dict[str, Tuple[float, float]] = {
    # (lat, lon) in degrees
    "friulano":   (46.06, 13.23),  # Udine, Friuli
    "fur":        (46.06, 13.23),
    "ladino":     (46.42, 11.85),  # Bolzano area, Dolomites
    "veneto":     (45.43, 12.33),  # Venice, Veneto
    "vec":        (45.43, 12.33),
    "lombardo":   (45.46, 9.19),   # Milan, Lombardy
    "lmo":        (45.46, 9.19),
    "ligure":     (44.41, 8.93),   # Genoa, Liguria
    "lij":        (44.41, 8.93),
    "napolitano": (40.85, 14.27),  # Naples, Campania
    "nap":        (40.85, 14.27),
    "siciliano":  (37.60, 14.02),  # central Sicily
    "scn":        (37.60, 14.02),
    "sardo":      (40.12, 9.01),   # central Sardinia
    "sc":         (40.12, 9.01),
    # standard Italian — Roma as the anchor
    "italiano":   (41.90, 12.49),
    "ita":        (41.90, 12.49),
}

ITALO_ROMANCE_DIALECTS = {
    "friulano", "ladino", "veneto", "lombardo", "ligure", "napolitano",
    "siciliano", "sardo",
    "fur", "vec", "lmo", "lij", "nap", "scn", "sc",
}


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

def load_npz(path: str) -> Tuple[np.ndarray, List[str]]:
    """Return (matrix, codes) from variety_vectors.npz."""
    d = np.load(path, allow_pickle=True)
    return d["matrix"].astype(np.float32), [str(c) for c in d["labels"]]


def load_csv_vectors(path: str) -> Tuple[np.ndarray, List[str]]:
    """Return (matrix, codes) from a CSV with the variety code as index."""
    df = pd.read_csv(path, index_col=0)
    return df.values.astype(np.float32), list(df.index)


def load_similarity_csv(path: str) -> Tuple[np.ndarray, List[str]]:
    """Return (similarity_matrix, codes) from similarity_matrix.csv."""
    df = pd.read_csv(path, index_col=0)
    return df.values.astype(np.float32), list(df.index)


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between rows of X."""
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-12)
    return Xn @ Xn.T


# --------------------------------------------------------------------------- #
# Method registry — every entry is a self-describing source of distances
# --------------------------------------------------------------------------- #

# (label, kind, path, taxonomy_hint)
#   kind in {"npz", "csv_vectors", "csv_sim"}
#   taxonomy_hint in {"italian_names", "iso"}
SAVED_VECTOR_SOURCES: List[Dict] = [
    {
        "label": "CANINE (FLORES, old)",
        "short": "canine_old",
        "kind":  "npz",
        "path":  "analysis/canine/old_experiments/flores/method_outputs/variety_vectors.npz",
        "taxonomy": "italian_names",
    },
    {
        "label": "XLM-R (FLORES, old)",
        "short": "xlmr_old",
        "kind":  "npz",
        "path":  "analysis/multilingual_xlmr/old_experiments/flores/method_outputs/variety_vectors.npz",
        "taxonomy": "italian_names",
    },
    {
        "label": "Word2Vec (FLORES, old)",
        "short": "w2v_flores_old",
        "kind":  "npz",
        "path":  "analysis/word2vec/old_experiments/flores/method_outputs/variety_vectors.npz",
        "taxonomy": "italian_names",
    },
    {
        "label": "Word2Vec (Wiki, old)",
        "short": "w2v_wiki_old",
        "kind":  "npz",
        "path":  "analysis/word2vec/old_experiments/wiki/method_outputs/variety_vectors.npz",
        "taxonomy": "iso_short",
    },
    {
        "label": "Sentence-finetuned baseline (FLORES, old)",
        "short": "sent_finetuned",
        "kind":  "csv_vectors",
        "path":  "analysis/sentence_finetuned/flores/method_outputs/baseline/variety_vectors.csv",
        "taxonomy": "italian_names",
    },
]

NEW_SIM_SOURCES: List[Dict] = [
    {
        "label": "CANINE TLM-OLDI (new)",
        "short": "canine_tlm",
        "path":  "analysis/canine/experiments/tlm_oldi_to_flores/evaluation_results/flores/centroid/similarity_matrix.csv",
    },
    {
        "label": "CANINE MLM-Wiki (new)",
        "short": "canine_mlm_wiki",
        "path":  "analysis/canine/experiments/mlm_wiki_to_flores_oldi/evaluation_results/flores/centroid/similarity_matrix.csv",
    },
    {
        "label": "CANINE MLM-Wiki+TLM-OLDI",
        "short": "canine_mlm_tlm",
        "path":  "analysis/canine/experiments/mlm_wiki_then_tlm_oldi_to_flores/evaluation_results/flores/centroid/similarity_matrix.csv",
    },
    {
        "label": "XLM-R TLM-OLDI",
        "short": "xlmr_tlm",
        "path":  "analysis/multilingual_xlmr/experiments/tlm_oldi_to_flores/evaluation_results/flores/centroid/similarity_matrix.csv",
    },
    {
        "label": "XLM-R MLM-Wiki",
        "short": "xlmr_mlm_wiki",
        "path":  "analysis/multilingual_xlmr/experiments/mlm_wiki_to_flores_oldi/evaluation_results/flores/centroid/similarity_matrix.csv",
    },
    {
        "label": "XLM-R MLM-Wiki+TLM-OLDI",
        "short": "xlmr_mlm_tlm",
        "path":  "analysis/multilingual_xlmr/experiments/mlm_wiki_then_tlm_oldi_to_flores/evaluation_results/flores/centroid/similarity_matrix.csv",
    },
    {
        "label": "Sentence-MiniLM MNRL-OLDI",
        "short": "minilm_mnrl",
        "path":  "analysis/sentence_minilm/experiments/mnrl_oldi_to_flores/evaluation_results/flores/centroid/similarity_matrix.csv",
    },
    {
        "label": "FastText (Wiki)",
        "short": "fasttext",
        "path":  "analysis/fasttext/experiments/wiki_to_flores_oldi/evaluation_results/flores/centroid/similarity_matrix.csv",
    },
    {
        "label": "Word2Vec (Wiki, new)",
        "short": "w2v_new",
        "path":  "analysis/word2vec/experiments/wiki_to_flores_oldi/evaluation_results/flores/centroid/similarity_matrix.csv",
    },
]


def load_source(src: Dict) -> Tuple[np.ndarray, List[str]]:
    """Return (similarity_matrix, codes) for any registry entry.

    For raw-vector sources, computes cosine similarity on the fly.
    For similarity-matrix sources, returns directly.
    """
    p = REPO_ROOT / src["path"]
    if src.get("kind") == "csv_sim" or "similarity_matrix" in src["path"]:
        return load_similarity_csv(p)
    if src.get("kind") == "npz":
        X, codes = load_npz(p)
        return cosine_similarity_matrix(X), codes
    if src.get("kind") == "csv_vectors":
        X, codes = load_csv_vectors(p)
        return cosine_similarity_matrix(X), codes
    raise ValueError(f"Unknown source kind: {src}")


def family_of(code: str) -> str:
    return FAMILY_GROUPS.get(code, "other")


def color_of(code: str) -> str:
    return FAMILY_COLORS.get(family_of(code), "#888888")


def display_of(code: str) -> str:
    return DISPLAY_NAMES.get(code, code)


def family_order() -> List[str]:
    return [
        "italo_romance", "italian", "romance",
        "germanic", "english",
        "slavic", "albanian", "uralic",
        "greek", "semitic",
    ]


def sort_codes_by_family(codes: List[str]) -> List[str]:
    order = {f: i for i, f in enumerate(family_order())}
    return sorted(codes, key=lambda c: (order.get(family_of(c), 99), c))
