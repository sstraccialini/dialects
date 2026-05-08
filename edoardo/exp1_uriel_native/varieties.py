"""
Varieties for Experiment 1: dialects with NATIVE coverage in URIEL/lang2vec.

Selection rationale: every dialect here has its own entry in
``URIEL_LANGUAGES`` (no k-NN imputation).  This lets us anchor the
analysis on URIEL as a reasonably reliable gold instead of fighting
against URIEL's silent imputation for missing dialects.

Note on test set: FLORES+ does NOT cover nap/pms/eml.  Evaluation here
uses an 80/20 hold-out split of the Wikipedia text per variety; the
"centroid" is computed on the held-out 20%.  See
``edoardo/exp1_uriel_native/README.md``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List


# 13 varieties: 6 dialects with native URIEL entry + 7 standards.
VARIETIES: List[tuple[str, str]] = [
    # Italo-Romance dialects (all natively in URIEL)
    ("nap", "italo_romance"),   # Neapolitan
    ("scn", "italo_romance"),   # Sicilian
    ("lij", "italo_romance"),   # Ligurian
    ("vec", "italo_romance"),   # Venetian
    ("pms", "italo_romance"),   # Piedmontese
    ("eml", "italo_romance"),   # Emilian-Romagnol
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

DIALECT_CODES = ["nap", "scn", "lij", "vec", "pms", "eml"]
STANDARD_CODES = ["ita", "spa", "fra", "cat", "deu", "slv", "eng"]


# Mapping our codes → ISO 639-3 used by URIEL/lang2vec.
# All of these are ALREADY in URIEL — verified by querying URIEL_LANGUAGES.
LANG2VEC_CODE: Dict[str, str] = {
    "nap": "nap", "scn": "scn", "lij": "lij", "vec": "vec",
    "pms": "pms", "eml": "eml",
    "ita": "ita", "spa": "spa", "fra": "fra", "cat": "cat",
    "deu": "deu", "slv": "slv", "eng": "eng",
}


# Glottocodes (Glottolog 5.x).  Used to join Grambank, PHOIBLE, Lexibank
# tables, which key on Glottocode rather than ISO.
GLOTTOCODE: Dict[str, str] = {
    "nap": "neap1235",
    "scn": "sici1248",
    "lij": "ligu1248",
    "vec": "vene1258",
    "pms": "piem1238",
    "eml": "emil1241",
    "ita": "ital1282",
    "spa": "stan1288",
    "fra": "stan1290",
    "cat": "stan1289",
    "deu": "stan1295",
    "slv": "slov1268",
    "eng": "stan1293",
}


# Hand-coded Glottolog hierarchy paths with DISTINCT leaves.
# Updated to reflect Glottolog 5.x classification of these specific dialects.
GLOTTOLOG_PATH: Dict[str, List[str]] = {
    "ita": ["IE", "Italic", "Romance", "Italo-Western", "Italo-Dalmatian", "Italo-Romance", "Italian"],
    "fra": ["IE", "Italic", "Romance", "Italo-Western", "Western Romance", "Gallo-Iberian", "Gallo-Romance", "Gallo-Rhaetian", "Oil", "French"],
    "spa": ["IE", "Italic", "Romance", "Italo-Western", "Western Romance", "Gallo-Iberian", "Ibero-Romance", "West Iberian", "Castilic", "Spanish"],
    "cat": ["IE", "Italic", "Romance", "Italo-Western", "Western Romance", "Gallo-Iberian", "Gallo-Romance", "Pyrenaic", "Catalan"],

    # Padanian sub-branch (Gallo-Italic) — distinct leaves per language
    "lij": ["IE", "Italic", "Romance", "Italo-Western", "Western Romance", "Gallo-Iberian", "Gallo-Romance", "Gallo-Rhaetian", "Padanian", "Ligurian"],
    "vec": ["IE", "Italic", "Romance", "Italo-Western", "Western Romance", "Gallo-Iberian", "Gallo-Romance", "Gallo-Rhaetian", "Padanian", "Venetian"],
    "pms": ["IE", "Italic", "Romance", "Italo-Western", "Western Romance", "Gallo-Iberian", "Gallo-Romance", "Gallo-Rhaetian", "Padanian", "Piedmontese"],
    "eml": ["IE", "Italic", "Romance", "Italo-Western", "Western Romance", "Gallo-Iberian", "Gallo-Romance", "Gallo-Rhaetian", "Padanian", "Emilian-Romagnol"],
    # Italo-Dalmatian — Sicilian and Neapolitan share a sub-branch
    "scn": ["IE", "Italic", "Romance", "Italo-Western", "Italo-Dalmatian", "Extreme-Southern-Italo-Romance", "Sicilian"],
    "nap": ["IE", "Italic", "Romance", "Italo-Western", "Italo-Dalmatian", "Continental-Southern-Italo-Romance", "Neapolitan"],

    # Non-Romance
    "deu": ["IE", "Germanic", "West Germanic", "High German", "German"],
    "eng": ["IE", "Germanic", "West Germanic", "Anglo-Frisian", "English"],
    "slv": ["IE", "Balto-Slavic", "Slavic", "South Slavic", "Western South Slavic", "Slovenian"],
}

assert set(LANG2VEC_CODE) == set(VARIETY_CODES)
assert set(GLOTTOCODE) == set(VARIETY_CODES)
assert set(GLOTTOLOG_PATH) == set(VARIETY_CODES)


# Gold cluster labels for ARI / NMI evaluation.
GENEALOGICAL_LABELS: Dict[str, str] = {
    "ita": "italo_romance",  "scn": "italo_romance",
    "nap": "italo_romance",  "lij": "italo_romance",
    "vec": "italo_romance",  "pms": "italo_romance",
    "eml": "italo_romance",
    "fra": "other_romance",  "spa": "other_romance",  "cat": "other_romance",
    "deu": "germanic",       "eng": "germanic",
    "slv": "slavic",
}

ROMANCE_FAMILIES = {"italo_romance", "italian", "romance"}
ROMANCE_BINARY_LABELS: Dict[str, str] = {
    c: ("romance" if VARIETY_GROUP[c] in ROMANCE_FAMILIES else "non_romance")
    for c in VARIETY_CODES
}


# Wikipedia file lookup.  Each variety lives in either
# Dataset/wiki/{normalized,not_normalized}/{dialects_in_both_OLDI_and_Flores,
# others_dialects, languages}/<code>.csv
# Used by the data-prep step that builds 80/20 hold-out splits.
DATASET_DIR = Path(__file__).resolve().parents[2] / "Dataset"
WIKI_NORMALIZED = DATASET_DIR / "wiki" / "normalized"
WIKI_NOT_NORMALIZED = DATASET_DIR / "wiki" / "not_normalized"


def wiki_csv_path(code: str, *, normalized: bool) -> Path:
    """Return the absolute path to the Wikipedia CSV for a variety."""
    base = WIKI_NORMALIZED if normalized else WIKI_NOT_NORMALIZED
    if code in DIALECT_CODES:
        # Group A (in OLDI/FLORES): scn, lij, vec
        if code in ("scn", "lij", "vec"):
            return base / "dialects_in_both_OLDI_and_Flores" / f"{code}.csv"
        # Group B (others_dialects): nap, pms, eml
        return base / "others_dialects" / f"{code}.csv"
    # Standards
    return base / "languages" / f"{code}.csv"


VARIETY_NAMES: Dict[str, str] = {
    "nap": "Neapolitan", "scn": "Sicilian", "lij": "Ligurian", "vec": "Venetian",
    "pms": "Piedmontese", "eml": "Emilian-Romagnol",
    "ita": "Italian", "spa": "Spanish", "fra": "French", "cat": "Catalan",
    "deu": "German", "slv": "Slovenian", "eng": "English",
}
