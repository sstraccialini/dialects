"""
Varieties for Experiment 2: original 6 dialects evaluated with native
(not_normalized) Wikipedia text and a richer gold set.

This is the OPPOSITE-NORMALIZATION counterpart to the existing
``edoardo/`` analysis (which used the normalized Wikipedia variant).
Pretrained encoders (XLM-R, CANINE, Sentence-MiniLM) actually want
native cased+accented text — feeding them lowercase-ASCII destroys
their subword/character vocabulary.  So this experiment is also a
fair re-evaluation of those methods.

Coverage caveat: of our 6 dialects (fur, lij, lmo, sc, scn, vec), only
3 have native URIEL entries (lij, scn, vec).  We use multiple golds
and report coverage per gold rather than relying on any single one.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List


# 13 varieties: original 6 dialects + 7 standards.  Same as edoardo/varieties_extra.
VARIETIES: List[tuple[str, str]] = [
    ("fur", "italo_romance"),
    ("lij", "italo_romance"),
    ("lmo", "italo_romance"),
    ("sc",  "italo_romance"),
    ("scn", "italo_romance"),
    ("vec", "italo_romance"),
    ("ita", "italian"),
    ("spa", "romance"),
    ("fra", "romance"),
    ("cat", "romance"),
    ("deu", "germanic"),
    ("slv", "slavic"),
    ("eng", "english"),
]

VARIETY_CODES = [code for code, _ in VARIETIES]
VARIETY_GROUP = dict(VARIETIES)
DIALECT_CODES = ["fur", "lij", "lmo", "sc", "scn", "vec"]
STANDARD_CODES = ["ita", "spa", "fra", "cat", "deu", "slv", "eng"]


# ISO mapping — same as the parent edoardo/varieties_extra.py
LANG2VEC_CODE: Dict[str, str] = {
    "fur": "fur", "lij": "lij", "lmo": "lmo", "sc": "srd",
    "scn": "scn", "vec": "vec",
    "ita": "ita", "spa": "spa", "fra": "fra", "cat": "cat",
    "deu": "deu", "slv": "slv", "eng": "eng",
}


GLOTTOCODE: Dict[str, str] = {
    "fur": "friu1240",
    "lij": "ligu1248",
    "lmo": "lomb1257",
    "sc":  "logu1236",   # Logudorese; could also use camp1261 or sard1257 umbrella
    "scn": "sici1248",
    "vec": "vene1258",
    "ita": "ital1282",
    "spa": "stan1288",
    "fra": "stan1290",
    "cat": "stan1289",
    "deu": "stan1295",
    "slv": "slov1268",
    "eng": "stan1293",
}


# Glottolog hierarchy paths — identical to the fixed version in
# edoardo/varieties_extra.py.  Repeated here so the experiment is
# self-contained.
GLOTTOLOG_PATH: Dict[str, List[str]] = {
    "ita": ["IE", "Italic", "Romance", "Italo-Western", "Italo-Dalmatian", "Italo-Romance", "Italian"],
    "fra": ["IE", "Italic", "Romance", "Italo-Western", "Western Romance", "Gallo-Iberian", "Gallo-Romance", "Gallo-Rhaetian", "Oil", "French"],
    "spa": ["IE", "Italic", "Romance", "Italo-Western", "Western Romance", "Gallo-Iberian", "Ibero-Romance", "West Iberian", "Castilic", "Spanish"],
    "cat": ["IE", "Italic", "Romance", "Italo-Western", "Western Romance", "Gallo-Iberian", "Gallo-Romance", "Pyrenaic", "Catalan"],
    "lmo": ["IE", "Italic", "Romance", "Italo-Western", "Western Romance", "Gallo-Iberian", "Gallo-Romance", "Gallo-Rhaetian", "Padanian", "Lombard"],
    "lij": ["IE", "Italic", "Romance", "Italo-Western", "Western Romance", "Gallo-Iberian", "Gallo-Romance", "Gallo-Rhaetian", "Padanian", "Ligurian"],
    "vec": ["IE", "Italic", "Romance", "Italo-Western", "Western Romance", "Gallo-Iberian", "Gallo-Romance", "Gallo-Rhaetian", "Padanian", "Venetian"],
    "fur": ["IE", "Italic", "Romance", "Italo-Western", "Western Romance", "Gallo-Iberian", "Gallo-Romance", "Gallo-Rhaetian", "Rhaetian", "Friulian"],
    "scn": ["IE", "Italic", "Romance", "Italo-Western", "Italo-Dalmatian", "Extreme-Southern-Italo-Romance", "Sicilian"],
    "sc":  ["IE", "Italic", "Romance", "Sardinian", "Logudorese-Campidanese", "Sardinian-Common"],
    "deu": ["IE", "Germanic", "West Germanic", "High German", "German"],
    "eng": ["IE", "Germanic", "West Germanic", "Anglo-Frisian", "English"],
    "slv": ["IE", "Balto-Slavic", "Slavic", "South Slavic", "Western South Slavic", "Slovenian"],
}

assert set(LANG2VEC_CODE) == set(VARIETY_CODES)
assert set(GLOTTOCODE) == set(VARIETY_CODES)
assert set(GLOTTOLOG_PATH) == set(VARIETY_CODES)


GENEALOGICAL_LABELS: Dict[str, str] = {
    "ita": "italo_romance", "scn": "italo_romance",
    "lmo": "italo_romance", "lij": "italo_romance", "vec": "italo_romance",
    "fur": "italo_romance", "sc":  "italo_romance",
    "fra": "other_romance", "spa": "other_romance", "cat": "other_romance",
    "deu": "germanic",      "eng": "germanic",
    "slv": "slavic",
}

ROMANCE_FAMILIES = {"italo_romance", "italian", "romance"}
ROMANCE_BINARY_LABELS: Dict[str, str] = {
    c: ("romance" if VARIETY_GROUP[c] in ROMANCE_FAMILIES else "non_romance")
    for c in VARIETY_CODES
}


VARIETY_NAMES: Dict[str, str] = {
    "fur": "Friulian", "lij": "Ligurian", "lmo": "Lombard",
    "sc":  "Sardinian", "scn": "Sicilian", "vec": "Venetian",
    "ita": "Italian", "spa": "Spanish", "fra": "French", "cat": "Catalan",
    "deu": "German", "slv": "Slovenian", "eng": "English",
}
