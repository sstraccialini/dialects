"""
Code mappings and gold-reference helpers that extend
``analysis._shared.varieties`` for the experiments under ``edoardo/``.

Source of truth for the 13 variety codes is ``analysis._shared.varieties``;
here we only add lookups for external resources (lang2vec, Glottolog) and
a default genealogical-cluster label set used for ARI comparisons.
"""
from __future__ import annotations

from typing import Dict, List

from analysis._shared.varieties import (
    VARIETY_CODES,
    VARIETY_GROUP,
    VARIETY_NAMES,
    DIALECT_CODES,
    MODERN_LANGUAGE_CODES,
    ROMANCE_FAMILIES,
)


# --------------------------------------------------------------------------- #
# lang2vec / URIEL ISO 639-3 codes.
# Most of our codes already match URIEL; only Sardinian needs remapping
# (their pipeline uses 639-1 ``sc``; URIEL uses 639-3 ``srd``).
# --------------------------------------------------------------------------- #
LANG2VEC_CODE: Dict[str, str] = {
    "fur": "fur", "lij": "lij", "lmo": "lmo", "sc": "srd",
    "scn": "scn", "vec": "vec",
    "ita": "ita", "spa": "spa", "fra": "fra", "cat": "cat",
    "deu": "deu", "slv": "slv", "eng": "eng",
}


# --------------------------------------------------------------------------- #
# Glottolog hierarchy paths for each variety.
# Used to derive a tree-based distance matrix as a second genealogical proxy
# (URIEL's ``genetic`` distance is also Glottolog-derived but heavily imputed,
# so a hand-coded tree at this granularity is a useful sanity check).
#
# Source: https://glottolog.org/resource/languoid/id/<glottocode>
# These paths follow Glottolog 5.x (May 2026 snapshot).  Where Glottolog
# disagrees with the proposal's grouping, we annotate the choice in-line.
# --------------------------------------------------------------------------- #
GLOTTOLOG_PATH: Dict[str, List[str]] = {
    # Indo-European > Italic > Romance
    "ita": ["IE", "Italic", "Romance", "Italo-Western", "Italo-Dalmatian", "Italo-Romance"],
    "fra": ["IE", "Italic", "Romance", "Italo-Western", "Western Romance", "Gallo-Iberian", "Gallo-Romance", "Gallo-Rhaetian", "Oil"],
    "spa": ["IE", "Italic", "Romance", "Italo-Western", "Western Romance", "Gallo-Iberian", "Ibero-Romance", "West Iberian", "Castilic"],
    "cat": ["IE", "Italic", "Romance", "Italo-Western", "Western Romance", "Gallo-Iberian", "Gallo-Romance", "Pyrenaic"],

    # Italo-Romance dialects covered by FLORES+/OLDI
    "lmo": ["IE", "Italic", "Romance", "Italo-Western", "Western Romance", "Gallo-Iberian", "Gallo-Romance", "Gallo-Rhaetian", "Padanian"],
    "lij": ["IE", "Italic", "Romance", "Italo-Western", "Western Romance", "Gallo-Iberian", "Gallo-Romance", "Gallo-Rhaetian", "Padanian"],
    "vec": ["IE", "Italic", "Romance", "Italo-Western", "Western Romance", "Gallo-Iberian", "Gallo-Romance", "Gallo-Rhaetian", "Padanian"],
    # Friulian: Glottolog places it under Rhaeto-Romance, distinct from Padanian
    "fur": ["IE", "Italic", "Romance", "Italo-Western", "Western Romance", "Gallo-Iberian", "Gallo-Romance", "Gallo-Rhaetian", "Rhaetian"],
    # Sicilian: Italo-Dalmatian like ita but separate sub-branch (Extreme Southern Italo-Romance)
    "scn": ["IE", "Italic", "Romance", "Italo-Western", "Italo-Dalmatian", "Extreme-Southern-Italo-Romance"],
    # Sardinian: separate Romance branch outside Italo-Western
    "sc":  ["IE", "Italic", "Romance", "Sardinian"],

    # Non-Romance
    "deu": ["IE", "Germanic", "West Germanic", "High German"],
    "eng": ["IE", "Germanic", "West Germanic", "Anglo-Frisian", "English"],
    "slv": ["IE", "Balto-Slavic", "Slavic", "South Slavic", "Western South Slavic"],
}

# Sanity check: every code in VARIETY_CODES must appear above.
assert set(GLOTTOLOG_PATH) == set(VARIETY_CODES), \
    f"Glottolog paths missing for {set(VARIETY_CODES) - set(GLOTTOLOG_PATH)}"
assert set(LANG2VEC_CODE) == set(VARIETY_CODES), \
    f"lang2vec codes missing for {set(VARIETY_CODES) - set(LANG2VEC_CODE)}"


# --------------------------------------------------------------------------- #
# Default cluster-agreement labels.
# Five genealogically-motivated clusters used as gold for ARI / V-measure.
# Modify here if you want a coarser/finer cut.
# --------------------------------------------------------------------------- #
GENEALOGICAL_LABELS: Dict[str, str] = {
    # Italo-Romance (Italo-Dalmatian + Padanian + Rhaetian + Sardinian-isolate)
    "ita": "italo_romance", "scn": "italo_romance",
    "lmo": "italo_romance", "lij": "italo_romance", "vec": "italo_romance",
    "fur": "italo_romance", "sc":  "italo_romance",
    # Other Romance (Western Romance, non-Italo-Dalmatian)
    "fra": "other_romance", "spa": "other_romance", "cat": "other_romance",
    # Germanic
    "deu": "germanic",      "eng": "germanic",
    # Slavic
    "slv": "slavic",
}

# Two-way: Romance vs non-Romance (binary, easy sanity check).
ROMANCE_BINARY_LABELS: Dict[str, str] = {
    c: ("romance" if VARIETY_GROUP[c] in ROMANCE_FAMILIES else "non_romance")
    for c in VARIETY_CODES
}


__all__ = [
    "LANG2VEC_CODE",
    "GLOTTOLOG_PATH",
    "GENEALOGICAL_LABELS",
    "ROMANCE_BINARY_LABELS",
    "VARIETY_CODES",
    "VARIETY_GROUP",
    "VARIETY_NAMES",
    "DIALECT_CODES",
    "MODERN_LANGUAGE_CODES",
    "ROMANCE_FAMILIES",
]
