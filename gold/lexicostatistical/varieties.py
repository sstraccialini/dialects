"""
Variety registry for the lexicostatistical gold matrix.

The current 13-variety set is hard-coded here; when the experiment grows
(more Wikipedia-covered varieties), update this file and re-run
``build_ldnd.py`` to regenerate the matrix.

Each code carries a ROLE used by the correlation analysis:

    - "dialect"   : Italo-Romance dialect we are studying
    - "italian"   : standard Italian (the central reference)
    - "external"  : a non-Italian comparison language

The role split drives the secondary correlation metric ``rho2`` in
``evaluation/correlate_against_gold.py``: it is computed only on pairs
(dialect × external), excluding intra-dialect pairs and pairs involving
standard Italian.
"""
from __future__ import annotations

from typing import Dict, List


# Codes ordered as they appear in the wordlist CSV (matches the matrix axes).
VARIETY_CODES: List[str] = [
    "ita",
    "fra", "spa", "cat",
    "deu", "slv", "eng",
    "fur", "lij", "lmo", "sc", "scn", "vec",
]

VARIETY_ROLE: Dict[str, str] = {
    "ita": "italian",
    "fra": "external", "spa": "external", "cat": "external",
    "deu": "external", "slv": "external", "eng": "external",
    "fur": "dialect", "lij": "dialect", "lmo": "dialect",
    "sc":  "dialect", "scn": "dialect", "vec": "dialect",
}

DIALECT_CODES: List[str] = [c for c, r in VARIETY_ROLE.items() if r == "dialect"]
ITALIAN_CODES: List[str] = [c for c, r in VARIETY_ROLE.items() if r == "italian"]
EXTERNAL_CODES: List[str] = [c for c, r in VARIETY_ROLE.items() if r == "external"]

assert set(VARIETY_CODES) == set(VARIETY_ROLE.keys())


# Sub-variety choice per dialect (for paper documentation).
DIALECT_SUB_VARIETIES: Dict[str, str] = {
    "fur": "Central Friulian (Udine koiné)",
    "lij": "Genoese",
    "lmo": "Western Lombard (Milanese-area)",
    "sc":  "Logudorese Sardinian",
    "scn": "Standard Sicilian (Palermo-area)",
    "vec": "Central Venetan (Venice / Padua)",
}
