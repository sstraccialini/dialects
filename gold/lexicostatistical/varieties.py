"""
Variety registry for the lexicostatistical gold matrix.

When the experiment grows (more Wikipedia-covered varieties), update
this file and re-run ``build_ldnd.py`` (or the rebuild SLURM job) to
regenerate the matrix.

Each code carries a ROLE used by the correlation analysis:

    - "dialect"   : Italo-Romance dialect we are studying
    - "italian"   : standard Italian (the central reference)
    - "external"  : a non-Italian comparison language (any family)

The role split drives the secondary correlation metric ``rho2`` in
``evaluation/correlate_against_gold.py``: it is computed only on pairs
(dialect × external), excluding intra-dialect pairs and pairs involving
standard Italian.
"""
from __future__ import annotations

from typing import Dict, List


# Codes ordered as they appear in the wordlist CSV (matches the matrix axes).
# Mirrors analysis._shared.varieties.VARIETIES (commit 8712c65).
VARIETY_CODES: List[str] = [
    # 6 Italo-Romance dialects under study
    "fur", "lij", "lmo", "sc", "scn", "vec",
    # Standard Italian (central reference)
    "ita",
    # Other Romance
    "fra", "spa", "cat", "por", "oci",
    # Germanic
    "deu", "eng",
    # Slavic
    "slv", "hrv",
    # Non-Indo-European (control)
    "hun",
]

VARIETY_ROLE: Dict[str, str] = {
    # Dialects
    "fur": "dialect", "lij": "dialect", "lmo": "dialect",
    "sc":  "dialect", "scn": "dialect", "vec": "dialect",
    # Italian
    "ita": "italian",
    # External (Romance + non-Romance)
    "fra": "external", "spa": "external", "cat": "external",
    "por": "external", "oci": "external",
    "deu": "external", "eng": "external",
    "slv": "external", "hrv": "external",
    "hun": "external",
}

DIALECT_CODES:  List[str] = [c for c, r in VARIETY_ROLE.items() if r == "dialect"]
ITALIAN_CODES:  List[str] = [c for c, r in VARIETY_ROLE.items() if r == "italian"]
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
