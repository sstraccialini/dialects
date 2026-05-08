"""
Build an expert-curated distance matrix for Italian dialect / language varieties.

This is an INITIAL DRAFT meant to be reviewed and refined by the team.
Distances are normalised to [0, 1]:
    0.00  same variety
    ~0.25 close sister varieties (e.g. Lombard ↔ Ligurian, both Gallo-Italic)
    ~0.40 same Romance branch but different sub-group
    ~0.55 cross-branch Romance (Italo-Romance ↔ Western Romance)
    ~0.65 isolate Romance vs mainstream Italo (Sardinian)
    ~0.85 cross-family but areal contact (Lombard ↔ German)
    ~0.92 cross-family, no contact (Sicilian ↔ Slovenian)

Rationale per cell follows Pellegrini's classification of Italian dialects
(``Carta dei dialetti d'Italia``, 1977) and the handbook
``The Dialects of Italy`` (Maiden & Parry, eds., 1997).  The matrix tries
to encode three signals simultaneously:

    1. Genealogy / sub-branch (dominant signal).
    2. Areal proximity (secondary, lower weight).
    3. Documented contact (tertiary, only well-attested cases:
       Lombard/Ligurian ↔ French/Occitan, Friulian ↔ German/Slavic,
       Sicilian ↔ Greek/Arabic substrate, Sardinian as isolate).

This file is meant to be edited.  The function ``EXPERT_DISTANCES`` is
defined explicitly cell by cell so that disagreements are easy to find
and revise.

Output: ``matrices/expert_dialectology.npz`` with keys matrix, labels, meta.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from edoardo.varieties_extra import VARIETY_CODES


# --------------------------------------------------------------------------- #
# Initial expert estimates.  Only the upper triangle (a < b alphabetically)
# is filled in; the matrix is symmetrised at build time.
# Edit values here to refine.
# --------------------------------------------------------------------------- #
EXPERT_PAIRWISE: Dict[Tuple[str, str], float] = {
    # --- Italo-Romance internal (6 dialects + ita) --------------------------
    # Gallo-Italic: lij, lmo, vec close.  fur is Rhaeto.  scn is Extreme South.
    ("lij", "lmo"): 0.25,   # both Gallo-Italic
    ("lij", "vec"): 0.30,
    ("lmo", "vec"): 0.30,
    ("fur", "lij"): 0.45,   # Rhaeto-Romance vs Gallo-Italic
    ("fur", "lmo"): 0.45,
    ("fur", "vec"): 0.40,   # geographically closer
    ("lij", "scn"): 0.55,   # cross-Italian (North vs South extremes)
    ("lmo", "scn"): 0.55,
    ("scn", "vec"): 0.55,
    ("fur", "scn"): 0.55,
    # Sardinian (sc) is the most distant Italo-Romance (separate Romance branch)
    ("lij", "sc"):  0.65,
    ("lmo", "sc"):  0.65,
    ("sc",  "vec"): 0.65,
    ("fur", "sc"):  0.65,
    ("sc",  "scn"): 0.60,   # both peripheral, somewhat closer
    # vs standard Italian
    ("ita", "lij"): 0.45,
    ("ita", "lmo"): 0.45,
    ("ita", "vec"): 0.40,
    ("fur", "ita"): 0.50,
    ("ita", "scn"): 0.45,
    ("ita", "sc"):  0.55,

    # --- Other Romance (fra, spa, cat) vs Italo-Romance ---------------------
    # Standard Italian to other Romance
    ("fra", "ita"): 0.45,
    ("ita", "spa"): 0.40,
    ("cat", "ita"): 0.45,
    # Gallo-Italic dialects to French/Catalan (gallo-romance areal closeness)
    ("fra", "lmo"): 0.50,
    ("fra", "lij"): 0.50,
    ("fra", "vec"): 0.55,
    ("cat", "lmo"): 0.55,
    ("cat", "lij"): 0.55,
    ("cat", "vec"): 0.55,
    ("lmo", "spa"): 0.60,
    ("lij", "spa"): 0.60,
    ("spa", "vec"): 0.60,
    # Friulian to other Romance (Rhaeto, less Gallo)
    ("fra", "fur"): 0.55,
    ("cat", "fur"): 0.55,
    ("fur", "spa"): 0.60,
    # Sardinian as isolate Romance
    ("fra", "sc"): 0.65,
    ("cat", "sc"): 0.65,
    ("sc",  "spa"): 0.60,    # some Iberian/Catalan substrate noted in lit
    # Sicilian: Italo-Dalmatian
    ("fra", "scn"): 0.55,
    ("cat", "scn"): 0.55,
    ("scn", "spa"): 0.55,
    # Among Western Romance themselves
    ("fra", "spa"): 0.45,
    ("cat", "fra"): 0.40,
    ("cat", "spa"): 0.35,    # Iberian closeness

    # --- Romance vs Germanic (deu, eng) -------------------------------------
    # Areal contact: Friulian ↔ German (Alpine), Lombard/Veneto ↔ German lighter
    ("deu", "fur"): 0.78,    # documented contact
    ("deu", "lmo"): 0.85,
    ("deu", "lij"): 0.88,
    ("deu", "vec"): 0.83,    # some Alpine areal
    ("deu", "ita"): 0.85,
    ("deu", "scn"): 0.92,
    ("deu", "sc"):  0.92,
    ("cat", "deu"): 0.90,
    ("deu", "fra"): 0.85,
    ("deu", "spa"): 0.90,
    # English (Germanic but very lexically distant from continental)
    ("eng", "fur"): 0.92,
    ("eng", "lmo"): 0.92,
    ("eng", "lij"): 0.92,
    ("eng", "vec"): 0.92,
    ("eng", "ita"): 0.90,
    ("eng", "scn"): 0.95,
    ("eng", "sc"):  0.95,
    ("cat", "eng"): 0.90,
    ("eng", "fra"): 0.88,
    ("eng", "spa"): 0.92,
    # Germanic internal
    ("deu", "eng"): 0.65,

    # --- Romance vs Slavic (slv) --------------------------------------------
    # Friulian ↔ Slovene: documented border contact
    ("fur", "slv"): 0.82,
    ("slv", "vec"): 0.92,    # weaker; some loanwords
    ("lmo", "slv"): 0.95,
    ("lij", "slv"): 0.95,
    ("scn", "slv"): 0.95,
    ("sc",  "slv"): 0.95,
    ("ita", "slv"): 0.92,
    ("cat", "slv"): 0.95,
    ("fra", "slv"): 0.95,
    ("slv", "spa"): 0.95,
    # vs Germanic
    ("deu", "slv"): 0.82,    # areal contact (Alpine)
    ("eng", "slv"): 0.92,
}


def _normalise(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a < b else (b, a)


def build_expert_matrix(codes: List[str]) -> Tuple[np.ndarray, List[str]]:
    n = len(codes)
    d = np.zeros((n, n), dtype=np.float64)
    missing: List[Tuple[str, str]] = []

    for i, a in enumerate(codes):
        for j in range(i + 1, n):
            b = codes[j]
            key = _normalise(a, b)
            if key in EXPERT_PAIRWISE:
                d[i, j] = d[j, i] = float(EXPERT_PAIRWISE[key])
            else:
                missing.append(key)

    if missing:
        # Hard fail: we want every pair to be explicit, no silent imputation.
        raise RuntimeError(
            f"Missing expert distances for {len(missing)} pairs:\n  "
            + "\n  ".join(f"{a}-{b}" for a, b in sorted(missing))
        )
    return d, codes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).parent / "matrices")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    codes = list(VARIETY_CODES)
    mat, _ = build_expert_matrix(codes)

    meta = {
        "source": "Hand-curated by team — initial draft",
        "based_on": [
            "Pellegrini (1977), Carta dei dialetti d'Italia",
            "Maiden & Parry (eds., 1997), The Dialects of Italy",
        ],
        "encoded_signals": ["genealogy", "areal_proximity", "documented_contact"],
        "n_pairs": len(EXPERT_PAIRWISE),
        "codes": codes,
        "warnings": [
            "Subjective ordinal estimates; revise via team review.",
            "Contact entries (deu↔fur, slv↔fur, etc.) are weighted lower than ",
            "what the genealogy alone would imply.",
        ],
    }
    out_path = args.out_dir / "expert_dialectology.npz"
    np.savez(
        out_path,
        matrix=mat,
        labels=np.array(codes, dtype=object),
        meta=np.array([json.dumps(meta)], dtype=object),
    )
    n_offdiag = mat.size - len(codes)
    print(f"  expert_dialectology  range=[{mat.min():.3f}, {mat.max():.3f}]  "
          f"mean_offdiag={mat.sum() / n_offdiag:.3f}  → {out_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
