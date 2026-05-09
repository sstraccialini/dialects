"""
Hand-curated expert distance matrix for Experiment 1 dialects:
nap, scn, lij, vec, pms, eml + ita, spa, fra, cat, deu, slv, eng.

Calibration scale (same as the parent edoardo/gold_references/build_expert.py):
    0.20-0.30   sister varieties in the same Italo-Romance sub-branch
    0.40-0.45   cross sub-branch within Italo-Romance
    0.45-0.55   ita ↔ Italo-Romance dialect; cross-Romance branches
    0.40-0.50   Romance internal (cat/spa/fra ↔ ita)
    0.55-0.65   peripheral / cross-major-branch Romance
    0.78-0.85   cross-family with attested contact (deu↔alpine, slv↔fur)
    0.85-0.92   cross-family no contact

Sub-branch grouping (Pellegrini 1977, Maiden & Parry 1997):
    Padanian Gallo-Italic     : lij, pms, eml, vec
    Italo-Dalmatian (Extreme) : scn, nap (sister continental Southern)
    Italo-Romance (Tuscan)    : ita

Run:
    python -m edoardo.exp1_uriel_native.build_expert_v2 \\
        --out-dir edoardo/exp1_uriel_native/gold_references/matrices
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from edoardo.exp1_uriel_native.varieties import VARIETY_CODES


# Hand-curated 78-pair distance matrix.  Edit values here to refine.
# Keys must satisfy a < b alphabetically (the builder normalises lookup).
EXPERT_PAIRWISE: Dict[Tuple[str, str], float] = {
    # --- Italo-Romance internal ----------------------------------------
    # Padanian Gallo-Italic siblings (close)
    ("lij", "pms"): 0.25,
    ("eml", "lij"): 0.30,
    ("lij", "vec"): 0.30,
    ("eml", "pms"): 0.30,
    ("pms", "vec"): 0.30,
    ("eml", "vec"): 0.25,
    # Extreme Southern Italo-Romance siblings
    ("nap", "scn"): 0.30,
    # Italo-Romance to Padanian (cross sub-branch within Italo-Romance)
    ("ita", "lij"): 0.45,
    ("ita", "pms"): 0.45,
    ("eml", "ita"): 0.40,
    ("ita", "vec"): 0.40,
    # Italo-Romance to Extreme Southern (Tuscan ↔ neapolitan/sicilian)
    ("ita", "nap"): 0.40,
    ("ita", "scn"): 0.45,
    # Padanian ↔ Extreme Southern (cross italo-romance, distant)
    ("lij", "nap"): 0.50,
    ("lij", "scn"): 0.55,
    ("nap", "pms"): 0.50,
    ("pms", "scn"): 0.55,
    ("nap", "vec"): 0.50,
    ("scn", "vec"): 0.55,
    ("eml", "nap"): 0.50,
    ("eml", "scn"): 0.55,

    # --- Other Romance (fra, spa, cat) internal ------------------------
    ("fra", "spa"): 0.45,
    ("cat", "fra"): 0.40,
    ("cat", "spa"): 0.35,

    # --- Other Romance ↔ Italo-Romance + Italian -----------------------
    # Standard Italian first
    ("fra", "ita"): 0.45,
    ("ita", "spa"): 0.40,
    ("cat", "ita"): 0.45,
    # French ↔ Padanian (Gallo-Romance areal proximity)
    ("fra", "lij"): 0.50,
    ("fra", "pms"): 0.50,
    ("eml", "fra"): 0.50,
    ("fra", "vec"): 0.55,
    # French ↔ Extreme Southern
    ("fra", "nap"): 0.55,
    ("fra", "scn"): 0.55,
    # Spanish ↔ Padanian
    ("lij", "spa"): 0.60,
    ("pms", "spa"): 0.60,
    ("eml", "spa"): 0.55,
    ("spa", "vec"): 0.60,
    # Spanish ↔ Extreme Southern (some southern italian → spanish historic ties)
    ("nap", "spa"): 0.55,
    ("scn", "spa"): 0.55,
    # Catalan ↔ Padanian
    ("cat", "lij"): 0.55,
    ("cat", "pms"): 0.55,
    ("cat", "eml"): 0.55,
    ("cat", "vec"): 0.55,
    # Catalan ↔ Extreme Southern
    ("cat", "nap"): 0.55,
    ("cat", "scn"): 0.55,

    # --- German (deu) ↔ everything Romance ----------------------------
    # Areal contact: alpine zones (vec/eml/pms milder, lij less so)
    ("deu", "lij"): 0.85,
    ("deu", "pms"): 0.83,
    ("deu", "vec"): 0.83,
    ("deu", "eml"): 0.85,
    ("deu", "ita"): 0.85,
    ("deu", "nap"): 0.92,
    ("deu", "scn"): 0.92,
    # German vs Other Romance
    ("deu", "fra"): 0.85,
    ("deu", "spa"): 0.90,
    ("cat", "deu"): 0.90,

    # --- English (eng) ↔ everything Romance ---------------------------
    ("eng", "lij"): 0.92,
    ("eng", "pms"): 0.92,
    ("eml", "eng"): 0.92,
    ("eng", "vec"): 0.92,
    ("eng", "ita"): 0.90,
    ("eng", "nap"): 0.95,
    ("eng", "scn"): 0.95,
    ("eng", "fra"): 0.88,
    ("eng", "spa"): 0.92,
    ("cat", "eng"): 0.90,

    # --- Slovenian (slv) ↔ Romance ------------------------------------
    # No specific Slavic-Romance contact in this set (slv-fur was the contact
    # case, and fur is NOT in exp1).  All values reflect cross-family distance.
    ("lij", "slv"): 0.95,
    ("pms", "slv"): 0.95,
    ("eml", "slv"): 0.95,
    ("slv", "vec"): 0.92,   # mild eastern areal residue
    ("ita", "slv"): 0.92,
    ("nap", "slv"): 0.95,
    ("scn", "slv"): 0.95,
    ("fra", "slv"): 0.95,
    ("slv", "spa"): 0.95,
    ("cat", "slv"): 0.95,

    # --- Non-Romance internal -----------------------------------------
    ("deu", "eng"): 0.65,
    ("deu", "slv"): 0.82,   # alpine areal
    ("eng", "slv"): 0.92,
}


def _normalise(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a < b else (b, a)


def build_matrix(codes: List[str]) -> np.ndarray:
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
        raise RuntimeError(
            f"Missing expert distances for {len(missing)} pairs:\n  "
            + "\n  ".join(f"{a}-{b}" for a, b in sorted(missing)))
    return d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    codes = list(VARIETY_CODES)
    mat = build_matrix(codes)
    meta = {
        "source": "Hand-curated by team (exp1 variety set)",
        "based_on": [
            "Pellegrini (1977), Carta dei dialetti d'Italia",
            "Maiden & Parry (eds., 1997), The Dialects of Italy",
        ],
        "encoded_signals": ["genealogy", "areal_proximity", "documented_contact"],
        "n_pairs": len(EXPERT_PAIRWISE),
        "codes": codes,
        "warnings": [
            "Subjective ordinal estimates; revise via team review.",
            "Calibration scale: 0.20 sisters → 0.92 cross-family no contact.",
        ],
    }
    out_path = args.out_dir / "expert_dialectology.npz"
    np.savez(out_path,
             matrix=mat,
             labels=np.array(codes, dtype=object),
             meta=np.array([json.dumps(meta)], dtype=object))
    n_off = mat.size - len(codes)
    print(f"  expert_dialectology  range=[{mat.min():.3f}, {mat.max():.3f}]  "
          f"mean_offdiag={mat.sum() / n_off:.3f}  → {out_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
