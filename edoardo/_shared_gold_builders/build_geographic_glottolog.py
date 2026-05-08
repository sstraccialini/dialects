"""
Build a great-circle (Haversine) geographic distance matrix from
Glottolog 5.x latitude / longitude.

Independent of URIEL ``geographic`` (which is a 299-D similarity vector
to fixed reference points).  Here we use the actual lat/lon of each
languoid and compute spherical distance in km, then normalise to [0, 1]
by dividing by the maximum off-diagonal distance in the matrix.

We ship inline coordinates for our 13 varieties (sourced from Glottolog
5.x as of May 2026) so the builder works without any download.  If
Glottolog updates, override ``COORDS_OVERRIDE`` in the varieties module.

CLI:
    python -m edoardo._shared_gold_builders.build_geographic_glottolog \\
        --varieties-module edoardo.exp1_uriel_native.varieties \\
        --out-dir edoardo/exp1_uriel_native/gold_references/matrices
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# Glottolog 5.x coordinates (May 2026 snapshot) for languoids relevant
# to our two experiments.  (latitude, longitude) in decimal degrees.
GLOTTOLOG_COORDS: Dict[str, Tuple[float, float]] = {
    # Italian dialects (centroid of speech area per Glottolog languoid record)
    "neap1235": (40.85, 14.25),       # Neapolitan
    "sici1248": (37.50, 14.00),       # Sicilian
    "ligu1248": (44.40, 8.95),        # Ligurian
    "vene1258": (45.45, 12.33),       # Venetian
    "piem1238": (45.07, 7.69),        # Piedmontese
    "emil1241": (44.50, 11.34),       # Emilian-Romagnol (umbrella)
    "friu1240": (46.07, 13.23),       # Friulian
    "lomb1257": (45.45, 9.18),        # Lombard
    # Sardinian — multiple Glottocodes; use Logudorese centroid
    "logu1236": (40.30, 9.15),        # Logudorese (sc default)
    "camp1261": (39.20, 9.10),        # Campidanese (alternative sc)
    "sard1257": (40.10, 9.10),        # Umbrella Sardinian
    # Standards
    "ital1282": (41.90, 12.49),       # Italian — Rome
    "stan1288": (40.42, -3.70),       # Spanish (Castilian) — Madrid
    "stan1290": (48.85,  2.35),       # French — Paris
    "stan1289": (41.39,  2.17),       # Catalan — Barcelona
    "stan1295": (52.52, 13.40),       # Standard German — Berlin
    "slov1268": (46.06, 14.51),       # Slovenian — Ljubljana
    "stan1293": (51.51, -0.13),       # Standard English — London
}


def _haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    R = 6371.0  # Earth radius in km
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def build(varieties_module: str, out_dir: Path) -> Path:
    mod = importlib.import_module(varieties_module)
    codes: List[str] = list(mod.VARIETY_CODES)
    gmap: Dict[str, str] = mod.GLOTTOCODE

    coords = {c: GLOTTOLOG_COORDS[gmap[c]] for c in codes
              if gmap[c] in GLOTTOLOG_COORDS}
    missing = [c for c in codes if c not in coords]

    n = len(codes)
    raw = np.zeros((n, n), dtype=np.float64)
    for i, ci in enumerate(codes):
        for j in range(i + 1, n):
            cj = codes[j]
            if ci not in coords or cj not in coords:
                raw[i, j] = raw[j, i] = np.nan
                continue
            raw[i, j] = raw[j, i] = _haversine_km(coords[ci], coords[cj])

    # Normalise by max off-diagonal so range ⊆ [0, 1].
    finite = raw[np.isfinite(raw) & (raw > 0)]
    denom = float(finite.max()) if finite.size else 1.0
    dist = raw / denom if denom > 0 else raw

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "geographic_glottolog.npz"
    meta = {
        "source":   "Glottolog 5.x languoid coordinates (May 2026 snapshot)",
        "metric":   "Haversine great-circle distance, normalised by max off-diagonal",
        "max_km":   denom,
        "varieties_module": varieties_module,
        "codes":    codes,
        "glottocodes": [gmap[c] for c in codes],
        "coords":   {c: list(coords[c]) for c in coords},
        "missing":  missing,
        "warnings": [
            "Coordinates are languoid centroids; for dialects with broad "
            "geographic spread (e.g. Lombard) this is approximate.",
        ],
    }
    np.savez(out_path,
             matrix=dist,
             labels=np.array(codes, dtype=object),
             meta=np.array([json.dumps(meta)], dtype=object))
    print(f"  geographic_glottolog  range=[{np.nanmin(dist):.3f}, "
          f"{np.nanmax(dist):.3f}]  missing={missing or '∅'}  → {out_path.name}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--varieties-module", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    build(args.varieties_module, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
