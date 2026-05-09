"""
Build the great-circle (Haversine) geographic gold distance matrix.

Distances are computed in km between the (lat, lon) centroid of each
variety in ``varieties.LATLON``, then divided by the maximum off-diagonal
distance so the matrix is in [0, 1].

Output: ``matrices/geographic_haversine.npz`` with keys
    matrix    (N, N) float — normalised distance
    labels    (N,)   variety codes
    meta      (1,)   JSON describing source + max_km

CLI:
    python -m gold.geographic.build_haversine \\
        --out-dir gold/geographic/matrices
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

from gold.geographic.varieties import LATLON


def _haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    R = 6371.0  # Earth radius (km)
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    codes: List[str] = list(LATLON.keys())
    n = len(codes)
    raw_km = np.zeros((n, n), dtype=np.float64)
    for i, ci in enumerate(codes):
        for j in range(i + 1, n):
            cj = codes[j]
            raw_km[i, j] = raw_km[j, i] = _haversine_km(LATLON[ci], LATLON[cj])

    finite = raw_km[raw_km > 0]
    max_km = float(finite.max()) if finite.size else 1.0
    matrix = raw_km / max_km

    meta = {
        "source": "Glottolog 5.x languoid centroids (hand-coded snapshot)",
        "metric": "Haversine great-circle distance, normalised by max off-diagonal",
        "max_km": max_km,
        "codes": codes,
        "coordinates_deg": {c: list(LATLON[c]) for c in codes},
        "warnings": [
            "Centroids are approximate; varieties with broad geographic "
            "spread (e.g. Lombard, Sardinian) collapse to a single point.",
        ],
    }
    out_path = args.out_dir / "geographic_haversine.npz"
    np.savez(out_path,
             matrix=matrix,
             labels=np.array(codes, dtype=object),
             meta=np.array([json.dumps(meta)], dtype=object))
    n_off = matrix.size - len(codes)
    print(f"  geographic_haversine  range=[{matrix.min():.3f}, {matrix.max():.3f}]  "
          f"max_km={max_km:.0f}  mean_offdiag={matrix.sum() / n_off:.3f}  → {out_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
