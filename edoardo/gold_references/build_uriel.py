"""
Build URIEL / lang2vec reference distance matrices for the 13 varieties.

URIEL aggregates typological data (WALS, PHOIBLE, Ethnologue, Glottolog).
``lang2vec`` 1.1.x exposes per-language feature vectors via
``get_features(langs, set_name)``; we compute cosine distance between
feature vectors per Littell et al. (2017): ``d = 1 - cos(θ)``.

Distance types we build:

    genetic       — ``fam``         (Glottolog family one-hot)
    syntactic     — ``syntax_knn``  (WALS syntax features, k-NN imputed)
    phonological  — ``phonology_knn``
    inventory     — ``inventory_knn``
    geographic    — ``geo``         (proximity to fixed reference points)
    featural      — concat of syntax + phonology + inventory

Caveats (must be cited in any paper that uses these matrices):
  - For low-resource Italian dialects (lmo, lij, vec, fur, scn) several
    URIEL features are MISSING; lang2vec imputes them via k-NN over the
    Glottolog tree, which can artificially push them toward Italian.
  - `genetic` is fully tree-derived → does NOT capture contact effects.
  - `geo` uses similarity to a fixed set of reference points, not raw
    lat/lon — fine for a coarse areal proxy, not for fine geography.

Output: one ``.npz`` per distance type under ``matrices/uriel_<type>.npz``,
with keys:
    matrix     (N, N) float64  symmetric distance matrix, zero diagonal
    labels     (N,)   str      variety codes in canonical order
    meta       (1,)   str      JSON describing how it was built

Run from the repo root:
    python -m edoardo.gold_references.build_uriel
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from edoardo.varieties_extra import LANG2VEC_CODE, VARIETY_CODES


# (output name, lang2vec feature set or list of sets to concatenate)
URIEL_RECIPES: Dict[str, Tuple[str, ...]] = {
    "genetic":      ("fam",),
    "syntactic":    ("syntax_knn",),
    "phonological": ("phonology_knn",),
    "inventory":    ("inventory_knn",),
    "geographic":   ("geo",),
    "featural":     ("syntax_knn", "phonology_knn", "inventory_knn"),
}


def _cosine_distance_matrix(X: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance, symmetric, zero diagonal, clipped to [0, 2]."""
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    Xn = X / norm
    sim = Xn @ Xn.T
    # numerical clipping
    sim = np.clip(sim, -1.0, 1.0)
    sim = (sim + sim.T) / 2.0
    d = 1.0 - sim
    np.fill_diagonal(d, 0.0)
    return d


def _features_for(iso_codes: List[str], feature_sets: Tuple[str, ...]) -> np.ndarray:
    """Concatenate feature vectors across the given sets for each language."""
    import lang2vec.lang2vec as l2v

    parts: List[np.ndarray] = []
    for fs in feature_sets:
        feats = l2v.get_features(iso_codes, fs)
        # `feats` is a dict {iso: list[float]} preserving insertion order.
        arr = np.array([feats[i] for i in iso_codes], dtype=np.float64)
        # URIEL marks "unknown / not applicable" with -1.  Treat as 0 so it
        # doesn't perturb cosine distance.  This matches lang2vec's own
        # legacy `distance()` behaviour.
        arr = np.where(arr < 0, 0.0, arr)
        parts.append(arr)
    return np.concatenate(parts, axis=1)


def _build_one(distance_type: str, codes: List[str]) -> np.ndarray:
    feature_sets = URIEL_RECIPES[distance_type]
    iso = [LANG2VEC_CODE[c] for c in codes]
    X = _features_for(iso, feature_sets)
    return _cosine_distance_matrix(X)


def _save(distance_type: str, mat: np.ndarray, codes: List[str],
          feature_sets: Tuple[str, ...], out_dir: Path) -> Path:
    meta = {
        "source": "URIEL via lang2vec 1.1.x",
        "distance_type": distance_type,
        "feature_sets": list(feature_sets),
        "metric": "cosine distance on feature vectors (Littell et al., 2017)",
        "codes": codes,
        "lang2vec_codes": [LANG2VEC_CODE[c] for c in codes],
        "url": "https://www.cs.cmu.edu/~dmortens/uriel.html",
        "warnings": [
            "Low-resource dialect features in URIEL are k-NN imputed from "
            "neighbouring Glottolog languoids — can pull dialects toward Italian.",
            "`genetic` is fully tree-derived; does not capture contact effects.",
        ],
    }
    out_path = out_dir / f"uriel_{distance_type}.npz"
    np.savez(
        out_path,
        matrix=mat,
        labels=np.array(codes, dtype=object),
        meta=np.array([json.dumps(meta)], dtype=object),
    )
    return out_path


def build_all(out_dir: Path, codes: List[str] | None = None,
              distance_types: List[str] | None = None) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    codes = list(codes or VARIETY_CODES)
    types = list(distance_types or URIEL_RECIPES.keys())
    written = []
    for dt in types:
        try:
            mat = _build_one(dt, codes)
        except Exception as exc:
            warnings.warn(f"URIEL {dt}: failed ({exc}) — skipping", stacklevel=2)
            continue
        p = _save(dt, mat, codes, URIEL_RECIPES[dt], out_dir)
        written.append(p)
        n_offdiag = mat.size - len(codes)
        print(f"  uriel_{dt:<13s}  range=[{mat.min():.3f}, {mat.max():.3f}]  "
              f"mean_offdiag={mat.sum() / n_offdiag:.3f}  → {p.name}")
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).parent / "matrices",
                    help="Where to write the .npz files.")
    ap.add_argument("--types", nargs="+", choices=list(URIEL_RECIPES),
                    default=None,
                    help=f"Subset of distance types (default: all {list(URIEL_RECIPES)}).")
    args = ap.parse_args(argv)

    print(f"Building URIEL gold matrices for {len(VARIETY_CODES)} varieties:")
    print(f"  {VARIETY_CODES}")
    print(f"  → {args.out_dir}")
    written = build_all(args.out_dir, distance_types=args.types)
    print(f"\nWrote {len(written)} URIEL distance matrices.")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
