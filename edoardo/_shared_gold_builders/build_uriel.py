"""
Build URIEL/lang2vec gold distance matrices for an arbitrary varieties module.

This is the parameterised version of ``edoardo/gold_references/build_uriel.py``.
Reads ``LANG2VEC_CODE`` from the varieties module to map our codes to
URIEL ISO 639-3 entries.

CLI:
    python -m edoardo._shared_gold_builders.build_uriel \\
        --varieties-module edoardo.exp1_uriel_native.varieties \\
        --out-dir edoardo/exp1_uriel_native/gold_references/matrices
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


URIEL_RECIPES: Dict[str, Tuple[str, ...]] = {
    "genetic":      ("fam",),
    "syntactic":    ("syntax_knn",),
    "phonological": ("phonology_knn",),
    "inventory":    ("inventory_knn",),
    "geographic":   ("geo",),
    "featural":     ("syntax_knn", "phonology_knn", "inventory_knn"),
}


def _cosine_distance_matrix(X: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    Xn = X / norm
    sim = Xn @ Xn.T
    sim = np.clip(sim, -1.0, 1.0)
    sim = (sim + sim.T) / 2.0
    d = 1.0 - sim
    np.fill_diagonal(d, 0.0)
    return d


def _features_for(iso_codes: List[str], feature_sets: Tuple[str, ...]) -> np.ndarray:
    import lang2vec.lang2vec as l2v
    parts: List[np.ndarray] = []
    for fs in feature_sets:
        feats = l2v.get_features(iso_codes, fs)
        arr = np.array([feats[i] for i in iso_codes], dtype=np.float64)
        arr = np.where(arr < 0, 0.0, arr)
        parts.append(arr)
    return np.concatenate(parts, axis=1)


def build(varieties_module: str, out_dir: Path,
          types: List[str] | None = None) -> List[Path]:
    mod = importlib.import_module(varieties_module)
    codes: List[str] = list(mod.VARIETY_CODES)
    iso = [mod.LANG2VEC_CODE[c] for c in codes]
    out_dir.mkdir(parents=True, exist_ok=True)
    types = list(types or URIEL_RECIPES.keys())
    written = []
    for dt in types:
        try:
            X = _features_for(iso, URIEL_RECIPES[dt])
            mat = _cosine_distance_matrix(X)
        except Exception as exc:
            warnings.warn(f"URIEL {dt}: failed ({exc}) — skipping", stacklevel=2)
            continue
        meta = {
            "source": "URIEL via lang2vec 1.1.x",
            "distance_type": dt,
            "feature_sets": list(URIEL_RECIPES[dt]),
            "metric": "cosine distance on feature vectors (Littell et al., 2017)",
            "varieties_module": varieties_module,
            "codes": codes,
            "lang2vec_codes": iso,
            "warnings": [
                "Low-resource varieties may have URIEL features k-NN imputed.",
                "`genetic` is tree-derived; does not capture contact effects.",
            ],
        }
        out_path = out_dir / f"uriel_{dt}.npz"
        np.savez(out_path,
                 matrix=mat,
                 labels=np.array(codes, dtype=object),
                 meta=np.array([json.dumps(meta)], dtype=object))
        written.append(out_path)
        n_off = mat.size - len(codes)
        print(f"  uriel_{dt:<13s}  range=[{mat.min():.3f}, {mat.max():.3f}]  "
              f"mean_offdiag={mat.sum() / n_off:.3f}  → {out_path.name}")
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--varieties-module", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--types", nargs="+", choices=list(URIEL_RECIPES), default=None)
    args = ap.parse_args(argv)
    written = build(args.varieties_module, args.out_dir, args.types)
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
