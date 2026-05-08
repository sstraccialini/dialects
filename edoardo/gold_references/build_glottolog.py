"""
Build a tree-based distance matrix from a hand-coded Glottolog hierarchy.

For each pair (a, b) we compute:

    d(a, b) = (depth(a) + depth(b) - 2 * depth(LCA(a, b))) / max_depth

where depth() is the path length from the root and LCA is the lowest
common ancestor on the hierarchy.  Distances are normalised to [0, 1].

This complements ``uriel_genetic``: lang2vec's genetic distance uses
Glottolog under the hood but with heavy k-NN imputation for missing
sub-branches.  The hand-coded version here is *only* tree topology
(no imputation), so disagreements between this matrix and
``uriel_genetic`` highlight where lang2vec's imputation is doing work.

Output: ``matrices/glottolog_tree.npz`` with keys
    matrix, labels, meta

Run from the repo root:
    python -m edoardo.gold_references.build_glottolog
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

from edoardo.varieties_extra import GLOTTOLOG_PATH, VARIETY_CODES


def _path_depth(path: List[str]) -> int:
    return len(path)


def _lca_depth(a: List[str], b: List[str]) -> int:
    """Number of leading nodes shared by two paths."""
    n = 0
    for x, y in zip(a, b):
        if x == y:
            n += 1
        else:
            break
    return n


def build_glottolog_distance(codes: List[str]) -> np.ndarray:
    paths = {c: GLOTTOLOG_PATH[c] for c in codes}
    max_depth = max(_path_depth(p) for p in paths.values())

    n = len(codes)
    d = np.zeros((n, n), dtype=np.float64)
    for i, a in enumerate(codes):
        for j, b in enumerate(codes):
            if i == j:
                continue
            la, lb = paths[a], paths[b]
            lca = _lca_depth(la, lb)
            raw = (_path_depth(la) - lca) + (_path_depth(lb) - lca)
            # normalise so that the most distant pair (no shared ancestor at all)
            # would have distance ~ 1.0 if both paths are at max depth.
            d[i, j] = raw / (2.0 * max_depth)
    # Symmetrise (guards against floating noise; should already be symmetric).
    d = (d + d.T) / 2.0
    np.fill_diagonal(d, 0.0)
    return d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).parent / "matrices")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    codes = list(VARIETY_CODES)
    mat = build_glottolog_distance(codes)

    meta = {
        "source": "Hand-coded Glottolog 5.x (May 2026 snapshot)",
        "method": "LCA-based path distance, normalised by 2*max_depth",
        "codes": codes,
        "paths": {c: GLOTTOLOG_PATH[c] for c in codes},
        "warnings": [
            "Tree topology only: no contact, no areal effects.",
            "Sub-branch labels are slightly simplified vs upstream Glottolog.",
        ],
    }
    out_path = args.out_dir / "glottolog_tree.npz"
    np.savez(
        out_path,
        matrix=mat,
        labels=np.array(codes, dtype=object),
        meta=np.array([json.dumps(meta)], dtype=object),
    )
    n_offdiag = mat.size - len(codes)
    print(f"  glottolog_tree   range=[{mat.min():.3f}, {mat.max():.3f}]  "
          f"mean_offdiag={mat.sum() / n_offdiag:.3f}  → {out_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
