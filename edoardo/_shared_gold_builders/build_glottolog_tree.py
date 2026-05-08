"""
Build a hand-coded Glottolog tree distance matrix for an arbitrary varieties
module.  Reads ``GLOTTOLOG_PATH`` from the varieties module.

LCA-based path distance, normalised by 2 * max_depth.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import List

import numpy as np


def _path_depth(path: List[str]) -> int:
    return len(path)


def _lca_depth(a: List[str], b: List[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x == y:
            n += 1
        else:
            break
    return n


def build(varieties_module: str, out_dir: Path) -> Path:
    mod = importlib.import_module(varieties_module)
    codes: List[str] = list(mod.VARIETY_CODES)
    paths = {c: mod.GLOTTOLOG_PATH[c] for c in codes}
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
            d[i, j] = raw / (2.0 * max_depth)
    d = (d + d.T) / 2.0
    np.fill_diagonal(d, 0.0)

    meta = {
        "source":   "Hand-coded Glottolog 5.x (May 2026 snapshot)",
        "method":   "LCA-based path distance, normalised by 2*max_depth",
        "varieties_module": varieties_module,
        "codes":    codes,
        "paths":    {c: paths[c] for c in codes},
        "warnings": [
            "Tree topology only: no contact, no areal effects.",
            "Sub-branch labels are simplified relative to upstream Glottolog.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "glottolog_tree.npz"
    np.savez(out_path,
             matrix=d,
             labels=np.array(codes, dtype=object),
             meta=np.array([json.dumps(meta)], dtype=object))
    n_off = d.size - len(codes)
    print(f"  glottolog_tree  range=[{d.min():.3f}, {d.max():.3f}]  "
          f"mean_offdiag={d.sum() / n_off:.3f}  → {out_path.name}")
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
