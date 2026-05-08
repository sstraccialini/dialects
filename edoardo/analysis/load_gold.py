"""
Load gold reference distance matrices written by edoardo/gold_references/.

Each gold is an ``.npz`` with keys ``matrix``, ``labels``, ``meta``.

Usage:
    from edoardo.analysis.load_gold import discover_golds, load_gold

    for g in discover_golds():
        mat, codes, meta = load_gold(g)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


GOLDS_DIR_DEFAULT = Path(__file__).resolve().parents[1] / "gold_references" / "matrices"


@dataclass(frozen=True)
class GoldRef:
    name: str             # "uriel_genetic"
    path: Path

    @property
    def family(self) -> str:
        """Top-level group: 'uriel', 'glottolog', 'expert', 'asjp'."""
        return self.name.split("_", 1)[0]


def discover_golds(matrices_dir: Path | None = None) -> List[GoldRef]:
    matrices_dir = matrices_dir or GOLDS_DIR_DEFAULT
    if not matrices_dir.exists():
        return []
    out = []
    for p in sorted(matrices_dir.glob("*.npz")):
        out.append(GoldRef(name=p.stem, path=p))
    return out


def load_gold(g: GoldRef) -> Tuple[np.ndarray, List[str], Dict]:
    data = np.load(g.path, allow_pickle=True)
    mat = np.asarray(data["matrix"], dtype=np.float64)
    labels = [str(x) for x in data["labels"]]
    try:
        meta = json.loads(str(data["meta"][0]))
    except Exception:
        meta = {}
    return mat, labels, meta
