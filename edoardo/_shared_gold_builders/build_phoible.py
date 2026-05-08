"""
Build a PHOIBLE phoneme-inventory distance matrix.

PHOIBLE 2.0.1 (Moran & McCloy 2019) provides phoneme inventories for
~3000 languages.  We compute Jaccard distance between varieties:

    d(A, B) = 1 - |A ∩ B| / |A ∪ B|

where A and B are the sets of phonemes attributed to each variety.

When a Glottocode has multiple PHOIBLE inventories (different sources:
SPA / UPSID / AA / PH / RA / GM / EA), we take the **union** of all
their phonemes.  This is a coarse choice but the most inclusive one;
alternatives (intersection, source-restricted) are documented in meta.

CLI:
    python -m edoardo._shared_gold_builders.build_phoible \\
        --varieties-module edoardo.exp1_uriel_native.varieties \\
        --out-dir edoardo/exp1_uriel_native/gold_references/matrices
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

from edoardo._shared_gold_builders.cldf_utils import (
    ensure_dataset, find_cldf_dir, read_cldf_table,
)


def _load_inventories(cldf_root: Path,
                      target_glottocodes: List[str]) -> Dict[str, Set[str]]:
    cldf = find_cldf_dir(cldf_root)
    langs = read_cldf_table(cldf / "languages.csv")
    glot_set = set(target_glottocodes)
    lang_id_to_glot = {row["ID"]: row["Glottocode"]
                       for row in langs
                       if (row.get("Glottocode") or "") in glot_set}

    inv: Dict[str, Set[str]] = defaultdict(set)
    values_path = cldf / "values.csv"
    rows = read_cldf_table(values_path)
    for row in rows:
        lid = row.get("Language_ID", "")
        if lid not in lang_id_to_glot:
            continue
        # PHOIBLE values store Parameter_ID = phoneme ID.
        phon = row.get("Parameter_ID", "")
        if phon:
            inv[lang_id_to_glot[lid]].add(phon)
    return inv


def _jaccard_distance(codes: List[str],
                      inventories: Dict[str, Set[str]],
                      glottocode_map: Dict[str, str]
                      ) -> tuple[np.ndarray, List[str]]:
    n = len(codes)
    d = np.zeros((n, n), dtype=np.float64)
    missing: List[str] = []
    for i, c in enumerate(codes):
        if not inventories.get(glottocode_map[c]):
            missing.append(c)
    for i, ci in enumerate(codes):
        ai = inventories.get(glottocode_map[ci], set())
        for j in range(i + 1, n):
            cj = codes[j]
            aj = inventories.get(glottocode_map[cj], set())
            if not ai or not aj:
                d[i, j] = d[j, i] = np.nan
                continue
            union = ai | aj
            if not union:
                d[i, j] = d[j, i] = np.nan
                continue
            inter = ai & aj
            d[i, j] = d[j, i] = 1.0 - len(inter) / len(union)
    return d, missing


def build(varieties_module: str, out_dir: Path) -> Path:
    mod = importlib.import_module(varieties_module)
    codes: List[str] = list(mod.VARIETY_CODES)
    gmap: Dict[str, str] = mod.GLOTTOCODE

    cldf_root = ensure_dataset("phoible")
    invs = _load_inventories(cldf_root, [gmap[c] for c in codes])
    dist, missing = _jaccard_distance(codes, invs, gmap)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phoible.npz"
    meta = {
        "source":   "PHOIBLE 2.0.1 (Moran & McCloy 2019)",
        "url":      "https://github.com/cldf-datasets/phoible/releases/tag/v2.0.1",
        "metric":   "Jaccard distance on phoneme inventories (union across PHOIBLE sources)",
        "varieties_module": varieties_module,
        "codes":    codes,
        "glottocodes": [gmap[c] for c in codes],
        "inventory_sizes": {c: len(invs.get(gmap[c], set())) for c in codes},
        "missing_in_phoible": missing,
        "warnings": [
            "PHOIBLE coverage of Italian dialects is partial; missing varieties become NaN.",
            "When a language has multiple inventories in PHOIBLE we union all phonemes.",
        ],
    }
    np.savez(out_path,
             matrix=dist,
             labels=np.array(codes, dtype=object),
             meta=np.array([json.dumps(meta)], dtype=object))
    finite = dist[np.isfinite(dist)]
    rng = (float(finite.min()), float(finite.max())) if finite.size else (np.nan, np.nan)
    print(f"  phoible         range=[{rng[0]:.3f}, {rng[1]:.3f}]  "
          f"missing={missing or '∅'}  → {out_path.name}")
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
