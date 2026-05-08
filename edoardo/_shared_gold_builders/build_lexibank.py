"""
Build a Lexibank lexical-feature distance matrix.

Lexibank-analysed v2.1 (List et al. 2022) aggregates ~2000 wordlist
datasets (including ASJP, IDS, NorthEuraLex, etc.) into per-language
feature vectors covering lexical typology and phonotactics.

Distance = cosine on the ``lexicon-values.csv`` feature vector for
each variety.  When multiple Lexibank entries exist per Glottocode
(e.g., ASJP entry + IDS entry), we average their feature vectors.

CLI:
    python -m edoardo._shared_gold_builders.build_lexibank \\
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
from typing import Dict, List

import numpy as np

from edoardo._shared_gold_builders.cldf_utils import (
    ensure_dataset, find_cldf_dir, read_cldf_table,
)


def _load_lexicon_features(cldf_root: Path,
                           target_glottocodes: List[str]
                           ) -> Dict[str, Dict[str, float]]:
    """Return ``{glottocode: {feature_id: averaged_value}}``."""
    cldf = find_cldf_dir(cldf_root)
    langs = read_cldf_table(cldf / "languages.csv")
    glot_set = set(target_glottocodes)
    keep_lang_ids: Dict[str, str] = {}
    for row in langs:
        gc = row.get("Glottocode", "") or ""
        if gc in glot_set:
            keep_lang_ids[row["ID"]] = gc

    # Lexibank-analysed has lexicon-values.csv with rows
    #   ID, Language_ID, Parameter_ID, Value, ...
    values_csv = cldf / "lexicon-values.csv"
    if not values_csv.exists():
        # Fallback to phonology-values for older releases.
        values_csv = cldf / "phonology-values.csv"
        if not values_csv.exists():
            raise RuntimeError(
                f"Neither lexicon-values.csv nor phonology-values.csv at {cldf}.")

    bucket: Dict[str, Dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    rows = read_cldf_table(values_csv)
    for row in rows:
        lid = row.get("Language_ID", "")
        if lid not in keep_lang_ids:
            continue
        gc = keep_lang_ids[lid]
        feat = row.get("Parameter_ID", "")
        val = row.get("Value", "")
        try:
            v = float(val)
        except (ValueError, TypeError):
            continue
        bucket[gc][feat].append(v)

    return {gc: {f: float(np.mean(vs)) for f, vs in feats.items()}
            for gc, feats in bucket.items()}


def _vectorise(features_by_glot: Dict[str, Dict[str, float]],
               codes: List[str], glottocode_map: Dict[str, str]
               ) -> tuple[np.ndarray, List[str], List[str]]:
    all_feats = sorted({f for d in features_by_glot.values() for f in d})
    n, f = len(codes), len(all_feats)
    if f == 0:
        return np.full((n, 0), np.nan), [], list(codes)
    feat_idx = {fid: i for i, fid in enumerate(all_feats)}
    mat = np.full((n, f), np.nan, dtype=np.float64)
    missing: List[str] = []
    for i, c in enumerate(codes):
        gc = glottocode_map[c]
        feats = features_by_glot.get(gc, {})
        if not feats:
            missing.append(c)
            continue
        for fid, v in feats.items():
            mat[i, feat_idx[fid]] = v
    return mat, all_feats, missing


def _cosine_distance_with_nan(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            mask = np.isfinite(X[i]) & np.isfinite(X[j])
            if mask.sum() < 5:
                d[i, j] = d[j, i] = np.nan
                continue
            a, b = X[i, mask], X[j, mask]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na == 0 or nb == 0:
                d[i, j] = d[j, i] = np.nan
                continue
            d[i, j] = d[j, i] = 1.0 - float(a @ b / (na * nb))
    return d


def build(varieties_module: str, out_dir: Path) -> Path:
    mod = importlib.import_module(varieties_module)
    codes: List[str] = list(mod.VARIETY_CODES)
    gmap: Dict[str, str] = mod.GLOTTOCODE

    cldf_root = ensure_dataset("lexibank")
    feats = _load_lexicon_features(cldf_root, [gmap[c] for c in codes])
    mat, feat_order, missing = _vectorise(feats, codes, gmap)
    dist = _cosine_distance_with_nan(mat)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "lexibank.npz"
    meta = {
        "source":   "Lexibank-analysed v2.1 (List et al. 2022, List & Forkel 2024)",
        "url":      "https://github.com/lexibank/lexibank-analysed/releases/tag/v2.1",
        "metric":   "cosine on aggregated lexicon-values.csv feature vector",
        "n_features_used": len(feat_order),
        "varieties_module": varieties_module,
        "codes":    codes,
        "glottocodes": [gmap[c] for c in codes],
        "missing_in_lexibank": missing,
        "warnings": [
            "When multiple Lexibank entries exist for one Glottocode, feature "
            "values are averaged (mean across sub-datasets).",
        ],
    }
    np.savez(out_path,
             matrix=dist,
             labels=np.array(codes, dtype=object),
             meta=np.array([json.dumps(meta)], dtype=object))
    finite = dist[np.isfinite(dist)]
    rng = (float(finite.min()), float(finite.max())) if finite.size else (np.nan, np.nan)
    print(f"  lexibank        range=[{rng[0]:.3f}, {rng[1]:.3f}]  "
          f"missing={missing or '∅'}  features={len(feat_order)}  → {out_path.name}")
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
