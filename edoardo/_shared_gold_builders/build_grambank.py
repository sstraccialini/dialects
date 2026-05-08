"""
Build a Grambank typology distance matrix.

Grambank (Skirgård et al. 2023, *Science Advances*) — 195 binary/categorical
typological features for ~2400 languages.  Distance = cosine distance on
the feature vectors after coding {0, 1, ?} as {0, 1, NaN}.

Coverage caveat: Grambank coverage of Italian dialects is **partial**.
Run-time output reports which target Glottocodes were missing; their
rows in the matrix become NaN and are dropped from downstream Spearman /
Mantel calculations.

CLI:
    python -m edoardo._shared_gold_builders.build_grambank \\
        --varieties-module edoardo.exp1_uriel_native.varieties \\
        --out-dir edoardo/exp1_uriel_native/gold_references/matrices
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from edoardo._shared_gold_builders.cldf_utils import (
    ensure_dataset, find_cldf_dir, read_cldf_table,
)


def _load_features(cldf_root: Path,
                   target_glottocodes: List[str]) -> Dict[str, Dict[str, float]]:
    """Return ``{glottocode: {feature_id: value}}``."""
    cldf = find_cldf_dir(cldf_root)
    langs = read_cldf_table(cldf / "languages.csv")
    glot_set = set(target_glottocodes)
    keep_lang_ids = {row["ID"]: row.get("Glottocode", "")
                     for row in langs
                     if (row.get("Glottocode") or "") in glot_set}
    if not keep_lang_ids:
        return {gc: {} for gc in target_glottocodes}

    values_path = cldf / "values.csv"
    out: Dict[str, Dict[str, float]] = {gc: {} for gc in target_glottocodes}
    rows = read_cldf_table(values_path)
    for row in rows:
        lid = row.get("Language_ID", "")
        if lid not in keep_lang_ids:
            continue
        gc = keep_lang_ids[lid]
        feat = row.get("Parameter_ID", "")
        val = row.get("Value", "")
        if val in ("", "?"):
            continue
        try:
            v = float(val)
        except ValueError:
            continue
        out[gc][feat] = v
    return out


def _vectorise(features_by_glot: Dict[str, Dict[str, float]],
               codes: List[str], glottocode_map: Dict[str, str]
               ) -> tuple[np.ndarray, List[str], List[str]]:
    """Convert per-Glottocode feature dicts into an aligned matrix.

    Returns
    -------
    matrix : (N, F) float, with NaN for varieties that had no features
    feat_order : list of feature IDs (length F)
    missing : list of variety codes that had zero features
    """
    all_feats = sorted({f for d in features_by_glot.values() for f in d})
    n = len(codes)
    f = len(all_feats)
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
    """Pairwise cosine distance; NaN-safe (uses pairwise complete features)."""
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

    cldf_root = ensure_dataset("grambank")
    feats = _load_features(cldf_root, [gmap[c] for c in codes])
    mat, feat_order, missing = _vectorise(feats, codes, gmap)
    dist = _cosine_distance_with_nan(mat)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "grambank.npz"
    meta = {
        "source":   "Grambank v1.0.3 (Skirgård et al. 2023)",
        "url":      "https://zenodo.org/records/7844558",
        "metric":   "cosine distance on binary feature vectors (NaN-safe pairwise)",
        "n_features_used": len(feat_order),
        "varieties_module": varieties_module,
        "codes":    codes,
        "glottocodes": [gmap[c] for c in codes],
        "missing_in_grambank": missing,
        "warnings": [
            "Grambank coverage of Italian dialects is partial.",
            "Missing-variety rows are NaN and get dropped from Spearman/Mantel.",
        ],
    }
    np.savez(out_path,
             matrix=dist,
             labels=np.array(codes, dtype=object),
             meta=np.array([json.dumps(meta)], dtype=object))
    finite = dist[np.isfinite(dist)]
    rng = (float(finite.min()), float(finite.max())) if finite.size else (np.nan, np.nan)
    print(f"  grambank        range=[{rng[0]:.3f}, {rng[1]:.3f}]  "
          f"missing={missing or '∅'}  features={len(feat_order)}  → {out_path.name}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--varieties-module", required=True,
                    help="Dotted path of the varieties module to use, e.g. "
                         "edoardo.exp1_uriel_native.varieties")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    build(args.varieties_module, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
