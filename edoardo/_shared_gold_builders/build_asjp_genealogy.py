"""
Build a second genealogical proxy from ASJP's automated classification.

ASJP publishes a classification (in ``languages.csv`` columns
``classification_glottolog`` and ``classification_wals``) plus their own
automated WALS-style class string.  We use the **automated cluster
classification** (column ``classification_wals`` if present, otherwise
fall back to the WALS family/genus prefix from the auto-classifier) and
compute LCA-style tree distance like ``glottolog_tree`` but with this
independent grouping.

Why this is useful: ``glottolog_tree`` and ``uriel_genetic`` both come
from the same Glottolog phylogeny.  ASJP's automated classification is
DERIVED FROM LEXICAL DATA — so disagreements between asjp_genealogy
and glottolog_tree highlight where lexical-similarity grouping
disagrees with the historical taxonomy.

CLI:
    python -m edoardo._shared_gold_builders.build_asjp_genealogy \\
        --varieties-module edoardo.exp1_uriel_native.varieties \\
        --out-dir edoardo/exp1_uriel_native/gold_references/matrices
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
import urllib.request
import warnings
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np


ASJP_URL = "https://github.com/lexibank/asjp/archive/refs/tags/v20.zip"
DEFAULT_CACHE = Path.home() / ".cache" / "asjp"


def _ensure_asjp(cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    zip_path = cache / "asjp_v20.zip"
    if not zip_path.exists():
        print(f"  downloading {ASJP_URL}")
        urllib.request.urlretrieve(ASJP_URL, zip_path)
    extract_dir = cache / "extracted"
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)
    inner = next((p for p in extract_dir.iterdir() if p.is_dir()), None)
    if inner is None:
        raise RuntimeError(f"ASJP archive at {zip_path} is empty.")
    return inner


def _load_classifications(cldf_root: Path,
                          iso_codes: List[str]) -> Dict[str, List[str]]:
    """Return ``{iso: [path components from ASJP classification_wals]}``."""
    cldf_dir = cldf_root / "cldf"
    if not cldf_dir.exists():
        cldf_dir = next((p for p in cldf_root.rglob("languages.csv")
                         if p.is_file())).parent
    out: Dict[str, List[str]] = {}
    iso_set = set(iso_codes)
    with (cldf_dir / "languages.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            iso = (row.get("ISO639P3code") or "").strip()
            if iso not in iso_set:
                continue
            cls = (row.get("classification_wals")
                   or row.get("Classification_WALS")
                   or row.get("classification_glottolog")
                   or "").strip()
            if not cls:
                continue
            # ASJP classification is dot- or comma-separated path
            parts = [p.strip() for p in cls.replace(",", ".").split(".") if p.strip()]
            if iso not in out:
                out[iso] = parts
    return out


def _path_depth(p: List[str]) -> int:
    return len(p)


def _lca_depth(a: List[str], b: List[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x == y:
            n += 1
        else:
            break
    return n


def build(varieties_module: str, out_dir: Path,
          cache: Path = DEFAULT_CACHE) -> Path:
    mod = importlib.import_module(varieties_module)
    codes: List[str] = list(mod.VARIETY_CODES)
    iso_codes = [mod.LANG2VEC_CODE[c] for c in codes]

    cldf_root = _ensure_asjp(cache)
    cls = _load_classifications(cldf_root, iso_codes)

    paths: Dict[str, List[str]] = {}
    missing: List[str] = []
    for c, iso in zip(codes, iso_codes):
        if iso in cls:
            paths[c] = cls[iso]
        else:
            missing.append(c)
            paths[c] = []  # empty path → distance = 1.0 from anything else

    if not any(paths.values()):
        warnings.warn("ASJP classifications empty for ALL varieties — skipping.",
                      stacklevel=2)
        return out_dir / "asjp_genealogy.npz"

    max_depth = max(_path_depth(p) for p in paths.values()) or 1
    n = len(codes)
    d = np.zeros((n, n), dtype=np.float64)
    for i, a in enumerate(codes):
        for j, b in enumerate(codes):
            if i == j:
                continue
            la, lb = paths[a], paths[b]
            if not la or not lb:
                d[i, j] = 1.0
                continue
            lca = _lca_depth(la, lb)
            raw = (_path_depth(la) - lca) + (_path_depth(lb) - lca)
            d[i, j] = raw / (2.0 * max_depth)
    d = (d + d.T) / 2.0
    np.fill_diagonal(d, 0.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "asjp_genealogy.npz"
    meta = {
        "source": "ASJP automated classification (Wichmann et al., v20)",
        "metric": "LCA-based path distance on ASJP's automated classification path",
        "varieties_module": varieties_module,
        "codes": codes,
        "iso_codes": iso_codes,
        "asjp_paths": {c: paths[c] for c in codes},
        "missing": missing,
        "warnings": [
            "ASJP automated classification differs from Glottolog in some "
            "branches; intentional — captures lexical similarity grouping.",
            "Missing varieties get distance 1.0 from all others.",
        ],
    }
    np.savez(out_path,
             matrix=d,
             labels=np.array(codes, dtype=object),
             meta=np.array([json.dumps(meta)], dtype=object))
    n_off = d.size - len(codes)
    print(f"  asjp_genealogy  range=[{d.min():.3f}, {d.max():.3f}]  "
          f"mean_offdiag={d.sum() / n_off:.3f}  missing={missing or '∅'}  → {out_path.name}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--varieties-module", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = ap.parse_args(argv)
    build(args.varieties_module, args.out_dir, args.cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())
