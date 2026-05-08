"""
Build an ASJP lexical-distance matrix using LDND.

Generic version: reads codes + ISO mapping from the varieties module.

CLI:
    python -m edoardo._shared_gold_builders.build_asjp_lexical \\
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
from typing import Dict, List, Tuple

import numpy as np


ASJP_URL = "https://github.com/lexibank/asjp/archive/refs/tags/v20.zip"
DEFAULT_CACHE = Path.home() / ".cache" / "asjp"


def _download(cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    zip_path = cache / "asjp_v20.zip"
    if not zip_path.exists():
        print(f"  downloading {ASJP_URL} → {zip_path}")
        urllib.request.urlretrieve(ASJP_URL, zip_path)
    extract_dir = cache / "extracted"
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)
    inner = next((p for p in extract_dir.iterdir() if p.is_dir()), None)
    if inner is None:
        raise RuntimeError(f"ASJP archive at {zip_path} is empty.")
    return inner


def _load_asjp_forms(cldf_root: Path,
                     iso_codes: List[str]) -> Dict[str, Dict[str, str]]:
    cldf_dir = cldf_root / "cldf"
    if not cldf_dir.exists():
        cands = list(cldf_root.rglob("languages.csv"))
        if not cands:
            raise RuntimeError(f"Cannot find languages.csv under {cldf_root}")
        cldf_dir = cands[0].parent

    languages_csv = cldf_dir / "languages.csv"
    forms_csv = cldf_dir / "forms.csv"
    iso_set = set(iso_codes)
    lang_id_to_iso: Dict[str, str] = {}
    with languages_csv.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            iso = (row.get("ISO639P3code") or "").strip()
            if iso in iso_set and iso not in lang_id_to_iso.values():
                lang_id_to_iso[row["ID"]] = iso

    out: Dict[str, Dict[str, str]] = {iso: {} for iso in iso_codes}
    with forms_csv.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            lid = row.get("Language_ID", "")
            if lid not in lang_id_to_iso:
                continue
            iso = lang_id_to_iso[lid]
            concept = row.get("Parameter_ID", "")
            form = row.get("Form", "")
            out[iso].setdefault(concept, form)
    return out


def _ldnd(forms_a: Dict[str, str], forms_b: Dict[str, str]) -> float:
    shared = sorted(set(forms_a) & set(forms_b))
    if len(shared) < 5:
        return float("nan")

    def _lev(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        if not s2:
            return len(s1)
        prev = list(range(len(s2) + 1))
        for i, ca in enumerate(s1):
            cur = [i + 1]
            for j, cb in enumerate(s2):
                cost = 0 if ca == cb else 1
                cur.append(min(cur[j] + 1, prev[j + 1] + 1, prev[j] + cost))
            prev = cur
        return prev[-1]

    def _norm(a: str, b: str) -> float:
        if not a and not b:
            return 0.0
        return _lev(a, b) / max(len(a), len(b))

    same = [_norm(forms_a[c], forms_b[c]) for c in shared]
    LD = float(np.mean(same))
    mismatched = [_norm(forms_a[ca], forms_b[cb])
                  for ca in shared for cb in shared if ca != cb]
    if not mismatched:
        return LD
    E = float(np.mean(mismatched))
    return LD / E if E > 0 else LD


def build(varieties_module: str, out_dir: Path, cache: Path,
          allow_download: bool = True) -> Path:
    mod = importlib.import_module(varieties_module)
    codes: List[str] = list(mod.VARIETY_CODES)
    iso_codes = [mod.LANG2VEC_CODE[c] for c in codes]

    if allow_download:
        cldf_root = _download(cache)
    else:
        cldf_root = cache / "extracted"
        if not cldf_root.exists():
            raise RuntimeError(f"No cached ASJP at {cache}; allow download.")

    forms = _load_asjp_forms(cldf_root, iso_codes)
    missing = [c for c, iso in zip(codes, iso_codes) if not forms.get(iso)]

    n = len(codes)
    mat = np.full((n, n), float("nan"), dtype=np.float64)
    np.fill_diagonal(mat, 0.0)
    for i, a in enumerate(codes):
        fa = forms.get(mod.LANG2VEC_CODE[a], {})
        if not fa:
            continue
        for j in range(i + 1, n):
            b = codes[j]
            fb = forms.get(mod.LANG2VEC_CODE[b], {})
            if not fb:
                continue
            d = _ldnd(fa, fb)
            mat[i, j] = mat[j, i] = d

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "asjp_ldnd.npz"
    meta = {
        "source": "ASJP database v20 (Wichmann et al.)",
        "url": ASJP_URL,
        "metric": "LDND (Levenshtein Distance Normalised Divided)",
        "varieties_module": varieties_module,
        "codes": codes,
        "iso_codes": iso_codes,
        "missing_iso_in_asjp": missing,
        "warnings": [
            "Italian dialect coverage in ASJP is partial; missing varieties become NaN.",
            "Downstream code drops NaN before correlation tests.",
        ],
    }
    np.savez(out_path,
             matrix=mat,
             labels=np.array(codes, dtype=object),
             meta=np.array([json.dumps(meta)], dtype=object))
    finite = mat[np.isfinite(mat)]
    rng = (float(finite.min()), float(finite.max())) if finite.size else (np.nan, np.nan)
    print(f"  asjp_ldnd       range=[{rng[0]:.3f}, {rng[1]:.3f}]  "
          f"missing={missing or '∅'}  → {out_path.name}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--varieties-module", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--no-download", dest="download", action="store_false", default=True)
    args = ap.parse_args(argv)
    build(args.varieties_module, args.out_dir, args.cache, args.download)
    return 0


if __name__ == "__main__":
    sys.exit(main())
