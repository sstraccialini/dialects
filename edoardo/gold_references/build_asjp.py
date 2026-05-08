"""
Build an ASJP-based lexical distance matrix.

ASJP (Automated Similarity Judgment Program; Wichmann et al.) provides
40 Swadesh-list items in a coarse phonetic alphabet (ASJPcode) for
~7000 languages.  The standard distance is LDND — Levenshtein Distance
Normalised Divided — which removes the chance-similarity baseline.

This builder downloads the public ASJP CLDF release once into a cache
directory and then computes LDND between the 13 varieties.  Coverage
notes (May 2026):
  - Italian dialects: ASJP often has only one entry per ISO code, with
    the orthographic Italian standard form.  Lombard, Ligurian, Sicilian
    DO have separate ASJP entries; Friulian and Venetian sometimes do.
  - Sardinian is well covered.
  - English, German, Slovenian, Italian, Spanish, French, Catalan: full
    coverage.

Missing varieties are flagged in the meta JSON; the resulting matrix
contains NaN rows/columns for them, and downstream Mantel/Spearman code
must filter those.

Run from the repo root:
    python -m edoardo.gold_references.build_asjp                # auto-download
    python -m edoardo.gold_references.build_asjp --no-download  # use cached only

Output: ``matrices/asjp_ldnd.npz`` with keys matrix, labels, meta.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import warnings
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from edoardo.varieties_extra import LANG2VEC_CODE, VARIETY_CODES


# Public ASJP release — pinned so results are reproducible.
ASJP_URL = "https://github.com/lexibank/asjp/archive/refs/tags/v20.zip"
DEFAULT_CACHE = Path.home() / ".cache" / "asjp"


# Mapping: our codes -> ASJP "iso_code" column.
# ASJP records use ISO 639-3.  Same fix as URIEL: sc → srd.
ASJP_ISO = {c: LANG2VEC_CODE[c] for c in VARIETY_CODES}


def _download(cache: Path) -> Path:
    """Download and unpack the ASJP release; return the unpacked root."""
    cache.mkdir(parents=True, exist_ok=True)
    zip_path = cache / "asjp_v20.zip"
    if not zip_path.exists():
        print(f"  downloading {ASJP_URL} → {zip_path}")
        urllib.request.urlretrieve(ASJP_URL, zip_path)  # noqa: S310 (trusted URL)
    extract_dir = cache / "extracted"
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)
    # Locate the cldf folder inside the extracted release.
    inner = next((p for p in extract_dir.iterdir() if p.is_dir()), None)
    if inner is None:
        raise RuntimeError(f"ASJP archive at {zip_path} is empty.")
    return inner


def _load_asjp_forms(cldf_root: Path, iso_codes: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Parse ASJP CLDF to return ``{iso_code: {concept_id: ASJP_string}}``.
    Concept IDs are the 40 Swadesh items used by ASJP (1..40).

    We only keep one languoid per ISO code (the first match) for simplicity.
    """
    import csv

    cldf_dir = cldf_root / "cldf"
    if not cldf_dir.exists():
        # alternative layout
        cands = list(cldf_root.rglob("languages.csv"))
        if not cands:
            raise RuntimeError(f"Cannot find languages.csv under {cldf_root}")
        cldf_dir = cands[0].parent

    languages_csv = cldf_dir / "languages.csv"
    forms_csv = cldf_dir / "forms.csv"
    parameters_csv = cldf_dir / "parameters.csv"
    for fp in (languages_csv, forms_csv, parameters_csv):
        if not fp.exists():
            raise RuntimeError(f"Missing {fp.name} in {cldf_dir}")

    # Map Language_ID -> ISO code (filter to the ISOs we want).
    iso_set = set(iso_codes)
    lang_id_to_iso: Dict[str, str] = {}
    with languages_csv.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            iso = (row.get("ISO639P3code") or row.get("Glottocode") or "").strip()
            if iso in iso_set and iso not in lang_id_to_iso.values():
                lang_id_to_iso[row["ID"]] = iso

    out: Dict[str, Dict[str, str]] = {iso: {} for iso in iso_codes}
    with forms_csv.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            lid = row["Language_ID"]
            if lid not in lang_id_to_iso:
                continue
            iso = lang_id_to_iso[lid]
            concept = row["Parameter_ID"]
            form = row["Form"]
            # Keep only the first form per (iso, concept).
            out[iso].setdefault(concept, form)
    return out


def _ldnd(forms_a: Dict[str, str], forms_b: Dict[str, str]) -> float:
    """Levenshtein Distance Normalised Divided (Wichmann et al.).

    For each shared concept, compute Levenshtein distance / max length.
    LD = mean of those.  LDND = LD / E, where E is the same statistic
    averaged over MISMATCHED concepts (chance-similarity baseline).
    """
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

    # Mismatched-concept baseline.
    mismatched = []
    for ca in shared:
        for cb in shared:
            if ca == cb:
                continue
            mismatched.append(_norm(forms_a[ca], forms_b[cb]))
    if not mismatched:
        return LD
    E = float(np.mean(mismatched))
    return LD / E if E > 0 else LD


def build_asjp_matrix(cache: Path, codes: List[str], allow_download: bool
                      ) -> Tuple[np.ndarray, List[str], List[str]]:
    iso_codes = [ASJP_ISO[c] for c in codes]
    cldf_root = cache / "extracted"
    if allow_download:
        cldf_root = _download(cache)
    elif not cldf_root.exists():
        raise RuntimeError(
            f"No cached ASJP at {cache}; pass --download or run once with internet.")
    forms = _load_asjp_forms(cldf_root, iso_codes)

    missing = [c for c, iso in zip(codes, iso_codes) if not forms.get(iso)]
    n = len(codes)
    mat = np.full((n, n), float("nan"), dtype=np.float64)
    np.fill_diagonal(mat, 0.0)
    for i, a in enumerate(codes):
        fa = forms.get(ASJP_ISO[a], {})
        if not fa:
            continue
        for j in range(i + 1, n):
            b = codes[j]
            fb = forms.get(ASJP_ISO[b], {})
            if not fb:
                continue
            d = _ldnd(fa, fb)
            mat[i, j] = mat[j, i] = d
    return mat, codes, missing


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).parent / "matrices")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                    help="Where to cache the ASJP download.")
    ap.add_argument("--no-download", dest="download", action="store_false",
                    default=True,
                    help="Use cached ASJP only; do not fetch from the network.")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    codes = list(VARIETY_CODES)
    try:
        mat, _, missing = build_asjp_matrix(args.cache, codes, args.download)
    except Exception as exc:
        warnings.warn(f"ASJP build failed ({exc}); writing empty stub.", stacklevel=2)
        mat = np.full((len(codes), len(codes)), float("nan"))
        np.fill_diagonal(mat, 0.0)
        missing = list(codes)

    meta = {
        "source": "ASJP database v20 (Wichmann et al.)",
        "url": ASJP_URL,
        "metric": "LDND (Levenshtein Distance Normalised Divided)",
        "codes": codes,
        "missing_iso_in_asjp": missing,
        "warnings": [
            "Italian dialect coverage in ASJP is partial; missing varieties become NaN rows.",
            "Downstream code must drop NaN before correlation tests.",
        ],
    }
    out_path = args.out_dir / "asjp_ldnd.npz"
    np.savez(
        out_path,
        matrix=mat,
        labels=np.array(codes, dtype=object),
        meta=np.array([json.dumps(meta)], dtype=object),
    )
    finite = mat[np.isfinite(mat)]
    rng = (float(finite.min()), float(finite.max())) if finite.size else (np.nan, np.nan)
    print(f"  asjp_ldnd        range=[{rng[0]:.3f}, {rng[1]:.3f}]  "
          f"missing={missing or '∅'}  → {out_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
