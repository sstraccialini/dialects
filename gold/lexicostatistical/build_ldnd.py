"""
Compute a lexicostatistical distance matrix from a parallel wordlist.

Pipeline:
    1. Load CSV (concept_en, ita, fra, ..., vec, notes).
    2. For each cell, take the first comma-separated form (canonical),
       strip whitespace, lowercase.
    3. Convert native orthography to ASJPcode (Wichmann et al. 2009)
       via a small ortographic-to-phonetic mapping per source language.
    4. For each pair (A, B), compute LDND = LD / E:
         LD = mean over concepts c of  lev(A_c, B_c) / max(|A_c|, |B_c|)
         E  = mean over (c1 != c2) of  lev(A_c1, B_c2) / max(|A_c1|, |B_c2|)
    5. Bootstrap CI: resample concept rows with replacement, recompute
       LDND, take 2.5/97.5 percentiles.

Output:
    matrices/lexicostat_ldnd.npz  with keys
        matrix       (N, N) float — LDND distances
        matrix_lo    (N, N) float — bootstrap 2.5 percentile
        matrix_hi    (N, N) float — bootstrap 97.5 percentile
        labels       (N,)   variety codes
        meta         JSON   metadata
    matrices/lexicostat_asjpcode.csv — every (concept, variety) → ASJPcode
        for audit / debugging.

Run:
    python -m gold.lexicostatistical.build_ldnd \
        --wordlist gold/lexicostatistical/wordlist_v1_asjp40.csv \
        --out-dir  gold/lexicostatistical/matrices \
        --bootstrap 1000
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


VARIETIES = ["ita", "fra", "spa", "cat", "deu", "slv", "eng",
             "fur", "lij", "lmo", "sc", "scn", "vec"]


# --------------------------------------------------------------------------- #
# Native orthography → ASJPcode (Wichmann et al. 2009).
# ASJPcode uses 7 vowels (i e E a 3 u o) and 27 consonants on ASCII.
# This is a coarse mapping per language; for paper-quality results, a real
# G2P (e.g. epitran) per language should replace this.  The simplification
# is good enough for cross-variety Levenshtein because all 13 varieties go
# through the same broad rules.
# --------------------------------------------------------------------------- #

_BASE_MAP = {
    # vowels (we collapse [əɪ] etc. to closest ASJP vowel)
    "à": "a", "á": "a", "â": "a", "ã": "a", "ä": "a",
    "è": "E", "é": "e", "ê": "e", "ë": "e",
    "ì": "i", "í": "i", "î": "i", "ï": "i",
    "ò": "o", "ó": "o", "ô": "o", "õ": "o", "ö": "o",
    "ù": "u", "ú": "u", "û": "u", "ü": "u",
    "y": "i",
    # special letters
    "ñ": "n",   # Spanish
    "ç": "S",   # Catalan, French → ASJP /ʃ/ symbol
    "ß": "s",   # German
    "œ": "o",   # French
    "æ": "E",
    "ø": "o",
    "ð": "d",
    "þ": "t",
    "č": "c",   # Slovenian → ASJP /tʃ/
    "š": "S",
    "ž": "Z",
    "ł": "l",
    "ń": "n",
    "ź": "Z",
    "ś": "S",
    "ć": "c",
    "ŋ": "N",
    # diacritic-stripped Italian dialect specials
    "ɖ": "d",   # cacuminal in Sicilian
    "ʃ": "S",
    "ʎ": "L",
    "ɲ": "N",
    "ʔ": "7",
    "ʁ": "r",
}


def _strip_diacritics(s: str) -> str:
    """Strip combining marks, keep base letters."""
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def _to_asjp(word: str, language: str) -> str:
    """
    Convert a native-orthography word to ASJPcode.

    Per-language digraph rules are applied first, then a base map, then
    diacritic stripping, then lowercasing.  This is intentionally simple;
    the goal is consistency across our 13 varieties, not phonetic accuracy.
    """
    w = word.strip().lower()
    # Drop punctuation entirely, keep only letters and a few marks.
    w = re.sub(r"[^\w\sàáâãäèéêëìíîïòóôõöùúûüçñńśźćčšžɖʃʎɲʔʁœæøðþßł]", "", w)
    w = w.replace(" ", "")

    # Per-language digraph normalisation BEFORE base map (so 'gn' = ɲ etc.)
    if language in ("ita", "fur", "lij", "lmo", "sc", "scn", "vec"):
        # Italian-style digraphs
        w = (w.replace("gn", "N")
              .replace("gli", "L")
              .replace("ggh", "g")    # scn double cacuminal kept simple
              .replace("ɖɖ", "d")
              .replace("dd", "d")
              .replace("cch", "k")
              .replace("ch", "k")
              .replace("gh", "g")
              .replace("ce", "ce")    # leave; later 'c' before 'e' handled
              .replace("ci", "ci")
              .replace("sce", "Se")
              .replace("sci", "Si")
              .replace("sc", "sk")    # before non-i/e becomes /sk/
              .replace("sch", "sk")
              .replace("zz", "z")
              .replace("qu", "kw")
              .replace("c", "k"))     # last-resort: c → k (digraph already replaced ce/ci)
    if language == "fra":
        w = (w.replace("ch", "S")
              .replace("eau", "o")
              .replace("au", "o")
              .replace("ou", "u")
              .replace("oi", "wa")
              .replace("ai", "E")
              .replace("ei", "E")
              .replace("eu", "o")
              .replace("on", "o")
              .replace("an", "a")
              .replace("en", "a")
              .replace("in", "E")
              .replace("un", "o")
              .replace("ill", "j")
              .replace("gn", "N")
              .replace("gu", "g")
              .replace("qu", "k"))
    if language == "spa":
        w = (w.replace("ch", "c")
              .replace("ll", "j")
              .replace("rr", "r")
              .replace("qu", "k"))
    if language == "cat":
        w = (w.replace("ny", "N")
              .replace("ll", "L")
              .replace("ix", "S")
              .replace("ig", "c")
              .replace("ç", "s"))
    if language == "deu":
        w = (w.replace("sch", "S")
              .replace("ch", "x")
              .replace("ck", "k")
              .replace("ng", "N")
              .replace("pf", "f")
              .replace("ie", "i")
              .replace("ei", "aj")
              .replace("eu", "oj")
              .replace("au", "aw")
              .replace("v", "f")
              .replace("z", "ts")
              .replace("ä", "E")
              .replace("ö", "o")
              .replace("ü", "u"))
    if language == "slv":
        # already mostly phonetic
        w = w.replace("nj", "N").replace("lj", "L").replace("dž", "Z")
    if language == "eng":
        # very crude; English orthography vs phonology is a known mess
        w = (w.replace("th", "t")
              .replace("sh", "S")
              .replace("ch", "c")
              .replace("ph", "f")
              .replace("ck", "k")
              .replace("ee", "i")
              .replace("oo", "u")
              .replace("ea", "i")
              .replace("ou", "aw")
              .replace("ai", "ej")
              .replace("ay", "ej"))

    # Base map for any remaining diacritics
    w = "".join(_BASE_MAP.get(c, c) for c in w)

    # Strip remaining combining marks
    w = _strip_diacritics(w)
    return w


# --------------------------------------------------------------------------- #
# Levenshtein
# --------------------------------------------------------------------------- #

# Use C-backed Levenshtein when available (~50× faster than pure Python).
try:
    import Levenshtein as _lev_lib
    def _levenshtein(s1: str, s2: str) -> int:
        return _lev_lib.distance(s1, s2)
except ImportError:
    def _levenshtein(s1: str, s2: str) -> int:
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


def _norm_lev(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    return _levenshtein(a, b) / max(len(a), len(b))


# --------------------------------------------------------------------------- #
# LDND
# --------------------------------------------------------------------------- #

def _ld_pair(forms_a: List[str], forms_b: List[str],
             concept_idx: List[int]) -> float:
    """Plain mean normalised Levenshtein over the concepts in concept_idx."""
    pairs = [(forms_a[i], forms_b[i]) for i in concept_idx
             if forms_a[i] and forms_b[i]]
    if len(pairs) < 5:
        return float("nan")
    return float(np.mean([_norm_lev(a, b) for a, b in pairs]))


def _ldnd_pair(forms_a: List[str], forms_b: List[str],
               concept_idx: List[int]) -> float:
    """LDND = LD / E; chance-baseline-normalised."""
    sub_a = [forms_a[i] for i in concept_idx]
    sub_b = [forms_b[i] for i in concept_idx]
    pairs = [(a, b) for a, b in zip(sub_a, sub_b) if a and b]
    if len(pairs) < 5:
        return float("nan")
    LD = float(np.mean([_norm_lev(a, b) for a, b in pairs]))
    cross = [_norm_lev(a, b)
             for i, (a, _) in enumerate(pairs)
             for j, (_, b) in enumerate(pairs) if i != j]
    if not cross:
        return LD
    E = float(np.mean(cross))
    return LD / E if E > 0 else LD


def _compute_one_pair(args):
    """Worker for multiprocessing.  Computes (i, j, ld, ldnd, ld_lo, ld_hi, ldnd_lo, ldnd_hi)."""
    i, j, forms_a, forms_b, n_concepts, bootstrap, seed = args
    ld_obs = _ld_pair(forms_a, forms_b, list(range(n_concepts)))
    ldnd_obs = _ldnd_pair(forms_a, forms_b, list(range(n_concepts)))
    ld_lo = ld_hi = ldnd_lo = ldnd_hi = float("nan")
    if bootstrap > 0 and (np.isfinite(ld_obs) or np.isfinite(ldnd_obs)):
        rng = np.random.default_rng(seed + i * 1000 + j)
        ld_samples, ldnd_samples = [], []
        for _ in range(bootstrap):
            idx = rng.choice(n_concepts, size=n_concepts, replace=True).tolist()
            s_ld = _ld_pair(forms_a, forms_b, idx)
            s_ldnd = _ldnd_pair(forms_a, forms_b, idx)
            if np.isfinite(s_ld):
                ld_samples.append(s_ld)
            if np.isfinite(s_ldnd):
                ldnd_samples.append(s_ldnd)
        if ld_samples:
            ld_lo = float(np.percentile(ld_samples, 2.5))
            ld_hi = float(np.percentile(ld_samples, 97.5))
        if ldnd_samples:
            ldnd_lo = float(np.percentile(ldnd_samples, 2.5))
            ldnd_hi = float(np.percentile(ldnd_samples, 97.5))
    return (i, j, ld_obs, ldnd_obs, ld_lo, ld_hi, ldnd_lo, ldnd_hi)


def compute_matrices(forms: Dict[str, List[str]],
                     varieties: List[str],
                     bootstrap: int = 0,
                     seed: int = 42,
                     n_jobs: int = 1
                     ) -> Dict[str, np.ndarray]:
    """
    Returns a dict with keys
        ld, ld_lo, ld_hi          — plain mean normalised Levenshtein + 95% CI
        ldnd, ldnd_lo, ldnd_hi    — LDND + 95% CI
    All matrices are NxN where N = len(varieties).
    """
    n = len(varieties)
    n_concepts = len(forms[varieties[0]])
    ld = np.zeros((n, n), dtype=np.float64)
    ld_lo = np.full((n, n), np.nan)
    ld_hi = np.full((n, n), np.nan)
    ldnd = np.zeros((n, n), dtype=np.float64)
    ldnd_lo = np.full((n, n), np.nan)
    ldnd_hi = np.full((n, n), np.nan)

    # Build the list of pairs to process
    tasks = []
    for i in range(n):
        for j in range(i + 1, n):
            tasks.append((i, j, forms[varieties[i]], forms[varieties[j]],
                          n_concepts, bootstrap, seed))

    if n_jobs <= 1:
        results = [_compute_one_pair(t) for t in tasks]
    else:
        from multiprocessing import Pool
        with Pool(n_jobs) as pool:
            results = pool.map(_compute_one_pair, tasks)

    for (i, j, ld_obs, ldnd_obs, ld_l, ld_h, ldnd_l, ldnd_h) in results:
        ld[i, j] = ld[j, i] = ld_obs
        ldnd[i, j] = ldnd[j, i] = ldnd_obs
        ld_lo[i, j] = ld_lo[j, i] = ld_l
        ld_hi[i, j] = ld_hi[j, i] = ld_h
        ldnd_lo[i, j] = ldnd_lo[j, i] = ldnd_l
        ldnd_hi[i, j] = ldnd_hi[j, i] = ldnd_h

    return {"ld": ld, "ld_lo": ld_lo, "ld_hi": ld_hi,
            "ldnd": ldnd, "ldnd_lo": ldnd_lo, "ldnd_hi": ldnd_hi}


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #

def _first_form(cell: str) -> str:
    """Take the first comma-separated form, strip slashes."""
    if not cell:
        return ""
    primary = cell.split(",")[0].split("/")[0].strip()
    return primary


def load_wordlist(path: Path) -> Dict[str, List[str]]:
    forms: Dict[str, List[str]] = {v: [] for v in VARIETIES}
    concepts = []
    with path.open(encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        for row in rd:
            concepts.append(row["concept_en"].strip())
            for v in VARIETIES:
                forms[v].append(_first_form(row.get(v, "")))
    return forms, concepts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wordlist", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="Parallel workers over (i,j) pairs (default 1).")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    forms_orth, concepts = load_wordlist(args.wordlist)

    # Convert to ASJPcode
    forms_asjp = {v: [_to_asjp(w, v) if w else "" for w in forms_orth[v]]
                  for v in VARIETIES}

    # Audit dump
    audit = args.out_dir / "lexicostat_asjpcode.csv"
    with audit.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["concept_en"]
                    + [f"{v}_orth" for v in VARIETIES]
                    + [f"{v}_asjp" for v in VARIETIES])
        for k, c in enumerate(concepts):
            row = [c]
            row += [forms_orth[v][k] for v in VARIETIES]
            row += [forms_asjp[v][k] for v in VARIETIES]
            wr.writerow(row)

    # Compute LD + LDND with bootstrap
    out = compute_matrices(forms_asjp, VARIETIES,
                           bootstrap=args.bootstrap, n_jobs=args.n_jobs)

    common_meta = {
        "source": "Wiktionary Swadesh-207 + gendered split",
        "n_concepts": len(concepts),
        "concepts": concepts,
        "n_bootstrap": args.bootstrap,
        "varieties": VARIETIES,
        "warnings": [
            "Multiple-form cells: only the first form is used for distance.",
            "Gendered concepts (he/she, this, that, they) split into _m/_f rows.",
            "ASJPcode mapping is heuristic; replace with epitran for finer phonetics.",
        ],
    }

    # 1. Mean normalised Levenshtein (the user's primary request)
    out_ld = args.out_dir / "lexicostat_lev_mean.npz"
    np.savez(out_ld,
             matrix=out["ld"], matrix_lo=out["ld_lo"], matrix_hi=out["ld_hi"],
             labels=np.array(VARIETIES, dtype=object),
             meta=np.array([json.dumps({**common_meta,
                                         "metric": "mean normalised Levenshtein"})],
                            dtype=object))

    # 2. LDND (chance-baseline normalised)
    out_ldnd = args.out_dir / "lexicostat_ldnd.npz"
    np.savez(out_ldnd,
             matrix=out["ldnd"], matrix_lo=out["ldnd_lo"], matrix_hi=out["ldnd_hi"],
             labels=np.array(VARIETIES, dtype=object),
             meta=np.array([json.dumps({**common_meta,
                                         "metric": "LDND (LD / chance baseline)"})],
                            dtype=object))

    # 3. Human-readable CSV exports of the two matrices.  We dump three CSVs
    #    per metric: the matrix itself, the bootstrap lower bound, and the
    #    upper bound.  For paper Table N you can grab the first one and
    #    annotate cells with [lo, hi] from the other two.
    import pandas as pd
    for label, key, fname in [
        ("LD",   "ld",   "lexicostat_lev_mean.csv"),
        ("LDND", "ldnd", "lexicostat_ldnd.csv"),
    ]:
        mat = out[key]
        lo = out[f"{key}_lo"]
        hi = out[f"{key}_hi"]
        # 1) plain mean
        pd.DataFrame(mat, index=VARIETIES, columns=VARIETIES).to_csv(
            args.out_dir / fname, float_format="%.4f")
        # 2) lower bound + upper bound (only if bootstrap was run)
        if np.isfinite(lo).any():
            pd.DataFrame(lo, index=VARIETIES, columns=VARIETIES).to_csv(
                args.out_dir / fname.replace(".csv", "_ci_lo.csv"), float_format="%.4f")
            pd.DataFrame(hi, index=VARIETIES, columns=VARIETIES).to_csv(
                args.out_dir / fname.replace(".csv", "_ci_hi.csv"), float_format="%.4f")
            # 3) compact "mean [lo, hi]" cells in one CSV — most readable for paper
            combined = np.empty(mat.shape, dtype=object)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    if i == j:
                        combined[i, j] = "—"
                    elif np.isfinite(lo[i, j]):
                        combined[i, j] = f"{mat[i, j]:.3f} [{lo[i, j]:.3f},{hi[i, j]:.3f}]"
                    else:
                        combined[i, j] = f"{mat[i, j]:.3f}"
            pd.DataFrame(combined, index=VARIETIES, columns=VARIETIES).to_csv(
                args.out_dir / fname.replace(".csv", "_with_ci.csv"))

    finite_ld = out["ld"][np.isfinite(out["ld"]) & (out["ld"] > 0)]
    finite_ldnd = out["ldnd"][np.isfinite(out["ldnd"]) & (out["ldnd"] > 0)]
    rng_ld = (float(finite_ld.min()), float(finite_ld.max())) if finite_ld.size else (np.nan, np.nan)
    rng_ldnd = (float(finite_ldnd.min()), float(finite_ldnd.max())) if finite_ldnd.size else (np.nan, np.nan)
    print(f"  lex_mean (LD)    range=[{rng_ld[0]:.3f}, {rng_ld[1]:.3f}]  → {out_ld.name}")
    print(f"  lex_ldnd         range=[{rng_ldnd[0]:.3f}, {rng_ldnd[1]:.3f}]  → {out_ldnd.name}")
    print(f"  audit dump       → {audit.name}")

    # Pretty print: full 13x13 matrix of mean Levenshtein
    print("\n--- Mean normalised Levenshtein (LD) — 13x13 matrix ---")
    print("       " + "  ".join(f"{v:>5}" for v in VARIETIES))
    for i, va in enumerate(VARIETIES):
        cells = [f"{out['ld'][i, j]:>5.3f}" if i != j else "  -  "
                 for j in range(len(VARIETIES))]
        print(f"  {va:<4} " + "  ".join(cells))

    # Distance to Italian with CI
    print("\nDistance to Italian — both metrics + 95% bootstrap CI:")
    if "ita" in VARIETIES:
        i_ita = VARIETIES.index("ita")
        print(f"  {'lang':<5}  {'LD':>6}  {'LD_CI95':<22}  {'LDND':>6}  {'LDND_CI95':<22}")
        for v in VARIETIES:
            i = VARIETIES.index(v)
            if i == i_ita:
                continue
            ld_v = out["ld"][i, i_ita]
            ldnd_v = out["ldnd"][i, i_ita]
            ld_lo = out["ld_lo"][i, i_ita]
            ld_hi = out["ld_hi"][i, i_ita]
            ldnd_lo = out["ldnd_lo"][i, i_ita]
            ldnd_hi = out["ldnd_hi"][i, i_ita]
            ld_ci = f"[{ld_lo:.3f}, {ld_hi:.3f}]" if np.isfinite(ld_lo) else "n/a"
            ldnd_ci = f"[{ldnd_lo:.3f}, {ldnd_hi:.3f}]" if np.isfinite(ldnd_lo) else "n/a"
            print(f"  {v:<5}  {ld_v:>6.3f}  {ld_ci:<22}  {ldnd_v:>6.3f}  {ldnd_ci:<22}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
