"""
Unified data loaders for the FINAL experiments (2026-05-10).

Three data sources, all return `dict[variety_code, list[str]]` so downstream
code is uniform:

  load_flores(text_variant=..., codes=None)
      Parallel evaluation set (1827 sentences × 17 varieties).

  load_oldi(text_variant=..., codes=None)
      Parallel ita↔dialect (5167 sentences × 7 varieties: ita + 6 dialects).

  load_wiki(text_variant=..., codes=None, sample_size, random_state)
      Per-variety Wiki sentences (sub-sampled, ~100k/variety).

Plus the cap-rule combiner used by FastText/Word2Vec/XLM-R/CANINE training:

  load_wiki_plus_oldi_dialect(text_variant, codes, sample_size, random_state)
      For each DIALECT: ALL OLDI dialect sentences first, then Wiki up to sample_size.
      For each NON-DIALECT: Wiki only, capped at sample_size.

Conventions:
  - text_variant='native'      → text with diacritics/case/punct/digits/non-ASCII
  - text_variant='normalized'  → aggressive_normalize applied (lowercase ASCII + spaces)
  - The slug used inside FLORES/OLDI CSVs is the Italian-name from FLORES_SLUG
    (italiano, friulano, ligure, ...).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .varieties import (
    VARIETY_CODES,
    DIALECT_CODES,
    FLORES_SLUG,
    FLORES_CLEANED_CSV, FLORES_CLEANED_NORM_CSV,
    OLDI_CLEANED_CSV,   OLDI_CLEANED_NORM_CSV,
    WIKI_VARIETY_DIR, WIKI_VARIETY_DIR_NATIVE,
    SAMPLE_SIZE, RANDOM_STATE,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _flores_csv_path(text_variant: str) -> Path:
    if text_variant == "native":
        return FLORES_CLEANED_CSV
    if text_variant == "normalized":
        return FLORES_CLEANED_NORM_CSV
    raise ValueError(f"text_variant must be 'native' or 'normalized', got {text_variant!r}")


def _oldi_csv_path(text_variant: str) -> Path:
    if text_variant == "native":
        return OLDI_CLEANED_CSV
    if text_variant == "normalized":
        return OLDI_CLEANED_NORM_CSV
    raise ValueError(f"text_variant must be 'native' or 'normalized', got {text_variant!r}")


def _wiki_csv_path(code: str, text_variant: str) -> Path:
    if text_variant == "native":
        d = WIKI_VARIETY_DIR_NATIVE.get(code)
    elif text_variant == "normalized":
        d = WIKI_VARIETY_DIR.get(code)
    else:
        raise ValueError(f"text_variant must be 'native' or 'normalized', got {text_variant!r}")
    if d is None:
        raise KeyError(f"No Wiki dir registered for code {code!r}.")
    return d / f"{code}.csv"


# --------------------------------------------------------------------------- #
# FLORES (parallel evaluation set)
# --------------------------------------------------------------------------- #
def load_flores(
    text_variant: str = "native",
    codes: Optional[List[str]] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """Load FLORES cleaned. All varieties have the SAME N (parallel).

    Returns:
        data:  {code: list[str]} — every list has the same length.
        stats: DataFrame [code, n_used].
    """
    path = _flores_csv_path(text_variant)
    df = pd.read_csv(path)
    if codes is None:
        codes = [c for c in VARIETY_CODES if FLORES_SLUG.get(c) in df.columns]

    data: Dict[str, List[str]] = {}
    rows = []
    for code in codes:
        slug = FLORES_SLUG.get(code)
        if slug is None or slug not in df.columns:
            if verbose:
                print(f"  [flores {code:>4}] not in FLORES — skipping")
            continue
        sents = df[slug].astype(str).tolist()
        data[code] = sents
        rows.append({"code": code, "n_used": len(sents)})
        if verbose:
            print(f"  [flores {code:>4}] {len(sents):>5d} sentences ({text_variant})")

    lengths = {c: len(s) for c, s in data.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"FLORES parallel requires equal lengths, got {lengths}")
    return data, pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# OLDI (parallel ita↔dialect)
# --------------------------------------------------------------------------- #
def load_oldi(
    text_variant: str = "native",
    codes: Optional[List[str]] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """Load OLDI cleaned. Available codes: ita + 6 italo-romance dialects."""
    path = _oldi_csv_path(text_variant)
    df = pd.read_csv(path)
    if codes is None:
        codes = [c for c in VARIETY_CODES if FLORES_SLUG.get(c) in df.columns]

    data: Dict[str, List[str]] = {}
    rows = []
    for code in codes:
        slug = FLORES_SLUG.get(code)
        if slug is None or slug not in df.columns:
            if verbose:
                print(f"  [oldi   {code:>4}] not in OLDI — skipping")
            continue
        sents = df[slug].astype(str).tolist()
        data[code] = sents
        rows.append({"code": code, "n_used": len(sents)})
        if verbose:
            print(f"  [oldi   {code:>4}] {len(sents):>5d} sentences ({text_variant})")

    lengths = {c: len(s) for c, s in data.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"OLDI parallel requires equal lengths, got {lengths}")
    return data, pd.DataFrame(rows)


def load_oldi_pairs(
    text_variant: str = "native",
    dialects: Optional[List[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Return a long DataFrame of (ita, dialect_text, dialect_code) triples.

    Used by LaBSE MNRL fine-tuning (Cell 10).

    Returns: DataFrame with columns ['ita', 'dial', 'dial_code'].
    """
    if dialects is None:
        dialects = list(DIALECT_CODES)
    path = _oldi_csv_path(text_variant)
    df = pd.read_csv(path)
    if "italiano" not in df.columns:
        raise KeyError("Expected 'italiano' column in OLDI cleaned CSV.")

    pairs = []
    for code in dialects:
        slug = FLORES_SLUG.get(code)
        if slug is None or slug not in df.columns:
            if verbose:
                print(f"  [pair  {code:>4}] missing in OLDI — skipping")
            continue
        sub = df[["italiano", slug]].copy()
        sub.columns = ["ita", "dial"]
        sub["dial_code"] = code
        pairs.append(sub)
        if verbose:
            print(f"  [pair  {code:>4}] {len(sub):>5d} ita↔{code} pairs ({text_variant})")
    if not pairs:
        return pd.DataFrame(columns=["ita", "dial", "dial_code"])
    return pd.concat(pairs, ignore_index=True)


# --------------------------------------------------------------------------- #
# Wiki (per-variety flat CSVs)
# --------------------------------------------------------------------------- #
def load_wiki(
    text_variant: str = "native",
    codes: Optional[List[str]] = None,
    sample_size: int = SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """Load + sub-sample Wiki sentences per variety.

    If a variety has fewer than `sample_size` sentences, take all.
    Otherwise random-sample with `random_state`.
    """
    if codes is None:
        codes = list(VARIETY_CODES)

    data: Dict[str, List[str]] = {}
    rows = []
    for code in codes:
        path = _wiki_csv_path(code, text_variant)
        if not path.exists():
            if verbose:
                print(f"  [wiki   {code:>4}] WARNING: {path} not found — skipping")
            continue
        df = pd.read_csv(path, usecols=["text"]).dropna(subset=["text"])
        n_avail = len(df)
        if n_avail <= sample_size:
            sampled = df["text"].astype(str).tolist()
        else:
            sampled = df.sample(n=sample_size, random_state=random_state)["text"].astype(str).tolist()
        data[code] = sampled
        rows.append({"code": code, "n_available": n_avail, "n_used": len(sampled)})
        if verbose:
            print(f"  [wiki   {code:>4}] available={n_avail:>7d}  used={len(sampled):>7d}  ({text_variant})")

    return data, pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Cap-rule combiner: Wiki + OLDI for dialects (FastText/Word2Vec/XLM-R/CANINE)
# --------------------------------------------------------------------------- #
def load_wiki_plus_oldi_dialect(
    text_variant: str = "native",
    codes: Optional[List[str]] = None,
    sample_size: int = SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """Wiki+OLDI cap rule (EXPERIMENTAL_PLAN.md §3.4).

    For DIALECTS:
        1. Take ALL OLDI dialect sentences.
        2. Add Wiki sentences (random-sampled with `random_state`) up to total `sample_size`.
        3. If OLDI alone exceeds `sample_size`, keep ALL OLDI (no cap).

    For NON-DIALECTS (standards):
        Wiki only, capped at `sample_size`. NO OLDI.
    """
    if codes is None:
        codes = list(VARIETY_CODES)

    # Pre-load OLDI once for all dialects
    oldi_data, _ = load_oldi(text_variant=text_variant, codes=DIALECT_CODES, verbose=False)

    data: Dict[str, List[str]] = {}
    rows = []
    for code in codes:
        is_dial = code in DIALECT_CODES
        if is_dial and code in oldi_data:
            oldi_sents = oldi_data[code]
            n_oldi = len(oldi_sents)

            wiki_path = _wiki_csv_path(code, text_variant)
            if wiki_path.exists():
                wiki_df = pd.read_csv(wiki_path, usecols=["text"]).dropna(subset=["text"])
                n_wiki_avail = len(wiki_df)
            else:
                wiki_df = pd.DataFrame(columns=["text"])
                n_wiki_avail = 0

            if n_oldi >= sample_size:
                # OLDI alone exceeds cap → keep ALL OLDI, no Wiki
                sents = list(oldi_sents)
                n_wiki_used = 0
            else:
                # Fill with Wiki up to sample_size total
                budget = sample_size - n_oldi
                if n_wiki_avail <= budget:
                    wiki_sents = wiki_df["text"].astype(str).tolist()
                else:
                    wiki_sents = wiki_df.sample(n=budget, random_state=random_state)["text"].astype(str).tolist()
                sents = list(oldi_sents) + wiki_sents
                n_wiki_used = len(wiki_sents)

            data[code] = sents
            rows.append({
                "code": code, "is_dialect": True,
                "n_oldi": n_oldi, "n_wiki_avail": n_wiki_avail,
                "n_wiki_used": n_wiki_used, "n_total": len(sents),
            })
            if verbose:
                print(f"  [W+O   {code:>4}] dialect: OLDI={n_oldi}  Wiki={n_wiki_used}  total={len(sents)}  ({text_variant})")
        else:
            # Standard or dialect-without-OLDI: Wiki only
            wiki_path = _wiki_csv_path(code, text_variant)
            if not wiki_path.exists():
                if verbose:
                    print(f"  [W+O   {code:>4}] WARNING: Wiki path missing — skipping")
                continue
            wiki_df = pd.read_csv(wiki_path, usecols=["text"]).dropna(subset=["text"])
            n_wiki_avail = len(wiki_df)
            if n_wiki_avail <= sample_size:
                sents = wiki_df["text"].astype(str).tolist()
            else:
                sents = wiki_df.sample(n=sample_size, random_state=random_state)["text"].astype(str).tolist()
            data[code] = sents
            rows.append({
                "code": code, "is_dialect": False,
                "n_oldi": 0, "n_wiki_avail": n_wiki_avail,
                "n_wiki_used": len(sents), "n_total": len(sents),
            })
            if verbose:
                print(f"  [W+O   {code:>4}] standard: Wiki={len(sents)}  total={len(sents)}  ({text_variant})")

    return data, pd.DataFrame(rows)
