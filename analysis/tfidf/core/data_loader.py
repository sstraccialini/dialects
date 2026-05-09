"""
Unified data loader for TF-IDF experiments — three corpora.

Each loader returns a `dict {variety_code: list[str]}` so downstream
code is uniform regardless of source.

  load_wiki_for_training(sample_size=...) — reads `Dataset/wiki/<group>/<code>.csv`
                                            and sub-samples to SAMPLE_SIZE.
  load_flores_parallel()                  — reads `Dataset/flores/normalized/<slug>.txt`
                                            (2,009 sentences/variety, fully parallel).
  load_oldi_parallel()                    — reads `Dataset/oldi/normalized/<code>_Latn.parquet`
                                            (6,193 sentences/variety, fully parallel).

Plus helpers:
  build_variety_documents(data)           — concatenate sentences per variety
                                            (used by `vectorize.fit_transform_*`)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .config import (
    VARIETY_CODES,
    WIKI_VARIETY_DIR,
    FLORES_DIR,
    FLORES_SLUG,
    OLDI_DIR,
    OLDI_PARQUET,
    SAMPLE_SIZE,
    RANDOM_STATE,
)


# --------------------------------------------------------------------------- #
# WIKI loader (sub-sampled training data)
# --------------------------------------------------------------------------- #
def _wiki_csv_path(code: str) -> Path:
    if code not in WIKI_VARIETY_DIR:
        raise ValueError(f"Unknown variety code {code!r}.")
    return WIKI_VARIETY_DIR[code] / f"{code}.csv"


def load_wiki_for_training(
    codes: List[str] = None,
    sample_size: int = SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """Load + sub-sample Wiki sentences per variety.

    If a variety has fewer than `sample_size` sentences, take all.
    Otherwise random-sample with `random_state`.

    Returns:
        data:  {code: list[str]}
        stats: DataFrame [code, n_available, n_used]
    """
    if codes is None:
        codes = VARIETY_CODES

    data: Dict[str, List[str]] = {}
    rows = []

    for code in codes:
        path = _wiki_csv_path(code)
        df = pd.read_csv(path, usecols=["text"]).dropna(subset=["text"])
        n_available = len(df)

        if n_available <= sample_size:
            sampled = df["text"].tolist()
        else:
            sampled = df.sample(n=sample_size, random_state=random_state)["text"].tolist()

        data[code] = sampled
        rows.append({"code": code, "n_available": n_available, "n_used": len(sampled)})

        if verbose:
            print(f"  [wiki   {code:>4}] available={n_available:>7d}  used={len(sampled):>7d}")

    return data, pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# FLORES loader (parallel evaluation data)
# --------------------------------------------------------------------------- #
def _flores_txt_path(code: str) -> Path:
    if code not in FLORES_SLUG:
        raise ValueError(f"Unknown FLORES code {code!r}.")
    return FLORES_DIR / f"{FLORES_SLUG[code]}.txt"


def load_flores_parallel(
    codes: List[str] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """Load FLORES+ parallel sentences (2,009 per variety, aligned by index).

    Returns:
        data:  {code: list[str]} — all dicts have IDENTICAL N (2,009).
        stats: DataFrame [code, n_available, n_used]
    """
    if codes is None:
        codes = VARIETY_CODES

    data: Dict[str, List[str]] = {}
    rows = []

    for code in codes:
        path = _flores_txt_path(code)
        with path.open("r", encoding="utf-8") as f:
            sents = [line.strip() for line in f if line.strip()]
        data[code] = sents
        rows.append({"code": code, "n_available": len(sents), "n_used": len(sents)})

        if verbose:
            print(f"  [flores {code:>4}] {len(sents):>5d} sentences")

    # Sanity check: all varieties must have the same N for parallel analyses.
    lengths = {c: len(s) for c, s in data.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"FLORES parallel requires equal lengths, got {lengths}")

    return data, pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# OLDI loader (parallel evaluation data, 6,193 sentences/variety)
# --------------------------------------------------------------------------- #
def _oldi_parquet_path(code: str) -> Path:
    if code not in OLDI_PARQUET:
        raise ValueError(f"Unknown OLDI code {code!r}.")
    return OLDI_DIR / f"{OLDI_PARQUET[code]}.parquet"


def load_oldi_parallel(
    codes: List[str] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """Load OLDI parallel sentences (6,193 per variety, aligned by `id` column).

    Returns:
        data:  {code: list[str]} — all sorted by `id`, all aligned.
        stats: DataFrame [code, n_available, n_used]
    """
    if codes is None:
        codes = VARIETY_CODES

    data: Dict[str, List[str]] = {}
    rows = []

    for code in codes:
        if code not in OLDI_PARQUET:
            if verbose:
                print(f"  [oldi   {code:>4}] not in OLDI dataset — skipping")
            continue
        path = _oldi_parquet_path(code)
        if not path.exists():
            print(f"  [oldi   {code:>4}] WARNING: {path} not found — skipping")
            continue
        df = pd.read_parquet(path)
        # Sort by id to guarantee alignment across varieties.
        df = df.sort_values("id").reset_index(drop=True)
        sents = df["text"].fillna("").astype(str).tolist()
        data[code] = sents
        rows.append({"code": code, "n_available": len(sents), "n_used": len(sents)})

        if verbose:
            print(f"  [oldi   {code:>4}] {len(sents):>5d} sentences")

    # Sanity check
    lengths = {c: len(s) for c, s in data.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"OLDI parallel requires equal lengths, got {lengths}")

    return data, pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Shared helper — build "one super-document per variety"
# --------------------------------------------------------------------------- #
def to_variety_document(sentences: List[str], joiner: str = " ") -> str:
    """Concatenate all sentences of a variety into one super-document.

    Plain space joiner preserves word boundaries (important for `char_wb`).
    """
    return joiner.join(sentences)


def build_variety_documents(
    data: Dict[str, List[str]],
    codes: List[str] = None,
) -> Tuple[List[str], List[str]]:
    """Return (documents, codes_ordered) aligned with `codes` if provided
    (defaults to `VARIETY_CODES`).
    """
    if codes is None:
        codes = VARIETY_CODES
    codes_ordered = [c for c in codes if c in data]
    documents = [to_variety_document(data[c]) for c in codes_ordered]
    return documents, codes_ordered
