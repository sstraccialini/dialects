"""Data loaders for CANINE — three corpora, identical contract to other methods."""
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
    OLDI_PAIR_DIALECTS,
    OLDI_PAIR_SLUG,
    SAMPLE_SIZE,
    RANDOM_STATE,
)


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


def _flores_txt_path(code: str) -> Path:
    if code not in FLORES_SLUG:
        raise ValueError(f"Unknown FLORES code {code!r}.")
    return FLORES_DIR / f"{FLORES_SLUG[code]}.txt"


def load_flores_parallel(
    codes: List[str] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
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
    lengths = {c: len(s) for c, s in data.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"FLORES parallel requires equal lengths, got {lengths}")
    return data, pd.DataFrame(rows)


def _oldi_parquet_path(code: str) -> Path:
    if code not in OLDI_PARQUET:
        raise ValueError(f"Unknown OLDI code {code!r}.")
    return OLDI_DIR / f"{OLDI_PARQUET[code]}.parquet"


def load_oldi_parallel(
    codes: List[str] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    if codes is None:
        codes = VARIETY_CODES
    data: Dict[str, List[str]] = {}
    rows = []
    for code in codes:
        path = _oldi_parquet_path(code)
        if not path.exists():
            print(f"  [oldi   {code:>4}] WARNING: {path} not found — skipping")
            continue
        df = pd.read_parquet(path)
        df = df.sort_values("id").reset_index(drop=True)
        sents = df["text"].fillna("").astype(str).tolist()
        data[code] = sents
        rows.append({"code": code, "n_available": len(sents), "n_used": len(sents)})
        if verbose:
            print(f"  [oldi   {code:>4}] {len(sents):>5d} sentences")
    lengths = {c: len(s) for c, s in data.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"OLDI parallel requires equal lengths, got {lengths}")
    return data, pd.DataFrame(rows)


def iter_labeled_sentences(
    data: Dict[str, List[str]],
    codes: List[str] = None,
) -> Tuple[List[str], List[str]]:
    if codes is None:
        codes = VARIETY_CODES
    sents: List[str] = []
    out_codes: List[str] = []
    for code in codes:
        if code not in data:
            continue
        for s in data[code]:
            sents.append(s)
            out_codes.append(code)
    return sents, out_codes


# --------------------------------------------------------------------------- #
# OLDI parallel pairs Italian↔dialect — for sentence-pair MLM training
# --------------------------------------------------------------------------- #
def load_oldi_pairs(code: str) -> List[Tuple[str, str]]:
    """Load (italian, dialect) parallel pairs for one Italo-Romance dialect."""
    if code not in OLDI_PAIR_SLUG:
        raise ValueError(f"No OLDI Italian-pair file for code {code!r}")
    slug = OLDI_PAIR_SLUG[code]
    tsv_path = OLDI_DIR / f"pairs_ita_{slug}.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(f"OLDI pairs file not found: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    pairs = []
    for _, row in df.iterrows():
        ita = str(row.get("italiano", "")).strip()
        dial = str(row.get(slug, "")).strip()
        if ita and dial and ita != "nan" and dial != "nan":
            pairs.append((ita, dial))
    return pairs


def load_all_oldi_pairs(
    codes: List[str] = None,
    verbose: bool = True,
) -> List[Tuple[str, str]]:
    """Concatenate OLDI Italian↔dialect pairs across all 6 dialects."""
    if codes is None:
        codes = OLDI_PAIR_DIALECTS
    out: List[Tuple[str, str]] = []
    for code in codes:
        try:
            pairs = load_oldi_pairs(code)
        except FileNotFoundError as e:
            if verbose:
                print(f"  [oldi pairs] WARNING: {e} — skipping {code}")
            continue
        if verbose:
            print(f"  [oldi pairs {code:>4}] {len(pairs):>6d} pairs")
        out.extend(pairs)
    return out
