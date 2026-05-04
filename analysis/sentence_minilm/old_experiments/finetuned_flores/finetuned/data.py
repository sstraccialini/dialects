"""
Fine-tuning–specific data loaders for MiniLM.

  - load_wiki_texts_for_tsdae   flat list of Wiki sentences for TSDAE
  - load_oldi_pairs / load_all_oldi_pairs   Italian ↔ dialect pairs for MNRL

Generic 3-corpus loaders live in `..core.data_loader`.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd

from ..core.config import (
    OLDI_DIR,
    WIKI_VARIETY_DIR,
    RANDOM_STATE,
)
from .config import (
    OLDI_PAIR_DIALECTS,
    OLDI_PAIR_SLUG,
    MAX_WIKI_SAMPLES,
)


def _wiki_csv_path(code: str) -> Path:
    if code not in WIKI_VARIETY_DIR:
        raise ValueError(f"Unknown variety code {code!r}.")
    return WIKI_VARIETY_DIR[code] / f"{code}.csv"


def load_wiki_texts_for_tsdae(
    codes: List[str] = None,
    max_per_lang: int = MAX_WIKI_SAMPLES,
    random_state: int = RANDOM_STATE,
    min_length: int = 20,
    verbose: bool = True,
) -> List[str]:
    if codes is None:
        codes = OLDI_PAIR_DIALECTS + ["ita"]
    out: List[str] = []
    for code in codes:
        path = _wiki_csv_path(code)
        if not path.exists():
            print(f"  [wiki] WARNING: {path} not found — skipping {code}")
            continue
        df = pd.read_csv(path, usecols=["text"], dtype=str, on_bad_lines="skip")
        texts = [t.strip() for t in df["text"].dropna() if len(str(t).strip()) >= min_length]
        if len(texts) > max_per_lang:
            texts = pd.Series(texts).sample(n=max_per_lang, random_state=random_state).tolist()
        if verbose:
            print(f"  [wiki  {code:>4}]  {len(texts):>6d} sentences")
        out.extend(texts)
    return out


def load_oldi_pairs(code: str) -> List[Tuple[str, str]]:
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
