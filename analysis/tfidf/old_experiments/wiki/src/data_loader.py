"""
Load per-variety CSV files and sub-sample sentences.

CSVs live under `Dataset/wiki/{dialects_in_both_OLDI_and_Flores,languages}/{code}.csv`
with columns: text, label, article_id. The text is already in
aggressive-normalized form (lowercase ASCII letters + spaces only),
produced by `Dataset/wiki/scripts/generation.py`.

Sub-sampling:
- if a variety has fewer than SAMPLE_SIZE sentences, take all of them
- otherwise random-sample without replacement using RANDOM_STATE
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

from config import (
    VARIETY_CODES,
    VARIETY_DIR,
    SAMPLE_SIZE,
    RANDOM_STATE,
)


def _csv_path(code: str) -> Path:
    """Return the CSV path for a given variety code, looking in the
    correct subfolder under Dataset/wiki/ via VARIETY_DIR mapping.
    """
    if code not in VARIETY_DIR:
        raise ValueError(f"Unknown variety code {code!r}. "
                         f"Expected one of {sorted(VARIETY_DIR)}.")
    return VARIETY_DIR[code] / f"{code}.csv"


def load_variety_sentences(
    code: str,
    sample_size: int = SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
) -> List[str]:
    """
    Load and sub-sample sentences for a single variety.

    Returns a list of strings. If the variety has fewer than `sample_size`
    sentences, returns all of them (no replacement).
    """
    path = _csv_path(code)
    if not path.exists():
        raise FileNotFoundError(f"CSV for '{code}' not found: {path}")

    df = pd.read_csv(path, usecols=["text"])
    df = df.dropna(subset=["text"]).reset_index(drop=True)

    n = len(df)
    if n == 0:
        raise ValueError(f"No sentences in {path}")

    if n <= sample_size:
        sampled = df["text"].tolist()
    else:
        sampled = df.sample(n=sample_size, random_state=random_state)["text"].tolist()

    return sampled


def load_all_varieties(
    codes: List[str] = VARIETY_CODES,
    sample_size: int = SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """
    Load and sample all varieties.

    Returns:
        data:  dict {code: list[str]} of sampled sentences.
        stats: DataFrame with columns [code, n_available, n_used], useful
               for reporting and diagnostics.
    """
    data: Dict[str, List[str]] = {}
    rows = []

    for code in codes:
        path = _csv_path(code)
        df = pd.read_csv(path, usecols=["text"]).dropna(subset=["text"])
        n_available = len(df)

        if n_available <= sample_size:
            sampled = df["text"].tolist()
        else:
            sampled = df.sample(n=sample_size, random_state=random_state)["text"].tolist()

        data[code] = sampled
        rows.append({
            "code": code,
            "n_available": n_available,
            "n_used": len(sampled),
        })

        if verbose:
            print(f"  [{code:>4}] available={n_available:>7d}  used={len(sampled):>7d}")

    stats = pd.DataFrame(rows)
    return data, stats


def to_variety_document(sentences: List[str], joiner: str = " ") -> str:
    """
    Concatenate all sentences of a variety into one 'super-document'.

    This is the form required by the "one document per variety"
    aggregation. Using a plain space joiner preserves word boundaries
    (important for the char_wb analyzer in char n-grams).
    """
    return joiner.join(sentences)


def build_variety_documents(
    data: Dict[str, List[str]],
    codes: List[str] = VARIETY_CODES,
) -> Tuple[List[str], List[str]]:
    """
    Convert the dict {code: sentences} to (documents, codes_ordered).

    Returns two aligned lists:
        documents[i]    = super-document of variety codes_ordered[i]
        codes_ordered   = variety order (uses VARIETY_CODES to stay
                          consistent with the rest of the pipeline).
    """
    codes_ordered = [c for c in codes if c in data]
    documents = [to_variety_document(data[c]) for c in codes_ordered]
    return documents, codes_ordered


if __name__ == "__main__":
    # Smoke test: print load statistics.
    print("Loading all varieties (smoke test)...")
    data, stats = load_all_varieties()
    print("\nStats:")
    print(stats.to_string(index=False))
    docs, codes = build_variety_documents(data)
    print(f"\n{len(docs)} super-documents built.")
    for c, d in zip(codes, docs):
        print(f"  [{c:>4}] len={len(d):>10d}  preview={d[:80]!r}...")
