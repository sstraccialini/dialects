"""
Load per-variety CSV files and sub-sample sentences.

CSVs in `wiki_data/{code}.csv` have columns:
    text, label, article_id
where `text` is already a cleaned, filtered sentence (len>20 chars)
produced by `data/generation.py`. This module only reads and samples.

The sentence pipeline needs `article_id` to aggregate sentence
embeddings inside each Wikipedia article before averaging across the
whole variety, so this data_loader returns pandas DataFrames (with
the `article_id` column preserved) in addition to the plain list-of-
sentences form used by the TF-IDF baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from config import (
    DATASETS_DIR,
    VARIETY_CODES,
    SAMPLE_SIZE,
    RANDOM_STATE,
)


def _csv_path(code: str) -> Path:
    """Return the CSV path for a given variety code."""
    return DATASETS_DIR / f"{code}.csv"


def load_variety_sentences(
    code: str,
    sample_size: int = SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
) -> List[str]:
    """
    Load and sub-sample sentences for a single variety.

    Returns a list of strings. If the variety has fewer than
    `sample_size` sentences, returns all of them (no replacement).
    """
    path = _csv_path(code)
    if not path.exists():
        raise FileNotFoundError(f"CSV for '{code}' not found: {path}")

    df = pd.read_csv(path, usecols=["text"]).dropna(subset=["text"]).reset_index(drop=True)
    n = len(df)
    if n == 0:
        raise ValueError(f"No sentences in {path}")

    if n <= sample_size:
        return df["text"].tolist()
    return df.sample(n=sample_size, random_state=random_state)["text"].tolist()


def load_all_varieties(
    codes: List[str] = VARIETY_CODES,
    sample_size: int = SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """
    Load and sample all varieties (flat list-of-sentences form).

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


def load_all_varieties_with_article_ids(
    codes: List[str] = VARIETY_CODES,
    sample_size: int = SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Load and sample all varieties preserving `article_id`.

    Returns:
        data:  dict {code: DataFrame(text, article_id)} of sampled rows.
        stats: DataFrame with columns [code, n_available, n_used,
               n_articles] for reporting and diagnostics.

    The DataFrame form is required by `fit_transform_sentence_by_article`
    in `sentence_vectorize.py`, which groups sentence embeddings by
    `article_id` before averaging.
    """
    data: Dict[str, pd.DataFrame] = {}
    rows = []

    for code in codes:
        path = _csv_path(code)
        if not path.exists():
            raise FileNotFoundError(f"CSV for '{code}' not found: {path}")

        df = (
            pd.read_csv(path, usecols=["text", "article_id"])
              .dropna(subset=["text"])
              .reset_index(drop=True)
        )
        n_available = len(df)
        if n_available == 0:
            raise ValueError(f"No sentences in {path}")

        if n_available <= sample_size:
            sampled = df
        else:
            sampled = df.sample(
                n=sample_size, random_state=random_state
            ).reset_index(drop=True)

        data[code] = sampled
        rows.append({
            "code": code,
            "n_available": n_available,
            "n_used": len(sampled),
            "n_articles": int(sampled["article_id"].nunique()),
        })

        if verbose:
            print(f"  [{code:>4}] available={n_available:>7d}  "
                  f"used={len(sampled):>7d}  articles={rows[-1]['n_articles']:>5d}")

    stats = pd.DataFrame(rows)
    return data, stats


if __name__ == "__main__":
    # Smoke test: print load statistics.
    print("Loading all varieties (article-aware form)...")
    data, stats = load_all_varieties_with_article_ids()
    print("\nStats:")
    print(stats.to_string(index=False))
