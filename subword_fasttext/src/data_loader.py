"""
Load per-variety CSV files and sub-sample sentences.

Identical in logic to the TF-IDF baseline data_loader; kept as a
separate file so each approach is self-contained and runnable without
touching the baseline directory.

CSVs live in `datasets/{code}.csv` with columns: text, label, article_id.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

from config import (
    DATASETS_DIR,
    VARIETY_CODES,
    SAMPLE_SIZE,
    RANDOM_STATE,
)


def _csv_path(code: str) -> Path:
    return DATASETS_DIR / f"{code}.csv"


def load_variety_sentences(
    code: str,
    sample_size: int = SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
) -> List[str]:
    path = _csv_path(code)
    if not path.exists():
        raise FileNotFoundError(f"CSV for '{code}' not found: {path}")
    df = pd.read_csv(path, usecols=["text"]).dropna(subset=["text"]).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"No sentences in {path}")
    if len(df) <= sample_size:
        return df["text"].tolist()
    return df.sample(n=sample_size, random_state=random_state)["text"].tolist()


def load_all_varieties(
    codes: List[str] = VARIETY_CODES,
    sample_size: int = SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """
    Load and sample all varieties.

    Returns:
        data:  {code: list[str]} of sampled sentences
        stats: DataFrame with columns [code, n_available, n_used]
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
        rows.append({"code": code, "n_available": n_available, "n_used": len(sampled)})

        if verbose:
            print(f"  [{code:>4}] available={n_available:>7d}  used={len(sampled):>7d}")

    return data, pd.DataFrame(rows)


def to_variety_document(sentences: List[str], joiner: str = " ") -> str:
    """Concatenate sentences into one super-document."""
    return joiner.join(sentences)


def build_variety_documents(
    data: Dict[str, List[str]],
    codes: List[str] = VARIETY_CODES,
) -> Tuple[List[str], List[str]]:
    """
    Convert {code: sentences} → (list_of_super_docs, codes_ordered).
    Used by the BPE pipeline (TF-IDF on BPE pieces needs one doc per variety).
    """
    docs = [to_variety_document(data[c]) for c in codes]
    return docs, list(codes)
