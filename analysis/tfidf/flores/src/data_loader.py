"""
Load FLORES+ per-variety .txt files and (optionally) sub-sample.

Each file `flores_data/flores_plus/<slug>.txt` contains 2009 sentences
(one per line): 997 dev + 1012 devtest concatenated in order. Lines
are already cleaned (no newlines, no CR) by the download script.

Key choices:
- Sub-sampling: if a variety has fewer than SAMPLE_SIZE sentences we
  take all of them. With FLORES+ the upper bound is 2009 and that is
  also the default SAMPLE_SIZE, so this effectively loads everything
  unless the user reduces SAMPLE_SIZE for sensitivity experiments.
- We return both a DataFrame (handy for exploration) and a dict
  {slug: list[str]} (used downstream for vectorization).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from config import (
    FLORES_DIR,
    VARIETY_CODES,
    SAMPLE_SIZE,
    RANDOM_STATE,
)


def _txt_path(slug: str) -> Path:
    """Return the .txt path for a given variety slug."""
    return FLORES_DIR / f"{slug}.txt"


def _read_sentences(path: Path) -> List[str]:
    """Read a FLORES+ .txt file as a list of stripped sentences."""
    if not path.exists():
        raise FileNotFoundError(f"FLORES+ file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        sents = [line.strip() for line in f]
    # drop fully-empty lines (should not happen with our download script
    # but stay defensive)
    sents = [s for s in sents if s]
    if not sents:
        raise ValueError(f"No sentences in {path}")
    return sents


def load_variety_sentences(
    slug: str,
    sample_size: int = SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
) -> List[str]:
    """
    Load (and optionally sub-sample) sentences for a single variety.

    Returns a list of strings. If the variety has fewer than
    `sample_size` sentences, returns all of them (no replacement).
    """
    sents = _read_sentences(_txt_path(slug))
    if len(sents) <= sample_size:
        return sents
    # deterministic sub-sampling via pandas for consistency with the
    # original baseline (same seeded behaviour across approaches).
    s = pd.Series(sents).sample(n=sample_size, random_state=random_state)
    return s.tolist()


def load_all_varieties(
    codes: List[str] = VARIETY_CODES,
    sample_size: int = SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """
    Load and (optionally) sample all varieties.

    Returns:
        data:  dict {slug: list[str]} of sampled sentences.
        stats: DataFrame with columns [code, n_available, n_used],
               useful for reporting and diagnostics.
    """
    data: Dict[str, List[str]] = {}
    rows = []

    for slug in codes:
        sents = _read_sentences(_txt_path(slug))
        n_available = len(sents)
        if n_available <= sample_size:
            sampled = sents
        else:
            sampled = pd.Series(sents).sample(
                n=sample_size, random_state=random_state
            ).tolist()

        data[slug] = sampled
        rows.append({
            "code": slug,
            "n_available": n_available,
            "n_used": len(sampled),
        })

        if verbose:
            print(f"  [{slug:>10}] available={n_available:>6d}  used={len(sampled):>6d}")

    stats = pd.DataFrame(rows)
    return data, stats


def to_variety_document(sentences: List[str], joiner: str = " ") -> str:
    """
    Concatenate all sentences of a variety into one super-document.

    Using a plain space joiner preserves word boundaries (important
    for the char_wb analyzer in char n-grams).
    """
    return joiner.join(sentences)


def build_variety_documents(
    data: Dict[str, List[str]],
    codes: List[str] = VARIETY_CODES,
) -> Tuple[List[str], List[str]]:
    """
    Convert the dict {slug: sentences} to (documents, codes_ordered).

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
        print(f"  [{c:>10}] len={len(d):>8d}  preview={d[:80]!r}...")
