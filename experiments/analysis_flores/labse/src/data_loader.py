"""
Load FLORES+ per-variety .txt files.

Each file `flores_data/flores_plus/<slug>.txt` contains 2009 sentences
(one per line): 997 dev + 1012 devtest concatenated in order. Lines
are already cleaned (no newlines, no CR) by the download script.

Key choices:
- FLORES+ has no article/document structure, so we return a flat list
  of sentences per variety. Sentence parallelism across varieties is
  itself a strong anchor: position i in every file is the translation
  of the same source sentence.
- Sub-sampling: if a variety has fewer than SAMPLE_SIZE sentences we
  take all of them. Default SAMPLE_SIZE is 2009 (= full set).
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
    return FLORES_DIR / f"{slug}.txt"


def _read_sentences(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"FLORES+ file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        sents = [line.strip() for line in f]
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
    return (
        pd.Series(sents)
          .sample(n=sample_size, random_state=random_state)
          .tolist()
    )


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
        stats: DataFrame with columns [code, n_available, n_used].
    """
    data: Dict[str, List[str]] = {}
    rows = []

    for slug in codes:
        sents = _read_sentences(_txt_path(slug))
        n_available = len(sents)
        if n_available <= sample_size:
            sampled = sents
        else:
            sampled = (
                pd.Series(sents)
                  .sample(n=sample_size, random_state=random_state)
                  .tolist()
            )

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


if __name__ == "__main__":
    # Smoke test: print load statistics.
    print("Loading all varieties (smoke test)...")
    data, stats = load_all_varieties()
    print("\nStats:")
    print(stats.to_string(index=False))
