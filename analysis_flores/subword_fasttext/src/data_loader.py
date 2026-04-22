"""
Load FLORES+ per-variety .txt files for the FastText / BPE pipeline.

Same contract as the TF-IDF loader: each file
`flores_data/flores_plus/<slug>.txt` contains 2009 sentences (one per
line). With FLORES+ the default SAMPLE_SIZE (2009) just returns
everything.

We expose both:
    - load_all_varieties() -> ({slug: list[str]}, stats DataFrame)
    - to_variety_document() / build_variety_documents() helpers used
      by the BPE+TF-IDF sub-pipeline.

The FastText pipeline consumes the *sentences* directly (it trains on
tokenised sentences, not super-documents), while the BPE+TF-IDF
pipeline aggregates sentences per variety to mirror Person 1's TF-IDF
baseline exactly.
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
    """Read a FLORES+ .txt file as a list of stripped, non-empty sentences."""
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
    return pd.Series(sents).sample(
        n=sample_size, random_state=random_state
    ).tolist()


def load_all_varieties(
    codes: List[str] = VARIETY_CODES,
    sample_size: int = SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """
    Load + (optional) sub-sample all varieties.

    Returns:
        data:  {slug: list[str]} of sampled sentences.
        stats: DataFrame [code, n_available, n_used].
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
    """Concatenate all sentences of a variety into a single super-document."""
    return joiner.join(sentences)


def build_variety_documents(
    data: Dict[str, List[str]],
    codes: List[str] = VARIETY_CODES,
) -> Tuple[List[str], List[str]]:
    """
    Convert {slug: sentences} to (documents, codes_ordered) aligned lists.

    Used by the BPE + TF-IDF sub-pipeline (one super-document per variety).
    """
    codes_ordered = [c for c in codes if c in data]
    documents = [to_variety_document(data[c]) for c in codes_ordered]
    return documents, codes_ordered


if __name__ == "__main__":
    print("Loading all varieties (smoke test)...")
    data, stats = load_all_varieties()
    print("\nStats:")
    print(stats.to_string(index=False))
    total_sents = sum(len(v) for v in data.values())
    print(f"\nTotal sentences across {len(data)} varieties: {total_sents}")
