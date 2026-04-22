"""
Text preprocessing before TF-IDF.

Two pipelines:

- word pipeline: lowercase + mask numbers + strip punctuation.
- char pipeline: lowercase + mask numbers. KEEPS punctuation
  (apostrophes are informative for dialects: "'a ", "l'e'", "c'e'" ...).

In both cases:
- We do NOT normalize unicode: diacritics and accents are distinctive
  traits and must be preserved.
- Mask numbers: any digit sequence becomes ' NUM ' (spaces on both
  sides so we don't create spurious char n-grams).
"""

from __future__ import annotations

import re
import string
from typing import Iterable, List

from config import (
    LOWERCASE,
    MASK_NUMBERS,
    NUMBER_TOKEN,
    STRIP_PUNCT_FOR_WORD,
)


# Pre-compile regexes for speed on large inputs.
_NUMBER_RE = re.compile(r"\d+")

# For the word pipeline we strip standard Latin punctuation plus a few
# typical typographic characters. We replace with space (never delete)
# to avoid merging adjacent words.
_WORD_PUNCT = (
    string.punctuation
    + "\u00ab\u00bb"   # French/Italian guillemets
    + "\u2018\u2019"   # curly single quotes
    + "\u201c\u201d"   # curly double quotes
    + "\u2013\u2014"   # en-dash, em-dash
    + "\u2026"         # ellipsis
)
_WORD_PUNCT_RE = re.compile(f"[{re.escape(_WORD_PUNCT)}]")

# Collapse multiple whitespace into a single space.
_WS_RE = re.compile(r"\s+")


def _basic_clean(text: str) -> str:
    """Steps shared by both pipelines: lowercase + mask numbers."""
    if LOWERCASE:
        text = text.lower()
    if MASK_NUMBERS:
        text = _NUMBER_RE.sub(NUMBER_TOKEN, text)
    return text


def preprocess_for_word(text: str) -> str:
    """Preprocessing for the WORD n-gram pipeline."""
    text = _basic_clean(text)
    if STRIP_PUNCT_FOR_WORD:
        text = _WORD_PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def preprocess_for_char(text: str) -> str:
    """Preprocessing for the CHAR n-gram pipeline (no punctuation stripping)."""
    text = _basic_clean(text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def batch(preprocess_fn, texts: Iterable[str]) -> List[str]:
    """Apply a preprocessing function to an iterable of strings."""
    return [preprocess_fn(t) for t in texts]


if __name__ == "__main__":
    samples = [
        "Buenos Aires è 'a capitale e 'a cità cchiù grossa 'e ll'Argentina.",
        "Joseph Maurice Ravel (1875-1937) fu nu cumpusituri francisi.",
        "Cette page présente la saison 1963-1964 de l'AS Saint-Étienne.",
        "Η σέλα της Ιππικής δεξιοτεχνίας είναι φτιαγμένη.",
    ]
    print("--- WORD preprocessing ---")
    for s in samples:
        print(f"  in : {s!r}")
        print(f"  out: {preprocess_for_word(s)!r}")
        print()
    print("--- CHAR preprocessing ---")
    for s in samples:
        print(f"  in : {s!r}")
        print(f"  out: {preprocess_for_char(s)!r}")
        print()
