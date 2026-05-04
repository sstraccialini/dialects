"""
Text preprocessing before TF-IDF.

Two pipelines:

- word pipeline: optional lowercase + optional mask numbers + optional strip punctuation.
- char pipeline: optional lowercase + optional mask numbers. KEEPS punctuation
  (apostrophes are informative for dialects: "'a ", "l'e'", "c'e'"...).

In both cases:
- We do NOT normalize unicode at this stage: diacritics and accents are
  distinctive traits and must be preserved (NB: the source Wiki/FLORES/OLDI
  CSVs are already aggressive-normalized at extraction time, so the input
  here is `[a-z\\s]+` — these flags are mostly idempotent now).
- Mask numbers: any digit sequence becomes `NUMBER_TOKEN` (with surrounding
  spaces so we don't create spurious char n-grams).
"""
from __future__ import annotations

import re
import string
from typing import Iterable, List

from .config import (
    LOWERCASE,
    MASK_NUMBERS,
    NUMBER_TOKEN,
    STRIP_PUNCT_FOR_WORD,
)


_NUMBER_RE = re.compile(r"\d+")
_WORD_PUNCT = string.punctuation + "«»‘’“”–—…"
_WORD_PUNCT_RE = re.compile(f"[{re.escape(_WORD_PUNCT)}]")
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
