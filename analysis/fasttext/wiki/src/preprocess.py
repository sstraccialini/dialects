"""
Text preprocessing shared by both pipelines.

Identical logic to the TF-IDF baseline; kept self-contained here.

Both the FastText and BPE pipelines use `preprocess_for_subword`:
- lowercase + mask numbers
- KEEP punctuation (apostrophes and diacritics are informative for
  dialects AND important for subword unit boundaries)
- Do NOT normalize unicode (distinctive diacritics must be preserved)

The word-pipeline variant (strip punctuation) is included for
completeness but is not used by Person 5's pipelines.
"""

from __future__ import annotations

import re
import string
from typing import Iterable, List

from config import LOWERCASE, MASK_NUMBERS, NUMBER_TOKEN, STRIP_PUNCT_FOR_WORD

_NUMBER_RE = re.compile(r"\d+")
_WORD_PUNCT = string.punctuation + "\u00ab\u00bb\u2018\u2019\u201c\u201d\u2013\u2014\u2026"
_WORD_PUNCT_RE = re.compile(f"[{re.escape(_WORD_PUNCT)}]")
_WS_RE = re.compile(r"\s+")


def _basic_clean(text: str) -> str:
    if LOWERCASE:
        text = text.lower()
    if MASK_NUMBERS:
        text = _NUMBER_RE.sub(NUMBER_TOKEN, text)
    return text


def preprocess_for_word(text: str) -> str:
    """Preprocessing for word-level analysis (strips punctuation)."""
    text = _basic_clean(text)
    if STRIP_PUNCT_FOR_WORD:
        text = _WORD_PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def preprocess_for_char(text: str) -> str:
    """
    Preprocessing for char-level / subword analysis.

    Keeps punctuation and diacritics — both are informative for
    dialect-specific orthography and for subword tokenization.
    """
    text = _basic_clean(text)
    return _WS_RE.sub(" ", text).strip()


# Alias used by both FastText and BPE pipelines.
preprocess_for_subword = preprocess_for_char


def batch(preprocess_fn, texts: Iterable[str]) -> List[str]:
    return [preprocess_fn(t) for t in texts]
