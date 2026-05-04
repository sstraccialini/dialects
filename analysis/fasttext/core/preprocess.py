"""
Text preprocessing for FastText / BPE pipelines.

Both pipelines use `preprocess_for_subword`:
- lowercase + (optional) mask numbers
- KEEP punctuation (apostrophes are informative for dialects)
- preserve diacritics

Note: the source CSVs are already aggressive-normalized at extraction
time, so most flags are idempotent.
"""
from __future__ import annotations

import re
import string
from typing import Iterable, List

from .config import LOWERCASE, MASK_NUMBERS, NUMBER_TOKEN, STRIP_PUNCT_FOR_WORD

_NUMBER_RE = re.compile(r"\d+")
_WORD_PUNCT = string.punctuation + "«»‘’“”–—…"
_WORD_PUNCT_RE = re.compile(f"[{re.escape(_WORD_PUNCT)}]")
_WS_RE = re.compile(r"\s+")


def _basic_clean(text: str) -> str:
    if LOWERCASE:
        text = text.lower()
    if MASK_NUMBERS:
        text = _NUMBER_RE.sub(NUMBER_TOKEN, text)
    return text


def preprocess_for_word(text: str) -> str:
    text = _basic_clean(text)
    if STRIP_PUNCT_FOR_WORD:
        text = _WORD_PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def preprocess_for_char(text: str) -> str:
    text = _basic_clean(text)
    return _WS_RE.sub(" ", text).strip()


preprocess_for_subword = preprocess_for_char


def batch(preprocess_fn, texts: Iterable[str]) -> List[str]:
    return [preprocess_fn(t) for t in texts]
