"""
Text preprocessing for the FastText / BPE pipelines.

Same philosophy as Person 1's TF-IDF baseline:
    - lowercase
    - mask digits with ' NUM '
    - keep diacritics (they are distinctive traits)

We expose two variants:

- preprocess_for_subword(text) -> str
  Used before FastText training and before passing to SentencePiece.
  Keeps punctuation (apostrophes + hyphens are informative for subword
  n-grams and for BPE merges). Collapses whitespace.

- preprocess_for_word(text) -> str
  Alternative where punctuation is stripped. Not used by the main
  subword pipeline, but kept for symmetry with the TF-IDF baseline in
  case we want to run BPE on already-word-tokenised inputs.

Tokenisation for FastText:
- tokenize_for_fasttext(text) -> List[str]
  We split on whitespace after preprocessing. FastText uses whole
  "words" as the base units and enriches them with char n-grams, so
  keeping words with their punctuation (e.g. "l'università") gives
  FastText a richer subword signal than forcing a hard split on the
  apostrophe.
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


# Pre-compile regexes for speed.
_NUMBER_RE = re.compile(r"\d+")

_WORD_PUNCT = (
    string.punctuation
    + "\u00ab\u00bb"   # French/Italian guillemets
    + "\u2018\u2019"   # curly single quotes
    + "\u201c\u201d"   # curly double quotes
    + "\u2013\u2014"   # en/em dashes
    + "\u2026"         # ellipsis
)
_WORD_PUNCT_RE = re.compile(f"[{re.escape(_WORD_PUNCT)}]")

_WS_RE = re.compile(r"\s+")


def _basic_clean(text: str) -> str:
    """Shared steps: lowercase + mask digits."""
    if LOWERCASE:
        text = text.lower()
    if MASK_NUMBERS:
        text = _NUMBER_RE.sub(NUMBER_TOKEN, text)
    return text


def preprocess_for_subword(text: str) -> str:
    """
    Preprocessing for FastText and BPE training.

    Keeps punctuation. Only cleans whitespace and numbers + lowercase.
    """
    text = _basic_clean(text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def preprocess_for_word(text: str) -> str:
    """Alternative with punctuation stripped (parity with TF-IDF word pipeline)."""
    text = _basic_clean(text)
    if STRIP_PUNCT_FOR_WORD:
        text = _WORD_PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def tokenize_for_fasttext(text: str) -> List[str]:
    """
    Whitespace tokenisation after subword-friendly preprocessing.

    FastText treats each token as a whole word and augments it with
    character n-grams, so we keep token boundaries minimal.
    """
    cleaned = preprocess_for_subword(text)
    if not cleaned:
        return []
    return cleaned.split(" ")


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
    print("--- SUBWORD preprocessing (kept punctuation) ---")
    for s in samples:
        print(f"  in : {s!r}")
        print(f"  out: {preprocess_for_subword(s)!r}")
        print(f"  tok: {tokenize_for_fasttext(s)}")
        print()
    print("--- WORD preprocessing (stripped punctuation) ---")
    for s in samples:
        print(f"  in : {s!r}")
        print(f"  out: {preprocess_for_word(s)!r}")
        print()
