"""
Text preprocessing for Word2Vec.

- lowercase (config.LOWERCASE)
- mask digit runs with ' NUM ' (config.MASK_NUMBERS)
- keep diacritics (distinctive for Romance)
- tokeniser: Unicode letter runs joined across apostrophes, e.g.:
      "l'università"  -> ["l'università"]
      "c'è, dai!"     -> ["c'è", "dai"]
  This is critical for the dialects (apostrophes are frequent and
  informative) and avoids the baseline defect where "c'è" was split
  into "c" + "è".

The number token is kept as a whole word ("NUM" after stripping the
surrounding spaces and joining); we produce it via `_normalize` before
tokenisation so it survives as a single token.
"""

from __future__ import annotations

import re
from typing import Iterable, List

from config import (
    LOWERCASE,
    MASK_NUMBERS,
    NUMBER_TOKEN,
)

# Digit runs -> " NUM " (spaces so NUM stays its own token)
_NUMBER_RE = re.compile(r"\d+")

# A "word" = Unicode letters, optionally chained via straight or
# typographic apostrophes. We match letters (\p{L}-ish via [^\W\d_])
# and allow arbitrary segments joined by ' or ’.
_TOKEN_RE = re.compile(r"[^\W\d_]+(?:[’'][^\W\d_]+)*", re.UNICODE)

_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Lowercase + mask numbers + collapse whitespace."""
    if LOWERCASE:
        text = text.lower()
    if MASK_NUMBERS:
        text = _NUMBER_RE.sub(NUMBER_TOKEN, text)
    return _WS_RE.sub(" ", text).strip()


def tokenize(text: str) -> List[str]:
    """Return the tokens of a sentence (empty list if nothing usable)."""
    normalised = _normalize(text)
    if not normalised:
        return []
    return _TOKEN_RE.findall(normalised)


def tokenize_many(texts: Iterable[str]) -> List[List[str]]:
    """Tokenise a batch; drops sentences that produce zero tokens."""
    out: List[List[str]] = []
    for t in texts:
        toks = tokenize(t)
        if toks:
            out.append(toks)
    return out


if __name__ == "__main__":
    samples = [
        "Buenos Aires è 'a capitale e 'a cità cchiù grossa 'e ll'Argentina.",
        "Joseph Maurice Ravel (1875-1937) fu nu cumpusituri francisi.",
        "Cette page présente la saison 1963-1964 de l'AS Saint-Étienne.",
        "Η σέλα της Ιππικής δεξιοτεχνίας είναι φτιαγμένη.",
    ]
    for s in samples:
        print(f"{s!r}\n  -> {tokenize(s)}\n")
