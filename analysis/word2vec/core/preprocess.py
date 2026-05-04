"""
Tokenizer for Word2Vec.

- lowercase + collapse whitespace (numbers already stripped at extraction).
- token regex keeps Unicode letter runs, allowing apostrophe-chained tokens
  ("l'università", "c'è") which are common in Italo-Romance dialects and
  must not be split.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

from .config import LOWERCASE, MASK_NUMBERS, NUMBER_TOKEN


_NUMBER_RE = re.compile(r"\d+")
_TOKEN_RE = re.compile(r"[^\W\d_]+(?:[’'][^\W\d_]+)*", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    if LOWERCASE:
        text = text.lower()
    if MASK_NUMBERS:
        text = _NUMBER_RE.sub(NUMBER_TOKEN, text)
    return _WS_RE.sub(" ", text).strip()


def tokenize(text: str) -> List[str]:
    normalised = _normalize(text)
    if not normalised:
        return []
    return _TOKEN_RE.findall(normalised)


def tokenize_many(texts: Iterable[str]) -> List[List[str]]:
    out: List[List[str]] = []
    for t in texts:
        toks = tokenize(t)
        if toks:
            out.append(toks)
    return out


def build_tokenised_corpus(
    data: Dict[str, List[str]],
    codes: List[str] = None,
) -> Tuple[List[List[str]], List[str]]:
    """
    Flatten {code: sentences} → (tokenised_sents, sentence_codes) aligned.
    Sentences with zero tokens are dropped.
    """
    from .config import VARIETY_CODES
    if codes is None:
        codes = VARIETY_CODES

    tokenised: List[List[str]] = []
    sentence_codes: List[str] = []
    for code in codes:
        if code not in data:
            continue
        for s in data[code]:
            toks = tokenize(s)
            if toks:
                tokenised.append(toks)
                sentence_codes.append(code)
    return tokenised, sentence_codes
