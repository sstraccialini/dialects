"""
Data loaders for XPhoneBERT phoneme experiments.

- For the 6 Italo-Romance dialects: read Manzini-Savoia IPA transcriptions
  and extract their empirical phoneme inventory (multi-character IPA
  units kept atomic via greedy longest-match tokenisation).
- For the 9 standard languages: hard-coded inventories from `phonemes.py`.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .config import (
    DIALECT_CODES,
    MS_DIR, MS_REGION_FILE,
)
from .phonemes import PHONEME_INVENTORY, get_inventory


# --------------------------------------------------------------------------- #
# IPA tokeniser
# --------------------------------------------------------------------------- #
# A small set of "diacritics that bind to the previous phoneme" — combining
# marks attached to the right of a base IPA letter form a single phoneme.
_IPA_BINDING_RIGHT = set([
    "ː",   # long
    "ˑ",   # half-long
    "ˤ",   # pharyngealised (Arabic emphatics)
    "ʰ",   # aspirated
    "ʷ",   # labialised
    "ʲ",   # palatalised
    "ˠ",   # velarised
    "˞",   # rhotacised
    "̯",   # non-syllabic (combining)
    "̃",   # nasalised (combining)
    "̬",   # voiced (combining)
    "̥",   # voiceless (combining)
    "̩",   # syllabic (combining)
])

# Two-character base phonemes that should be kept atomic (affricates, common
# digraphs). XPhoneBERT's tokenizer accepts these as single units.
_IPA_DIGRAPHS = [
    "tʃ", "dʒ", "ts", "dz", "tɕ", "dʑ",
    "pf", "kʷ", "ɡʷ",
    "aʊ", "aɪ", "eɪ", "oʊ", "ɔɪ",  # English diphthongs (no combining mark)
    "aɪ̯", "aʊ̯", "ɔʏ̯",            # German diphthongs (with non-syllabic mark)
    "ɑ̃", "ɛ̃", "ɔ̃", "œ̃",         # French nasal vowels
    "tˤ", "dˤ", "sˤ", "ðˤ",          # Arabic emphatics
]
# Sort by length desc so greedy match prefers longer digraphs first.
_IPA_DIGRAPHS.sort(key=len, reverse=True)


def tokenize_ipa(s: str) -> List[str]:
    """Greedy longest-match phoneme tokenisation of an IPA string.

    Whitespace and the IPA primary-stress mark "ˈ" / secondary "ˌ" /
    syllable separator "." / phrase break "|" are dropped. Combining
    diacritics attach to the *previous* base phoneme.
    """
    if not s:
        return []
    out: List[str] = []
    i = 0
    n = len(s)
    skip_chars = {" ", "\t", "\n", "ˈ", "ˌ", ".", "|", "•", "…", "/"}
    while i < n:
        c = s[i]
        if c in skip_chars:
            i += 1
            continue
        # Try multi-char digraphs first
        matched = None
        for d in _IPA_DIGRAPHS:
            if s.startswith(d, i):
                matched = d
                break
        if matched:
            out.append(matched)
            i += len(matched)
            continue
        # Single base char + any binding diacritics
        ph = c
        i += 1
        while i < n and s[i] in _IPA_BINDING_RIGHT:
            ph += s[i]
            i += 1
        out.append(ph)
    return out


# --------------------------------------------------------------------------- #
# Dialect data (Manzini-Savoia)
# --------------------------------------------------------------------------- #
def load_ms_region(code: str) -> pd.DataFrame:
    """Load the raw Manzini-Savoia CSV for one dialect code."""
    if code not in MS_REGION_FILE:
        raise KeyError(f"No MS region file for code {code!r}")
    path = Path(MS_DIR) / MS_REGION_FILE[code]
    if not path.exists():
        raise FileNotFoundError(f"MS file not found: {path}")
    df = pd.read_csv(path, dtype=str).fillna("")
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    return df


def dialect_phonemes(code: str, *, min_count: int = 2) -> Tuple[List[str], Counter]:
    """Empirical phoneme inventory for a dialect, extracted from MS.

    Returns (sorted unique phonemes, full counter).
    `min_count` filters out singletons (typos, errors, ultra-rare).
    """
    df = load_ms_region(code)
    counter: Counter = Counter()
    for line in df["text"].tolist():
        for p in tokenize_ipa(line):
            counter[p] += 1
    phonemes = sorted([p for p, n in counter.items() if n >= min_count])
    return phonemes, counter


def dialect_phonemes_all(*, min_count: int = 2) -> Dict[str, List[str]]:
    """Same as dialect_phonemes but for all 6 dialects, as a dict."""
    return {c: dialect_phonemes(c, min_count=min_count)[0] for c in DIALECT_CODES}


# --------------------------------------------------------------------------- #
# Standard-language data
# --------------------------------------------------------------------------- #
def standard_phonemes(code: str) -> List[str]:
    """Hard-coded phoneme inventory for one standard language."""
    return get_inventory(code)


def all_phoneme_inventories(*, min_count: int = 2) -> Dict[str, List[str]]:
    """Return phoneme inventory for every variety in the 15-set."""
    out: Dict[str, List[str]] = {}
    for code in DIALECT_CODES:
        out[code] = dialect_phonemes(code, min_count=min_count)[0]
    for code in PHONEME_INVENTORY:
        out[code] = list(PHONEME_INVENTORY[code])
    return out
