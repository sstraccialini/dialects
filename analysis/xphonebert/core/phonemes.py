"""
Phoneme inventories for the 9 standard languages.

Each list is the IPA inventory of the standard variety, distilled from
PHOIBLE/Wikipedia/standard reference grammars. We keep affricates as
single multi-char units (e.g. "tʃ", "dʒ") because that's how XPhoneBERT
expects phonemes to be tokenised.

For the 6 dialects we don't hard-code an inventory — we extract it
empirically from Manzini-Savoia (see `data_loader.dialect_phonemes`).

Notes:
- We don't include vowel-length contrasts as separate phonemes when
  they're predictable (e.g. Italian gemination, French no-length).
- Diphthongs are written as two-symbol clusters (e.g. "aɪ̯") for the
  languages where they're distinctive (English, German).
- Allophones (e.g. Spanish [β ð ɣ] for /b d g/) are listed separately
  when they're commonly transcribed as such in IPA databases.
"""
from __future__ import annotations

from typing import Dict, List


PHONEME_INVENTORY: Dict[str, List[str]] = {
    # --------------------------------------------------------------------- #
    # Italian — 30 phonemes (standard varieties; geminates not listed
    # separately, vowel length predictable).
    # --------------------------------------------------------------------- #
    "ita": [
        # vowels
        "a", "e", "ɛ", "i", "o", "ɔ", "u",
        # consonants
        "p", "b", "t", "d", "k", "g",
        "f", "v", "s", "z", "ʃ",
        "m", "n", "ɲ",
        "l", "ʎ", "r",
        # affricates
        "ts", "dz", "tʃ", "dʒ",
        # glides
        "j", "w",
    ],

    # --------------------------------------------------------------------- #
    # Spanish — 24 phonemes (Castilian + common allophones)
    # --------------------------------------------------------------------- #
    "spa": [
        "a", "e", "i", "o", "u",
        "p", "b", "t", "d", "k", "g",
        "f", "θ", "s", "x",
        "m", "n", "ɲ",
        "l", "ʎ", "ɾ", "r",
        "tʃ",
        "j", "w",
        "β", "ð", "ɣ",  # allophones of /b d g/
    ],

    # --------------------------------------------------------------------- #
    # French — 37 phonemes (standard Parisian)
    # --------------------------------------------------------------------- #
    "fra": [
        # oral vowels
        "a", "ɑ", "e", "ɛ", "i", "o", "ɔ", "u",
        "y", "ø", "œ", "ə",
        # nasal vowels
        "ɑ̃", "ɛ̃", "ɔ̃", "œ̃",
        # consonants
        "p", "b", "t", "d", "k", "g",
        "f", "v", "s", "z", "ʃ", "ʒ",
        "m", "n", "ɲ", "ŋ",
        "l", "ʁ",
        # glides
        "j", "ɥ", "w",
    ],

    # --------------------------------------------------------------------- #
    # Catalan — 31 phonemes (Central / standard)
    # --------------------------------------------------------------------- #
    "cat": [
        "a", "e", "ɛ", "i", "o", "ɔ", "u", "ə",
        "p", "b", "t", "d", "k", "g",
        "f", "v", "s", "z", "ʃ", "ʒ",
        "m", "n", "ɲ", "ŋ",
        "l", "ʎ", "ɾ", "r",
        "tʃ", "dʒ",
        "j", "w",
    ],

    # --------------------------------------------------------------------- #
    # German — 41 phonemes (Standard German, length contrasts kept)
    # --------------------------------------------------------------------- #
    "deu": [
        # short vowels
        "a", "ɛ", "ɪ", "ɔ", "ʊ", "ʏ", "œ", "ə", "ɐ",
        # long / tense vowels
        "aː", "eː", "iː", "oː", "uː", "yː", "øː", "ɛː",
        # diphthongs
        "aɪ̯", "aʊ̯", "ɔʏ̯",
        # consonants
        "p", "b", "t", "d", "k", "g",
        "pf", "ts", "tʃ",
        "f", "v", "s", "z", "ʃ", "ʒ", "ç", "x", "h",
        "m", "n", "ŋ",
        "l", "ʁ",
        # glides
        "j",
    ],

    # --------------------------------------------------------------------- #
    # Slovenian — 29 phonemes
    # --------------------------------------------------------------------- #
    "slv": [
        "a", "ɛ", "e", "i", "ɔ", "o", "u", "ə",
        "p", "b", "t", "d", "k", "g",
        "f", "v", "s", "z", "ʃ", "ʒ", "x",
        "m", "n",
        "l", "ɾ", "r",
        "ts", "tʃ", "dʒ",
        "j",
    ],

    # --------------------------------------------------------------------- #
    # English — 44 phonemes (RP / General American mix)
    # --------------------------------------------------------------------- #
    "eng": [
        # vowels
        "æ", "ɑ", "ɛ", "ɪ", "i", "ɒ", "ɔ", "ʊ", "u", "ʌ", "ə", "ɝ",
        # diphthongs
        "aɪ", "aʊ", "eɪ", "oʊ", "ɔɪ",
        # consonants
        "p", "b", "t", "d", "k", "g",
        "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h",
        "m", "n", "ŋ",
        "l", "ɹ",
        # affricates
        "tʃ", "dʒ",
        # glides
        "j", "w",
    ],

    # --------------------------------------------------------------------- #
    # Greek (Modern) — 29 phonemes
    # --------------------------------------------------------------------- #
    "ell": [
        "a", "e", "i", "o", "u",
        "p", "b", "t", "d", "k", "g",
        "f", "v", "θ", "ð", "s", "z", "x", "ɣ",
        "m", "n", "ɲ", "ŋ",
        "l", "ʎ", "ɾ",
        "ts", "dz",
        "j",
    ],

    # --------------------------------------------------------------------- #
    # Arabic (Modern Standard) — 34 phonemes
    # Emphatics marked with the IPA superscript ˤ.
    # --------------------------------------------------------------------- #
    "arb": [
        # vowels (3 short, 3 long)
        "a", "i", "u", "aː", "iː", "uː",
        # plain consonants
        "b", "t", "d", "k", "q", "ʔ",
        "f", "θ", "ð", "s", "z", "ʃ",
        "x", "ɣ", "ħ", "ʕ", "h",
        # emphatics
        "tˤ", "dˤ", "sˤ", "ðˤ",
        # liquids / nasals / glides
        "m", "n", "l", "r",
        # affricate
        "dʒ",
        # glides
        "j", "w",
    ],
}


def get_inventory(code: str) -> List[str]:
    if code not in PHONEME_INVENTORY:
        raise KeyError(
            f"No hard-coded phoneme inventory for {code!r}. "
            f"Standard codes available: {sorted(PHONEME_INVENTORY)}"
        )
    return list(PHONEME_INVENTORY[code])
