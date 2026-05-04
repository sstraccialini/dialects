"""
Apply aggressive normalization to FLORES+ files.

Reads originals from `Dataset/flores/not_normalized/` and writes
normalized versions to `Dataset/flores/normalized/`.

For parallel.tsv: every language column is normalized individually
(except metadata columns `sentence_id` and `split`).

Aggressive normalize keeps in sync with `Dataset/wiki/scripts/generation.py`
(same function definition copy-pasted; if you edit one, edit the other).

Run with: python Dataset/flores/scripts/normalize.py
"""
import re
import unicodedata
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]   # Dataset/flores/
SRC_DIR = ROOT / "not_normalized"
DST_DIR = ROOT / "normalized"

META_COLS = {"sentence_id", "split"}


# --------------------------------------------------------------------------- #
# Aggressive normalize (copy of Dataset/wiki/scripts/generation.py).
# --------------------------------------------------------------------------- #
_EXPLICIT_MAP = str.maketrans({
    "ß": "ss", "ł": "l", "Ł": "L",
    "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE",
    "ø": "o", "Ø": "O",
    "đ": "d", "Đ": "D",
    "ð": "d", "Ð": "D",
    "þ": "th", "Þ": "TH",
})
_DIACRITICS = re.compile(r"[̀-ͯ]")
_DIGITS = re.compile(r"\d+")
_ROMAN_UPPER = re.compile(r"\b[IVXLCDM]{2,}\b")
_NON_LATIN_LOWER = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")


def aggressive_normalize(text):
    """Lowercase ASCII letters + spaces only. See generation.py for details."""
    if not isinstance(text, str):
        return text
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_EXPLICIT_MAP)
    text = _ROMAN_UPPER.sub(" ", text)
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = _DIACRITICS.sub("", text)
    text = _DIGITS.sub(" ", text)
    text = _NON_LATIN_LOWER.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


# --------------------------------------------------------------------------- #
# Main: normalize every .txt and parallel.tsv.
# --------------------------------------------------------------------------- #
def main():
    if not SRC_DIR.is_dir():
        raise SystemExit(f"Source dir not found: {SRC_DIR}")
    DST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading from: {SRC_DIR}")
    print(f"Writing to:   {DST_DIR}")
    print()

    # Per-language .txt files (one sentence per line).
    for src in sorted(SRC_DIR.glob("*.txt")):
        lines = src.read_text(encoding="utf-8").splitlines()
        normalized = [aggressive_normalize(line) for line in lines]
        dst = DST_DIR / src.name
        dst.write_text("\n".join(normalized) + "\n", encoding="utf-8")
        print(f"[ok]    {src.name:20s}  {len(normalized):>5,} lines")

    # parallel.tsv: per-column normalize (skip only metadata columns).
    for src in sorted(SRC_DIR.glob("*.tsv")):
        df = pd.read_csv(src, sep="\t")
        for col in df.columns:
            if col in META_COLS:
                continue
            df[col] = df[col].apply(aggressive_normalize)
        dst = DST_DIR / src.name
        df.to_csv(dst, sep="\t", index=False)
        non_meta = [c for c in df.columns if c not in META_COLS]
        print(f"[ok]    {src.name:20s}  {len(df):>5,} rows × {len(non_meta)} normalized cols")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
