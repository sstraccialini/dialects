"""
Apply aggressive normalization to OLDI parquet + pairs TSV files.

Reads originals from `Dataset/oldi/not_normalized/` and writes
normalized versions to `Dataset/oldi/normalized/`. The parquet files
keep their original schema with the `text` column replaced by its
normalized version.

Aggressive normalize keeps in sync with `Dataset/wiki/scripts/generation.py`
(same function definition copy-pasted; if you edit one, edit the other).

Run with: python Dataset/oldi/scripts/normalize.py
"""
import re
import unicodedata
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]   # Dataset/oldi/
SRC_DIR = ROOT / "not_normalized"
DST_DIR = ROOT / "normalized"

# Don't normalize these column names (they're metadata, not text).
META_COLS = {"id", "iso_639_3", "iso_15924", "glottocode", "url", "last_updated",
             "sentence_id", "split"}


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
# Main: normalize every parquet and TSV.
# --------------------------------------------------------------------------- #
def main():
    if not SRC_DIR.is_dir():
        raise SystemExit(f"Source dir not found: {SRC_DIR}")
    DST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading from: {SRC_DIR}")
    print(f"Writing to:   {DST_DIR}")
    print()

    # Per-language parquet (schema: id, iso_639_3, iso_15924, glottocode,
    # text, url, last_updated). Apply normalize to text column only.
    for src in sorted(SRC_DIR.glob("*.parquet")):
        df = pd.read_parquet(src)
        if "text" in df.columns:
            df["text"] = df["text"].apply(aggressive_normalize)
        dst = DST_DIR / src.name
        df.to_parquet(dst, index=False)
        print(f"[ok]    {src.name:35s}  {len(df):>6,} rows")

    # Parallel pairs TSV (e.g., pairs_ita_friulano.tsv has language columns).
    for src in sorted(SRC_DIR.glob("*.tsv")):
        df = pd.read_csv(src, sep="\t")
        normalized_cols = []
        for col in df.columns:
            if col in META_COLS:
                continue
            if df[col].dtype == "object":
                df[col] = df[col].apply(aggressive_normalize)
                normalized_cols.append(col)
        dst = DST_DIR / src.name
        df.to_csv(dst, sep="\t", index=False)
        print(f"[ok]    {src.name:35s}  {len(df):>6,} rows × {len(normalized_cols)} normalized cols")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
