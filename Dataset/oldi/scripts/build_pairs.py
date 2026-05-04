"""
Build / extend OLDI parallel-pair TSV files for the 3 machine-translated
languages (deu, cat, slv).

Reads from `Dataset/oldi/not_normalized/`:
  - ita_Latn.parquet    (italian human translations, OLDI-native)
  - deu_Latn.parquet    (German, Google-MT — produced by translate_missing.py)
  - cat_Latn.parquet    (Catalan, Google-MT)
  - slv_Latn.parquet    (Slovenian, Google-MT)
  - pairs_all.tsv       (existing 11-column pair file)

Writes to `Dataset/oldi/not_normalized/`:
  - pairs_ita_tedesco.tsv      (3 cols: id, italiano, tedesco)
  - pairs_ita_catalano.tsv     (3 cols: id, italiano, catalano)
  - pairs_ita_sloveno.tsv      (3 cols: id, italiano, sloveno)
  - pairs_all.tsv  REWRITTEN   (14 cols: original 11 + tedesco + catalano + sloveno)

Both pair files preserve `id` as the primary alignment key; row `i`
contains the same source sentence across all language columns.

After running this, `python Dataset/oldi/scripts/normalize.py` will pick
up the new TSV files (it globs `*.tsv` from `not_normalized/`) and write
their aggressive-normalized counterparts to `normalized/`.

CAVEAT: deu/cat/slv columns here are MACHINE-TRANSLATED (Google), unlike
the human-translated dialect columns. They MUST NOT be used as training
pairs for fine-tuning — they're intended only for cross-lingual cosine
similarity at the sentence level.

Run with:
    python Dataset/oldi/scripts/build_pairs.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]    # Dataset/oldi/
SRC_DIR = ROOT / "not_normalized"

# (parquet ISO code, column name in TSV files using Italian language names).
NEW_LANGS = [
    ("deu", "tedesco"),
    ("cat", "catalano"),
    ("slv", "sloveno"),
]
ITA_PARQUET = SRC_DIR / "ita_Latn.parquet"
PAIRS_ALL = SRC_DIR / "pairs_all.tsv"


def load_text_by_id(parquet_path: Path) -> pd.Series:
    """Load a parquet's text column indexed by id, sorted ascending."""
    df = pd.read_parquet(parquet_path)
    if "id" not in df.columns or "text" not in df.columns:
        raise ValueError(f"{parquet_path} missing id/text columns")
    return df.sort_values("id").set_index("id")["text"]


def main():
    if not ITA_PARQUET.exists():
        raise SystemExit(f"Missing {ITA_PARQUET}")

    ita = load_text_by_id(ITA_PARQUET)
    print(f"Loaded ita: {len(ita):,} sentences")

    # Load the 3 new languages (must already exist — produced by translate_missing.py).
    new_texts = {}
    for iso3, ita_name in NEW_LANGS:
        p = SRC_DIR / f"{iso3}_Latn.parquet"
        if not p.exists():
            raise SystemExit(
                f"Missing {p}. Run `python Dataset/oldi/scripts/translate_missing.py` first."
            )
        s = load_text_by_id(p)
        if not s.index.equals(ita.index):
            raise SystemExit(
                f"id mismatch between ita_Latn.parquet and {iso3}_Latn.parquet"
            )
        new_texts[ita_name] = s
        print(f"Loaded {iso3} ({ita_name}): {len(s):,} sentences")

    # 1. Per-language pair files: id, italiano, <lang>.
    for iso3, ita_name in NEW_LANGS:
        df = pd.DataFrame({
            "id": ita.index,
            "italiano": ita.values,
            ita_name: new_texts[ita_name].values,
        })
        out = SRC_DIR / f"pairs_ita_{ita_name}.tsv"
        df.to_csv(out, sep="\t", index=False)
        print(f"  -> {out.name}  ({len(df):,} rows × {len(df.columns)} cols)")

    # 2. Extended pairs_all.tsv: original columns + 3 new ones.
    if not PAIRS_ALL.exists():
        raise SystemExit(f"Missing {PAIRS_ALL}")
    pairs = pd.read_csv(PAIRS_ALL, sep="\t")
    print()
    print(f"pairs_all.tsv before: {pairs.shape[0]:,} rows × {pairs.shape[1]} cols  ({list(pairs.columns)})")

    # Drop new columns if a previous run already added them, then re-add to
    # keep this script idempotent.
    for _, ita_name in NEW_LANGS:
        if ita_name in pairs.columns:
            pairs = pairs.drop(columns=[ita_name])

    if "id" not in pairs.columns:
        raise SystemExit("pairs_all.tsv has no `id` column — cannot align.")

    # Align by id (assume pairs_all.tsv has the same 0..6192 ids).
    pairs = pairs.set_index("id")
    for iso3, ita_name in NEW_LANGS:
        pairs[ita_name] = new_texts[ita_name]
    pairs = pairs.reset_index()

    # Verify no row dropped or NaN introduced via the join.
    n_nan = int(pairs[[name for _, name in NEW_LANGS]].isna().sum().sum())
    if n_nan:
        raise SystemExit(
            f"{n_nan} NaNs in extended pair file — id mismatch between pairs_all.tsv and parquets."
        )

    pairs.to_csv(PAIRS_ALL, sep="\t", index=False)
    print(f"pairs_all.tsv after:  {pairs.shape[0]:,} rows × {pairs.shape[1]} cols  ({list(pairs.columns)})")

    print()
    print("Done. Next: `python Dataset/oldi/scripts/normalize.py` to write normalized TSVs.")


if __name__ == "__main__":
    main()
