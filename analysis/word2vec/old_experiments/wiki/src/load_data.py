"""
Load Wikipedia per-variety CSVs into one combined DataFrame, with
optional per-variety sub-sampling.

CSVs live under `Dataset/wiki/{dialects_in_both_OLDI_and_Flores,languages}/<code>.csv`.
The exact subfolder per variety is given by VARIETY_DIR in config.py.
"""

from pathlib import Path
import pandas as pd

from config import (
    VARIETY_CODES,
    VARIETY_DIR,
    SAMPLE_SIZE,
    RANDOM_STATE,
)


def load_all_csvs(
    data_dir: Path = None,           # kept for legacy signature; ignored
    text_column: str = "text",
    sample_size: int = SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
    codes=VARIETY_CODES,
) -> pd.DataFrame:
    """Read each variety CSV from its mapped subfolder, sub-sample if
    above sample_size, and concatenate into one DataFrame.

    Returns columns: text, label?, article_id?, variety, source_file.
    """
    rows = []

    for code in codes:
        if code not in VARIETY_DIR:
            print(f"Warning: '{code}' not in VARIETY_DIR. Skipping.")
            continue

        csv_path = VARIETY_DIR[code] / f"{code}.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found. Skipping.")
            continue

        df = pd.read_csv(csv_path)
        if text_column not in df.columns:
            raise ValueError(f"{csv_path.name} does not contain column '{text_column}'")

        # Drop empty/NaN early to make sample size accurate.
        df = df.dropna(subset=[text_column])
        df = df[df[text_column].astype(str).str.strip() != ""]

        # Sub-sample if above cap.
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=random_state)

        keep_cols = [text_column]
        if "article_id" in df.columns:
            keep_cols.append("article_id")
        if "label" in df.columns:
            keep_cols.append("label")

        tmp = df[keep_cols].copy()
        tmp = tmp.rename(columns={text_column: "text"})
        tmp["variety"] = code
        tmp["source_file"] = csv_path.name
        rows.append(tmp)

    if not rows:
        raise ValueError("No CSV files loaded — check VARIETY_DIR mapping in config.py.")

    out = pd.concat(rows, ignore_index=True)
    out["text"] = out["text"].fillna("").astype(str)
    out = out[out["text"].str.strip() != ""].reset_index(drop=True)
    return out
