from pathlib import Path
import pandas as pd


def load_all_csvs(data_dir: Path, text_column: str) -> pd.DataFrame:
    rows = []

    for csv_path in sorted(data_dir.glob("*.csv")):
        if csv_path.stem.endswith("_meta"):
            continue
        variety = csv_path.stem.lower()
        df = pd.read_csv(csv_path)

        if text_column not in df.columns:
            raise ValueError(f"{csv_path.name} does not contain column '{text_column}'")

        keep_cols = [text_column]

        if "article_id" in df.columns:
            keep_cols.append("article_id")

        if "label" in df.columns:
            keep_cols.append("label")

        tmp = df[keep_cols].copy()
        tmp = tmp.rename(columns={text_column: "text"})
        tmp["variety"] = variety
        tmp["source_file"] = csv_path.name
        rows.append(tmp)

    if not rows:
        raise ValueError(f"No CSV files found in {data_dir}")

    out = pd.concat(rows, ignore_index=True)
    out["text"] = out["text"].fillna("").astype(str)
    out = out[out["text"].str.strip() != ""].reset_index(drop=True)
    return out
