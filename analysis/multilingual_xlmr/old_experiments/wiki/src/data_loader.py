import pandas as pd
from tqdm import tqdm
from .config import VARIETY_DIR, LANGUAGES, SAMPLES_PER_LANG, RANDOM_SEED


def load_data(limit_per_lang=SAMPLES_PER_LANG, seed=RANDOM_SEED):
    """
    Load data from the dataset CSV files and sample evenly.

    CSVs live under Dataset/wiki/{dialects_in_both_OLDI_and_Flores,languages}/
    via the VARIETY_DIR mapping in config.py.

    Returns:
        pd.DataFrame containing text, label (language id), original article_id, lang.
    """
    all_data = []

    for lang in tqdm(LANGUAGES, desc="Loading datasets"):
        if lang not in VARIETY_DIR:
            print(f"Warning: '{lang}' not in VARIETY_DIR mapping. Skipping.")
            continue

        file_path = VARIETY_DIR[lang] / f"{lang}.csv"
        if not file_path.exists():
            print(f"Warning: Data file for language '{lang}' not found at {file_path}. Skipping.")
            continue

        try:
            df = pd.read_csv(file_path, header=0,
                             names=["text", "label", "article_id"],
                             on_bad_lines='skip', engine='python')

            # Filter empty strings or NaNs
            df = df.dropna(subset=['text'])
            df = df[df['text'].str.strip() != ""]

            # Sample data, capped at limit_per_lang
            sample_size = min(len(df), limit_per_lang)
            df_sampled = df.sample(n=sample_size, random_state=seed)

            df_sampled['lang'] = lang
            all_data.append(df_sampled)

        except Exception as e:
            print(f"Failed to load {lang}.csv: {e}")

    if not all_data:
        raise ValueError("No valid datasets were loaded.")

    full_df = pd.concat(all_data, ignore_index=True)
    return full_df
