import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from .config import DATASETS_DIR, LANGUAGES, SAMPLES_PER_LANG, RANDOM_SEED

def load_data(limit_per_lang=SAMPLES_PER_LANG, seed=RANDOM_SEED):
    """
    Load data from the dataset CSV files and sample evenly.
    Returns:
        pd.DataFrame containing text, label (language id), and original article_id
    """
    all_data = []
    
    # We look for files named <lang>.csv in DATASETS_DIR
    for lang in tqdm(LANGUAGES, desc="Loading datasets"):
        file_path = DATASETS_DIR / f"{lang}.csv"
        
        if not file_path.exists():
            print(f"Warning: Data file for language '{lang}' not found at {file_path}. Skipping.")
            continue
            
        try:
            # Depending on structure, expect 'text', 'label', 'article_id' based on preview
            df = pd.read_csv(file_path, header=0, names=["text", "label", "article_id"], on_bad_lines='skip', engine='python')
            
            # Filter empty strings or NaNs
            df = df.dropna(subset=['text'])
            df = df[df['text'].str.strip() != ""]
            
            # Sample data safely regarding length
            sample_size = min(len(df), limit_per_lang)
            df_sampled = df.sample(n=sample_size, random_state=seed)
            
            df_sampled['lang'] = lang
            all_data.append(df_sampled)
            
        except Exception as e:
            print(f"Failed to load {lang}.csv: {e}")

    if not all_data:
        raise ValueError(f"No valid datasets were loaded from {DATASETS_DIR}")
        
    full_df = pd.concat(all_data, ignore_index=True)
    return full_df
