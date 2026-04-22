import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from preprocess import tokenize


def average_word_vectors(tokens, model):
    vectors = [model.wv[t] for t in tokens if t in model.wv]
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


def build_text_vectors(df: pd.DataFrame, model: Word2Vec) -> pd.DataFrame:
    rows = []

    for idx, row in df.iterrows():
        tokens = tokenize(row["text"])
        vec = average_word_vectors(tokens, model)
        if vec is None:
            continue

        rows.append({
            "row_id": idx,
            "variety": row["variety"],
            "vector": vec,
        })

    return pd.DataFrame(rows)


def build_variety_vectors(text_vectors: pd.DataFrame) -> pd.DataFrame:
    grouped = text_vectors.groupby("variety")["vector"].apply(list)

    rows = []
    for variety, vectors in grouped.items():
        mean_vec = np.mean(np.stack(vectors), axis=0)
        rows.append({"variety": variety, "vector": mean_vec})

    return pd.DataFrame(rows)


def save_variety_vectors(variety_vectors: pd.DataFrame, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.stack(variety_vectors["vector"].to_list())
    labels = variety_vectors["variety"].to_list()
    np.savez(out_path, labels=np.array(labels), matrix=matrix)
