from pathlib import Path
import pandas as pd
from gensim.models import Word2Vec

from preprocess import tokenize


def train_word2vec(
    df: pd.DataFrame,
    model_path: Path,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    sg: int = 1,
    workers: int = 4,
    epochs: int = 15,
    seed: int = 42,
) -> Word2Vec:
    sentences = [tokenize(text) for text in df["text"]]
    sentences = [s for s in sentences if s]

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        workers=workers,
        epochs=epochs,
        seed=seed,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    return model
