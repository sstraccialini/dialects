import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from config import (
    DATA_DIR,
    RESULTS_DIR,
    MODEL_DIR,
    MATRIX_DIR,
    FIGURE_DIR,
    TEXT_COLUMN,
    WORD2VEC_CONFIG,
)
from load_data import load_all_csvs
from train import train_word2vec
from build_vectors import build_text_vectors, build_variety_vectors, save_variety_vectors
from analyze import cosine_similarity_matrix, save_similarity_matrix, plot_pca, nearest_neighbors


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading CSV files from {DATA_DIR} ...")
    df = load_all_csvs(DATA_DIR, TEXT_COLUMN)

    combined_path = RESULTS_DIR / "combined_corpus.csv"
    df.to_csv(combined_path, index=False)
    print(f"Saved combined corpus to {combined_path}")

    model_path = MODEL_DIR / "word2vec.model"
    print("Training Word2Vec model ...")
    train_word2vec(df, model_path, **WORD2VEC_CONFIG)
    print(f"Saved model to {model_path}")

    print("Building text and variety vectors ...")
    model = Word2Vec.load(str(model_path))
    text_vectors = build_text_vectors(df, model)
    variety_vectors = build_variety_vectors(text_vectors)

    vectors_path = MATRIX_DIR / "variety_vectors.npz"
    save_variety_vectors(variety_vectors, vectors_path)
    print(f"Saved variety vectors to {vectors_path}")

    print("Computing similarity matrix and PCA ...")
    data = np.load(vectors_path, allow_pickle=True)
    labels = data["labels"].tolist()
    matrix = data["matrix"]

    sim_df = cosine_similarity_matrix(labels, matrix)
    sim_path = MATRIX_DIR / "cosine_similarity.csv"
    save_similarity_matrix(sim_df, sim_path)

    nn_df = nearest_neighbors(sim_df)
    nn_path = MATRIX_DIR / "nearest_neighbors.csv"
    nn_df.to_csv(nn_path, index=False)

    pca_path = FIGURE_DIR / "pca_varieties.png"
    plot_pca(labels, matrix, pca_path)

    print(f"Saved similarity matrix to {sim_path}")
    print(f"Saved nearest neighbors to {nn_path}")
    print(f"Saved PCA plot to {pca_path}")


if __name__ == "__main__":
    main()
