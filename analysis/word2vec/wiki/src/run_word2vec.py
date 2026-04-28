"""
End-to-end orchestrator for the Word2Vec approach on the Wikipedia corpus.

Launch (from repo root, with venv active):
    python analysis/word2vec/wiki/src/run_word2vec.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
for p in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import (
    DATA_DIR, MODEL_DIR, TEXT_COLUMN, WORD2VEC_CONFIG,
    outputs_subdir, evaluation_subdir,
)
from load_data import load_all_csvs
from train import train_word2vec
from build_vectors import build_text_vectors, build_variety_vectors, save_variety_vectors

from evaluation.evaluation import run_evaluation


def main():
    out_dir = outputs_subdir()

    print(f"Loading CSV files from {DATA_DIR} ...")
    df = load_all_csvs(DATA_DIR, TEXT_COLUMN)

    combined_path = out_dir / "combined_corpus.csv"
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

    vectors_path = out_dir / "variety_vectors.npz"
    save_variety_vectors(variety_vectors, vectors_path)
    print(f"Saved variety vectors to {vectors_path}")

    matrix = np.stack(variety_vectors["vector"].to_list())
    labels = variety_vectors["variety"].to_list()

    print("\n--- Central evaluation ---")
    run_evaluation(
        variety_vectors=matrix,
        variety_codes=labels,
        out_dir=evaluation_subdir(),
        method_label="Word2Vec (Wiki)",
    )

    print(f"\nMethod outputs:       {out_dir}")
    print(f"Evaluation artefacts: {evaluation_subdir()}")


if __name__ == "__main__":
    main()
