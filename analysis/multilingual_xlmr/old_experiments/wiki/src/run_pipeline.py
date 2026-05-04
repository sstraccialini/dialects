"""
End-to-end orchestrator for the multilingual XLM-R approach on the
Wikipedia corpus.

Launch (from repo root, with venv active):
    python analysis/multilingual_xlmr/wiki/src/run_pipeline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
for p in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import MODEL_NAME, outputs_subdir, evaluation_subdir
from data_loader import load_data
from embedder import Embedder

from evaluation.evaluation import run_evaluation


def main():
    start_time = time.time()
    print(f"Loading data using MODEL: {MODEL_NAME}")

    df = load_data()
    print(f"Loaded {len(df)} sentences across {df['lang'].nunique()} languages.")

    print("Initializing multilingual embedder (XLM-R / mBERT)")
    embedder = Embedder(model_name=MODEL_NAME)

    texts = df["text"].tolist()
    embeddings = embedder.encode(texts, batch_size=32)
    print(f"Encoding matrix shape: {embeddings.shape}")

    out_dir = outputs_subdir()
    np.savez_compressed(
        out_dir / "sentence_vectors.npz",
        vectors=embeddings.astype(np.float32),
        codes=df["lang"].to_numpy(),
        model_name=np.asarray(MODEL_NAME),
    )

    # Per-variety centroid (mean over sentences) — analysis/evaluation reads
    # the language-level matrix downstream.
    lang_codes = sorted(df["lang"].unique())
    matrix = np.vstack([
        embeddings[df["lang"].to_numpy() == lang].mean(axis=0)
        for lang in lang_codes
    ]).astype(np.float32)
    pd.DataFrame(matrix, index=lang_codes).to_csv(
        out_dir / "variety_vectors.csv", float_format="%.6f"
    )
    np.savez_compressed(
        out_dir / "variety_vectors.npz",
        matrix=matrix,
        labels=np.asarray(lang_codes),
        model_name=np.asarray(MODEL_NAME),
    )

    print("\n--- Central evaluation ---")
    run_evaluation(
        variety_vectors=matrix,
        variety_codes=lang_codes,
        out_dir=evaluation_subdir(),
        method_label=f"XLM-R Wiki ({MODEL_NAME})",
    )

    elapsed = time.time() - start_time
    print(f"\nMethod outputs:       {out_dir}")
    print(f"Evaluation artefacts: {evaluation_subdir()}")
    print(f"Pipeline completed in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
