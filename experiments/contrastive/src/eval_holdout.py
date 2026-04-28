"""
Re-evaluate the contrastive model using ONLY held-out FLORES indices.

The original run_contrastive.py evaluation embedded all 2009 sentences
per variety, including the 1507 sentences seen during training. This
script restricts evaluation to the 502 held-out indices for a clean
generalization measure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from config import REPO_ROOT, FLORES_DIR, RESULTS_DIR, OUTPUT_MODEL, EVAL_VARIETIES, BATCH_SIZE


def load_variety(name: str) -> list[str]:
    with open(FLORES_DIR / f"{name}.txt", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    print("=== Eval-only on FLORES test holdout ===")
    print()

    split_path = RESULTS_DIR / "train_test_split.json"
    print(f"Loading {split_path} ...")
    with open(split_path) as f:
        split = json.load(f)
    # The script saved test indices per dialect, but the indices are
    # globally consistent across dialects (shared shuffle of FLORES rows).
    test_indices = sorted(set().union(*split["test_indices_per_dialect"].values()))
    print(f"  test indices: {len(test_indices)} (min={min(test_indices)}, max={max(test_indices)})")
    print()

    print(f"Loading model: {OUTPUT_MODEL}")
    model = SentenceTransformer(OUTPUT_MODEL)
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
    print(f"  device: {model.device}")
    print()

    centroids = {}
    print(f"Encoding {len(EVAL_VARIETIES)} varieties on holdout subset")
    for var in EVAL_VARIETIES:
        sents_full = load_variety(var)
        # Restrict to test indices that exist in this variety
        sents_holdout = [sents_full[i] for i in test_indices if i < len(sents_full)]
        emb = model.encode(sents_holdout, batch_size=BATCH_SIZE, show_progress_bar=False,
                           convert_to_numpy=True)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-9)
        centroid = emb.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid).clip(min=1e-9)
        centroids[var] = centroid
        print(f"  {var:12s}  {len(sents_holdout)} sentences")
    print()

    labels = list(centroids.keys())
    n = len(labels)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim = float(np.dot(centroids[labels[i]], centroids[labels[j]]))
            dist[i, j] = 1.0 - sim
    df = pd.DataFrame(dist, index=labels, columns=labels)
    out_csv = RESULTS_DIR / "distances_holdout.csv"
    df.to_csv(out_csv, float_format="%.6f")
    print(f"Saved: {out_csv}")

    print()
    print("=== top-3 NN (excluding self) ===")
    for var in df.index:
        row = df[var].drop(var).sort_values()
        top3 = row.head(3)
        formatted = ", ".join(f"{idx}({val:.4f})" for idx, val in top3.items())
        print(f"  {var:12s} -> {formatted}")


if __name__ == "__main__":
    main()
