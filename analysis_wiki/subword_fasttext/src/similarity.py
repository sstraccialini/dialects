"""
Pairwise cosine distance matrix and nearest-neighbor utilities.

Identical logic to the TF-IDF baseline; works on both dense numpy
arrays (FastText embeddings) and sparse scipy matrices (BPE TF-IDF).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

from config import results_subdir


def cosine_distance_matrix(X) -> np.ndarray:
    """Return the n × n cosine distance matrix (1 − cosine_similarity)."""
    sim = cosine_similarity(X)
    sim = (sim + sim.T) / 2.0   # enforce symmetry
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return dist


def save_distance_matrix(
    dist: np.ndarray,
    codes: List[str],
    pipeline: str,
) -> Path:
    """Save distance matrix as CSV to results/<pipeline>/distances.csv."""
    df = pd.DataFrame(dist, index=codes, columns=codes)
    out = results_subdir(pipeline) / "distances.csv"
    df.to_csv(out, float_format="%.6f")
    return out


def nearest_neighbors(
    dist: np.ndarray,
    codes: List[str],
    k: int = 3,
) -> pd.DataFrame:
    """For each variety, return the k nearest neighbors (self excluded)."""
    rows = []
    for i in range(len(codes)):
        d = dist[i].copy()
        d[i] = np.inf
        order = np.argsort(d)
        row = {"code": codes[i]}
        for rank, idx in enumerate(order[:k], start=1):
            row[f"nn_{rank}"] = codes[idx]
            row[f"dist_{rank}"] = float(dist[i, idx])
        rows.append(row)
    return pd.DataFrame(rows)


def save_nearest_neighbors(nn_df: pd.DataFrame, pipeline: str) -> Path:
    out = results_subdir(pipeline) / "nearest_neighbors.csv"
    nn_df.to_csv(out, index=False)
    return out


def save_top_features(top_dict: dict, pipeline: str) -> Path:
    """
    Persist top-features dict as long-form CSV:
    columns = [code, rank, feature, weight].
    Written to results/<pipeline>/top_features.csv.
    """
    rows = []
    for code, feats in top_dict.items():
        for rank, (f, w) in enumerate(feats, start=1):
            rows.append({"code": code, "rank": rank, "feature": f, "weight": w})
    out = results_subdir(pipeline) / "top_features.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out
