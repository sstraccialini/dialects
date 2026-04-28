"""
Pairwise cosine distance + nearest-neighbour utilities for Word2Vec.

Accepts a dense n_varieties x D embedding matrix. The cosine distance
matrix is symmetrised and clipped to [0, 2] to avoid tiny float errors.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from config import results_subdir


def cosine_distance_matrix(X: np.ndarray) -> np.ndarray:
    sim = cosine_similarity(X)
    sim = (sim + sim.T) / 2.0
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    np.clip(dist, 0.0, None, out=dist)
    return dist


def save_distance_matrix(
    dist: np.ndarray,
    codes: List[str],
    pipeline: str = "",
) -> Path:
    """Save labelled distance matrix. pipeline='' -> results/distances.csv."""
    df = pd.DataFrame(dist, index=codes, columns=codes)
    out = results_subdir(pipeline) / "distances.csv"
    df.to_csv(out, float_format="%.6f")
    return out


def nearest_neighbors(
    dist: np.ndarray,
    codes: List[str],
    k: int = 3,
) -> pd.DataFrame:
    n = dist.shape[0]
    rows = []
    for i in range(n):
        d = dist[i].copy()
        d[i] = np.inf
        order = np.argsort(d)
        row = {"code": codes[i]}
        for rank, idx in enumerate(order[:k], start=1):
            row[f"nn_{rank}"] = codes[idx]
            row[f"dist_{rank}"] = float(dist[i, idx])
        rows.append(row)
    return pd.DataFrame(rows)


def save_nearest_neighbors(
    nn_df: pd.DataFrame,
    pipeline: str = "",
) -> Path:
    out = results_subdir(pipeline) / "nearest_neighbors.csv"
    nn_df.to_csv(out, index=False)
    return out
