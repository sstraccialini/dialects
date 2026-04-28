"""
Compute pairwise distance matrices across varieties.

Given X (n_varieties, hidden_size) and the ordered list of codes, we
produce:
- cosine SIMILARITY matrix (n x n)
- cosine DISTANCE matrix   (n x n) = 1 - similarity

Matrices are saved as CSV with header (codes) and index (codes), into
`results/<pipeline>/distances.csv`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from config import results_subdir


def cosine_distance_matrix(X) -> np.ndarray:
    """Return the n x n cosine distance matrix (1 - cosine_similarity)."""
    sim = cosine_similarity(X)
    # Force numerical symmetry and zero diagonal.
    sim = (sim + sim.T) / 2.0
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return dist


def save_distance_matrix(
    dist: np.ndarray,
    codes: List[str],
    pipeline: str,
) -> Path:
    """Save the distance matrix as CSV with row/column labels."""
    df = pd.DataFrame(dist, index=codes, columns=codes)
    out = results_subdir(pipeline) / "distances.csv"
    df.to_csv(out, float_format="%.6f")
    return out


def nearest_neighbors(
    dist: np.ndarray,
    codes: List[str],
    k: int = 3,
) -> pd.DataFrame:
    """
    For each variety, return the k nearest neighbors (self excluded).

    Output: DataFrame with columns [code, nn_1, dist_1, nn_2, dist_2, ...].
    """
    n = dist.shape[0]
    rows = []
    for i in range(n):
        d = dist[i].copy()
        d[i] = np.inf  # exclude self
        order = np.argsort(d)
        row = {"code": codes[i]}
        for rank, idx in enumerate(order[:k], start=1):
            row[f"nn_{rank}"] = codes[idx]
            row[f"dist_{rank}"] = float(dist[i, idx])
        rows.append(row)
    return pd.DataFrame(rows)


def save_nearest_neighbors(
    nn_df: pd.DataFrame,
    pipeline: str,
) -> Path:
    """Persist the nearest-neighbors table to results/<pipeline>/nearest_neighbors.csv."""
    out = results_subdir(pipeline) / "nearest_neighbors.csv"
    nn_df.to_csv(out, index=False)
    return out
