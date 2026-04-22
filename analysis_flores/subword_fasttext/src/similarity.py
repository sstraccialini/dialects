"""
Pairwise cosine distance + nearest-neighbour utilities.

Accepts either:
    - a sparse/dense feature matrix (used for BPE + TF-IDF)
    - a dense n_varieties x D embedding matrix (used for FastText)

Both cases are handled uniformly via sklearn's cosine_similarity, which
does the right thing for both sparse and dense inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from config import results_subdir


def cosine_distance_matrix(X) -> np.ndarray:
    """Return the n x n cosine distance matrix (1 - cosine_similarity)."""
    sim = cosine_similarity(X)
    sim = (sim + sim.T) / 2.0
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    # Guard against tiny negative distances from floating-point error.
    np.clip(dist, 0.0, None, out=dist)
    return dist


def save_distance_matrix(
    dist: np.ndarray,
    codes: List[str],
    pipeline: str,
) -> Path:
    """Save the labelled distance matrix to results/<pipeline>/distances.csv."""
    df = pd.DataFrame(dist, index=codes, columns=codes)
    out = results_subdir(pipeline) / "distances.csv"
    df.to_csv(out, float_format="%.6f")
    return out


def nearest_neighbors(
    dist: np.ndarray,
    codes: List[str],
    k: int = 3,
) -> pd.DataFrame:
    """For each variety, return k nearest neighbours (self excluded)."""
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
    pipeline: str,
) -> Path:
    """Save the nearest-neighbours table."""
    out = results_subdir(pipeline) / "nearest_neighbors.csv"
    nn_df.to_csv(out, index=False)
    return out


def save_top_features(
    top_dict: Dict[str, List[Tuple[str, float]]],
    pipeline: str,
    filename: str = "top_features.csv",
) -> Path:
    """Persist top-features dict as long-form CSV."""
    rows = []
    for code, feats in top_dict.items():
        for rank, (f, w) in enumerate(feats, start=1):
            rows.append({"code": code, "rank": rank, "feature": f, "weight": w})
    df = pd.DataFrame(rows)
    out = results_subdir(pipeline) / filename
    df.to_csv(out, index=False)
    return out
