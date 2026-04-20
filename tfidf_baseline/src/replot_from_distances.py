"""
Regenerate dendrogram.png, projection_mds.png and projection_tsne.png
from the already-saved distances.csv files, without rerunning TF-IDF.

Use this after editing labels/colors/legend in config.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import results_subdir
from cluster import hierarchical_linkage, plot_dendrogram
from visualize import project_mds, project_tsne, plot_projection


def replot(pipeline: str) -> None:
    dist_path = results_subdir(pipeline) / "distances.csv"
    if not dist_path.exists():
        raise FileNotFoundError(dist_path)

    df = pd.read_csv(dist_path, index_col=0)
    codes = list(df.index)
    dist = df.values.astype(float)

    # Dendrogram
    Z = hierarchical_linkage(dist, method="average")
    plot_dendrogram(
        Z, codes,
        title=f"Hierarchical clustering ({pipeline} n-grams, cosine + average linkage)",
        out_path=results_subdir(pipeline) / "dendrogram.png",
    )

    # MDS
    coords_mds = project_mds(dist)
    plot_projection(
        coords_mds, codes,
        title=f"MDS projection ({pipeline} n-grams, cosine distance)",
        out_path=results_subdir(pipeline) / "projection_mds.png",
    )

    # t-SNE
    coords_tsne = project_tsne(dist)
    plot_projection(
        coords_tsne, codes,
        title=f"t-SNE projection ({pipeline} n-grams, cosine, perplexity=4)",
        out_path=results_subdir(pipeline) / "projection_tsne.png",
    )

    print(f"[{pipeline}] dendrogram + MDS + t-SNE regenerated")


def main() -> None:
    for pipeline in ("word", "char"):
        replot(pipeline)


if __name__ == "__main__":
    main()
