"""
2D projections of varieties (MDS and t-SNE) for visualization.

Inputs: a 14x14 cosine distance matrix and the ordered list of codes.

MDS and t-SNE produce different layouts:
- MDS preserves global variance; relative distances are faithful.
- t-SNE emphasizes local clusters; good for perceptual groupings but
  distorts global distances.

We feed the precomputed cosine distance matrix directly (metric='precomputed').
With only 14 points this is robust and fast.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
from sklearn.manifold import MDS, TSNE

from config import (
    results_subdir,
    VARIETY_GROUP,
    VARIETY_NAMES,
    GROUP_COLORS,
    GROUP_NAMES,
)


def _colors_for(codes: List[str]) -> List[str]:
    return [GROUP_COLORS.get(VARIETY_GROUP[c], "#333333") for c in codes]


def project_mds(dist: np.ndarray, random_state: int = 42) -> np.ndarray:
    """
    Classical MDS on a precomputed distance matrix.
    With only 14 points this is effectively "PCA on the distance space".
    """
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=random_state,
        normalized_stress="auto",
    )
    return mds.fit_transform(dist)


def project_tsne(dist: np.ndarray, perplexity: float = 4.0, random_state: int = 42) -> np.ndarray:
    """
    t-SNE with precomputed distances. Perplexity is kept low because
    n=14; typical values are 3-5.
    """
    tsne = TSNE(
        n_components=2,
        metric="precomputed",
        init="random",
        perplexity=perplexity,
        random_state=random_state,
        max_iter=2000,
    )
    return tsne.fit_transform(dist)


def plot_projection(
    coords: np.ndarray,
    codes: List[str],
    title: str,
    out_path: Path,
) -> Path:
    """
    Scatter plot of the 2D projection, colored by linguistic family,
    with full English names as labels.
    """
    colors = _colors_for(codes)
    labels = [VARIETY_NAMES[c] for c in codes]

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=130, edgecolor="white",
               linewidth=1.0, zorder=2)
    for i, name in enumerate(labels):
        ax.annotate(
            name,
            (coords[i, 0], coords[i, 1]),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            color=colors[i],
        )
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=8,
                   color=color, label=GROUP_NAMES.get(group, group))
        for group, color in GROUP_COLORS.items()
    ]
    # Legend always outside the plot (upper-right), identical across MDS,
    # t-SNE and dendrogram. Never overlaps the data.
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=9, frameon=False, title="Family",
              borderaxespad=0.0)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def visualize_pipeline(
    dist: np.ndarray,
    codes: List[str],
    pipeline: str,
) -> dict:
    """
    Produce two plots (MDS, t-SNE) for one pipeline, saved to
    results/<pipeline>/projection_mds.png and projection_tsne.png.
    """
    out_dir = results_subdir(pipeline)

    coords_mds = project_mds(dist)
    path_mds = out_dir / "projection_mds.png"
    plot_projection(
        coords_mds, codes,
        title=f"MDS projection ({pipeline} n-grams, cosine distance)",
        out_path=path_mds,
    )

    coords_tsne = project_tsne(dist)
    path_tsne = out_dir / "projection_tsne.png"
    plot_projection(
        coords_tsne, codes,
        title=f"t-SNE projection ({pipeline} n-grams, cosine, perplexity=4)",
        out_path=path_tsne,
    )

    return {"mds": str(path_mds), "tsne": str(path_tsne)}


if __name__ == "__main__":
    from data_loader import load_all_varieties, build_variety_documents
    from vectorize import fit_transform_word, fit_transform_char
    from similarity import cosine_distance_matrix

    print("Loading + vectorizing...")
    data, _ = load_all_varieties(verbose=False)
    docs, codes = build_variety_documents(data)

    print("Word pipeline...")
    Xw, _ = fit_transform_word(docs)
    Dw = cosine_distance_matrix(Xw)
    pw = visualize_pipeline(Dw, codes, "word")

    print("Char pipeline...")
    Xc, _ = fit_transform_char(docs)
    Dc = cosine_distance_matrix(Xc)
    pc = visualize_pipeline(Dc, codes, "char")

    print("\nSaved:")
    for k, v in {**{f"word_{k}": v for k, v in pw.items()},
                 **{f"char_{k}": v for k, v in pc.items()}}.items():
        print(f"  {k}: {v}")
