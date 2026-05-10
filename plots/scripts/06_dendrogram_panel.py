"""
06_dendrogram_panel.py — agglomerative-clustering dendrograms,
one per saved method, on a single figure for direct comparison.

Output: plots/outputs/06_dendrogram_panel.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    PLOTS_DIR, SAVED_VECTOR_SOURCES, NEW_SIM_SOURCES,
    load_source, color_of, display_of,
)


def _draw_dendro(ax, sim, codes, title):
    n = len(codes)
    dist = np.clip(1.0 - sim, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2  # ensure symmetric
    Z = linkage(squareform(dist, checks=False), method="average")

    labels = [display_of(c) for c in codes]
    label_colors = {display_of(c): color_of(c) for c in codes}

    ddata = dendrogram(
        Z, ax=ax, labels=labels, leaf_rotation=60, leaf_font_size=8,
        color_threshold=0.0, above_threshold_color="#888888",
    )
    # color tick labels by family
    for tick in ax.get_xmajorticklabels():
        tick.set_color(label_colors.get(tick.get_text(), "#000000"))
        tick.set_fontweight("bold")

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    sources = SAVED_VECTOR_SOURCES + NEW_SIM_SOURCES
    loaded = []
    for s in sources:
        try:
            sim, codes = load_source(s)
            loaded.append((s["label"], sim, codes))
        except FileNotFoundError:
            continue

    n = len(loaded)
    cols = 4 if n >= 8 else 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 4.0))
    axes = axes.ravel()
    for i, (label, sim, codes) in enumerate(loaded):
        _draw_dendro(axes[i], sim, codes, label)
    for j in range(len(loaded), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Hierarchical clustering: how each model groups the same varieties\n"
        "(Average linkage on cosine distance. Italo-Romance dialects in red.)",
        fontsize=14, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = PLOTS_DIR / "06_dendrogram_panel.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
