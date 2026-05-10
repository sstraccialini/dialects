"""
01_method_comparison_panel.py — multi-model similarity heatmap panel.

Renders one similarity heatmap per method on a single canvas so the eye can
compare which model emphasises which structure (Italo-Romance cluster,
Romance-vs-Slavic split, etc.).

Output: plots/outputs/01_method_comparison_panel.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    PLOTS_DIR, SAVED_VECTOR_SOURCES, NEW_SIM_SOURCES,
    load_source, sort_codes_by_family, display_of, color_of,
)


def panel_heatmap(ax, sim, codes, title):
    sorted_c = sort_codes_by_family(codes)
    order = [codes.index(c) for c in sorted_c]
    M = sim[np.ix_(order, order)]
    n = len(sorted_c)

    im = ax.imshow(M, cmap="RdYlGn", vmin=-0.2, vmax=1.0, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([display_of(c) for c in sorted_c], rotation=60, ha="right", fontsize=6.5)
    ax.set_yticklabels([display_of(c) for c in sorted_c], fontsize=6.5)
    for i, c in enumerate(sorted_c):
        col = color_of(c)
        ax.get_xticklabels()[i].set_color(col)
        ax.get_yticklabels()[i].set_color(col)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    return im


def main():
    sources = SAVED_VECTOR_SOURCES + NEW_SIM_SOURCES
    avail = []
    for s in sources:
        try:
            sim, codes = load_source(s)
            avail.append((s["label"], sim, codes))
            print(f"  loaded {s['label']:<45} {sim.shape}")
        except FileNotFoundError as e:
            print(f"  SKIP   {s['label']}: {e}")

    n = len(avail)
    cols = 4 if n >= 8 else 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 4.6, rows * 4.6),
        gridspec_kw={"hspace": 0.55, "wspace": 0.35},
    )
    axes = axes.ravel()
    last_im = None
    for i, (label, sim, codes) in enumerate(avail):
        last_im = panel_heatmap(axes[i], sim, codes, label)
    for j in range(len(avail), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Variety similarity matrices across all saved models\n"
        "(Cosine similarity, sorted by family. Italo-Romance dialects highlighted in red.)",
        fontsize=14, fontweight="bold", y=0.995,
    )
    cbar = fig.colorbar(
        last_im, ax=axes.tolist(),
        orientation="horizontal", fraction=0.025, pad=0.04, shrink=0.5,
    )
    cbar.set_label("Cosine similarity", fontsize=10)

    out = PLOTS_DIR / "01_method_comparison_panel.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
