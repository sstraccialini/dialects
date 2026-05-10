"""
02_italy_dialect_map.py — geographic map of Italo-Romance dialects
with edges weighted by embedding similarity.

Two-panel figure:
  LEFT  : XLM-R (transformer, FLORES) — semantic encoder
  RIGHT : Word2Vec (Wikipedia) — distributional/lexical encoder

Each dialect sits at the lat/lon of its regional capital.  Edges connect
every dialect pair; thickness encodes the cosine similarity between their
variety embeddings.  Marker size encodes similarity to standard Italian.

A stylised Italian peninsula outline is sketched in the background so the
geography is recognisable without external GIS dependencies.

Output: plots/outputs/02_italy_dialect_map.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.path import Path as MplPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    PLOTS_DIR, REPO_ROOT,
    load_npz, cosine_similarity_matrix,
    DIALECT_COORDS, ITALO_ROMANCE_DIALECTS,
    color_of, display_of,
)


# A hand-traced very-stylised Italy boot.  Each (lon, lat) point.
# Not a precise outline — purpose is recognisability + aesthetics.
ITALY_OUTLINE = [
    (7.5, 45.9), (8.6, 46.4), (10.5, 46.6), (12.2, 46.7),
    (13.6, 46.5), (13.9, 45.6), (13.5, 45.3),
    (12.5, 45.4), (12.9, 44.6), (13.8, 43.8), (14.0, 42.7),
    (14.7, 42.0), (15.2, 41.4), (16.0, 41.9), (17.0, 40.9),
    (18.4, 40.1), (18.5, 39.8), (17.3, 39.9), (17.2, 38.9),
    (16.6, 38.8), (15.9, 38.4), (15.7, 38.0), (16.1, 37.9),
    (16.0, 37.4),
    (15.6, 37.1), (12.4, 36.7), (12.3, 38.0), (12.7, 38.1),
    (13.5, 38.2), (15.2, 38.3),  # Sicily mostly
    (15.6, 38.7), (15.8, 39.7), (15.9, 40.5), (14.0, 40.6),
    (13.8, 40.7), (12.6, 40.1), (10.5, 41.5),
    (11.2, 42.5), (10.3, 43.0), (9.8, 44.0), (8.5, 44.4),
    (7.6, 43.8), (7.9, 44.5), (6.9, 45.1), (7.0, 45.5),
    (7.5, 45.9),
]

SARDINIA_OUTLINE = [
    (8.4, 41.2), (9.0, 41.2), (9.7, 41.0), (9.9, 40.8),
    (9.7, 40.0), (9.6, 39.4), (9.0, 39.1), (8.4, 38.9),
    (8.4, 39.5), (8.7, 40.1), (8.4, 41.2),
]


def _draw_outline(ax, coords, *, fc="#f4ecd8", ec="#8b6f47", alpha=0.6, lw=1.0):
    poly = mpatches.Polygon(
        np.array(coords), closed=True, facecolor=fc,
        edgecolor=ec, alpha=alpha, linewidth=lw, zorder=1,
    )
    ax.add_patch(poly)


def _ita_sea(ax):
    sea = mpatches.Rectangle(
        (5.5, 36.0), 14.5, 11.0, facecolor="#e6f1f7",
        edgecolor="none", zorder=0,
    )
    ax.add_patch(sea)


def _draw_panel(ax, sim, codes, title):
    _ita_sea(ax)
    _draw_outline(ax, ITALY_OUTLINE)
    _draw_outline(ax, SARDINIA_OUTLINE)

    code_to_idx = {c: i for i, c in enumerate(codes)}
    italian_code = "italiano" if "italiano" in code_to_idx else "ita"

    dialects_present = [c for c in DIALECT_COORDS
                        if c in code_to_idx and c in ITALO_ROMANCE_DIALECTS]
    italian_present = italian_code in code_to_idx

    # ---- edges between every pair of dialects ----
    pair_sims = []
    for i in range(len(dialects_present)):
        for j in range(i + 1, len(dialects_present)):
            ci, cj = dialects_present[i], dialects_present[j]
            s = float(sim[code_to_idx[ci], code_to_idx[cj]])
            pair_sims.append(s)
    smin, smax = min(pair_sims), max(pair_sims)
    spread = max(smax - smin, 1e-6)

    for i in range(len(dialects_present)):
        for j in range(i + 1, len(dialects_present)):
            ci, cj = dialects_present[i], dialects_present[j]
            yi, xi = DIALECT_COORDS[ci]
            yj, xj = DIALECT_COORDS[cj]
            s = float(sim[code_to_idx[ci], code_to_idx[cj]])
            t = (s - smin) / spread
            ax.plot(
                [xi, xj], [yi, yj],
                color=plt.cm.RdYlGn(0.15 + 0.7 * t),
                linewidth=0.6 + 4.5 * t,
                alpha=0.35 + 0.55 * t,
                zorder=2,
            )

    # ---- nodes: marker size encodes similarity to standard Italian ----
    if italian_present:
        ita_sims = [float(sim[code_to_idx[ci], code_to_idx[italian_code]])
                    for ci in dialects_present]
        i_min, i_max = min(ita_sims), max(ita_sims)
        i_spread = max(i_max - i_min, 1e-6)
    else:
        ita_sims = [0.5] * len(dialects_present)
        i_min, i_spread = 0.0, 1.0

    for ci, isim in zip(dialects_present, ita_sims):
        y, x = DIALECT_COORDS[ci]
        t = (isim - i_min) / i_spread
        size = 200 + 600 * t
        ax.scatter(x, y, s=size, c=color_of(ci),
                   edgecolor="#222222", linewidth=1.2, zorder=4)
        # label
        ax.annotate(
            display_of(ci), (x, y),
            xytext=(8, 6), textcoords="offset points",
            fontsize=9, fontweight="bold",
            color="#222222",
            zorder=5,
        )

    # standard Italian = star
    if italian_present:
        y, x = DIALECT_COORDS[italian_code]
        ax.scatter(x, y, marker="*", s=850, c="#ff7f0e",
                   edgecolor="#222222", linewidth=1.2, zorder=4)
        ax.annotate(
            "Italian (std)", (x, y),
            xytext=(10, 4), textcoords="offset points",
            fontsize=10, fontweight="bold",
            color="#cc6600", zorder=5,
        )

    ax.set_xlim(5.5, 19.5)
    ax.set_ylim(36.0, 47.0)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    sources = [
        (
            "Transformer view — XLM-R / FLORES (zero-shot)\n"
            "edges = inter-dialect cosine similarity",
            REPO_ROOT / "analysis/multilingual_xlmr/old_experiments/flores/method_outputs/variety_vectors.npz",
        ),
        (
            "Lexical view — Word2Vec / Wikipedia\n"
            "edges = inter-dialect cosine similarity",
            REPO_ROOT / "analysis/word2vec/old_experiments/flores/method_outputs/variety_vectors.npz",
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 9))
    for ax, (title, p) in zip(axes, sources):
        X, codes = load_npz(p)
        sim = cosine_similarity_matrix(X)
        _draw_panel(ax, sim, codes, title)

    fig.suptitle(
        "Italian dialects on the map: same geography, different embedding spaces",
        fontsize=15, fontweight="bold", y=0.99,
    )

    legend_handles = [
        mpatches.Patch(color="#d62728", label="Italo-Romance dialect"),
        mpatches.Patch(color="#ff7f0e", label="★ Standard Italian"),
        plt.Line2D([0], [0], color=plt.cm.RdYlGn(0.85), linewidth=4,
                   label="Thicker / greener edge = higher embedding similarity"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
                   markeredgecolor="#222222", markersize=14,
                   label="Larger node = higher similarity to standard Italian"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, -0.02), fontsize=10,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])

    out = PLOTS_DIR / "02_italy_dialect_map.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
