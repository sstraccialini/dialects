"""
04_embedding_galaxy.py — 2D MDS projection of variety embeddings.

Two side-by-side panels: XLM-R (transformer) vs Word2Vec (lexical).
Each variety is plotted as a node positioned via metric MDS on the cosine
distance matrix. Halo glow = family color, label inside.

Output: plots/outputs/04_embedding_galaxy.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    PLOTS_DIR, REPO_ROOT,
    load_npz, cosine_similarity_matrix,
    family_of, color_of, display_of, FAMILY_COLORS, FAMILY_DISPLAY,
    ITALO_ROMANCE_DIALECTS,
)


def _mds_2d(sim: np.ndarray, *, seed: int = 7) -> np.ndarray:
    dist = np.clip(1.0 - sim, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    mds = MDS(n_components=2, dissimilarity="precomputed",
              random_state=seed, n_init=8, normalized_stress="auto")
    return mds.fit_transform(dist)


def _draw_panel(ax, X, codes, title):
    sim = cosine_similarity_matrix(X)
    pts = _mds_2d(sim)

    italian_code = "italiano" if "italiano" in codes else "ita"
    if italian_code in codes:
        ita_idx = codes.index(italian_code)
        ita_pos = pts[ita_idx]
        # nearest-3 to standard Italian get connecting line
        sims_to_ita = sim[ita_idx]
        order = np.argsort(-sims_to_ita)
        # skip self
        order = [i for i in order if i != ita_idx][:3]
        for j in order:
            ax.plot([ita_pos[0], pts[j, 0]], [ita_pos[1], pts[j, 1]],
                    color="#ff7f0e", alpha=0.45, linewidth=1.3,
                    linestyle="--", zorder=1)

    for i, c in enumerate(codes):
        col = color_of(c)
        is_dialect = c in ITALO_ROMANCE_DIALECTS
        # halo glow (3 layers)
        for r, a in [(900, 0.10), (600, 0.18), (380, 0.30)]:
            ax.scatter(pts[i, 0], pts[i, 1], s=r, c=col,
                       alpha=a, edgecolor="none", zorder=2)
        # core marker
        marker = "*" if c in ("italiano", "ita") else "o"
        size = 360 if marker == "*" else (220 if is_dialect else 180)
        ax.scatter(pts[i, 0], pts[i, 1], s=size, c=col,
                   edgecolor="white", linewidth=1.5, marker=marker, zorder=3)
        # label
        ax.annotate(
            display_of(c),
            (pts[i, 0], pts[i, 1]),
            xytext=(0, -22 if is_dialect else 18),
            textcoords="offset points",
            ha="center", fontsize=8.5, fontweight="bold",
            color="#1a1a1a", zorder=4,
            bbox=dict(boxstyle="round,pad=0.18", fc="white",
                      ec=col, lw=1.0, alpha=0.85),
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_color("#cccccc")
    ax.set_facecolor("#0d1023")  # dark space-like background


def main():
    sources = [
        ("XLM-R • FLORES (zero-shot, transformer)",
         REPO_ROOT / "analysis/multilingual_xlmr/old_experiments/flores/method_outputs/variety_vectors.npz"),
        ("Word2Vec • Wikipedia (lexical-distributional)",
         REPO_ROOT / "analysis/word2vec/old_experiments/flores/method_outputs/variety_vectors.npz"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(17, 9), facecolor="#0d1023")
    for ax, (title, path) in zip(axes, sources):
        X, codes = load_npz(path)
        _draw_panel(ax, X, codes, title)

    fig.suptitle(
        "Embedding galaxy — varieties as stars in 2-D MDS space\n"
        "Dashed orange lines: nearest-3 neighbours of standard Italian",
        fontsize=15, fontweight="bold", color="white", y=0.99,
    )

    # legend
    handles = [
        plt.scatter([], [], s=200, c=FAMILY_COLORS[fam],
                    label=FAMILY_DISPLAY[fam], edgecolor="white", lw=1)
        for fam in ("italo_romance", "italian", "romance",
                    "germanic", "english", "slavic", "greek", "semitic")
    ]
    leg = fig.legend(
        handles=handles, loc="lower center", ncol=8, frameon=False,
        bbox_to_anchor=(0.5, -0.01), fontsize=10,
        labelcolor="white",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    out = PLOTS_DIR / "04_embedding_galaxy.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="#0d1023")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
