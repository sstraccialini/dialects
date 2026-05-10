"""
05_cross_method_agreement.py — do different methods rank variety pairs
the same way?  We compute Spearman ρ between the upper-triangles of every
pair of similarity matrices, restricted to varieties they share.

Output: plots/outputs/05_cross_method_agreement.png
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    PLOTS_DIR, SAVED_VECTOR_SOURCES, NEW_SIM_SOURCES,
    load_source,
)


def main():
    sources = SAVED_VECTOR_SOURCES + NEW_SIM_SOURCES
    loaded = []
    for s in sources:
        try:
            sim, codes = load_source(s)
            loaded.append((s["short"], s["label"], sim, codes))
        except FileNotFoundError:
            continue

    n = len(loaded)
    print(f"Loaded {n} methods.")

    rho = np.full((n, n), np.nan)
    n_overlap = np.zeros((n, n), dtype=int)
    for i in range(n):
        rho[i, i] = 1.0
        for j in range(i + 1, n):
            ci, cj = loaded[i][3], loaded[j][3]
            shared = [c for c in ci if c in cj]
            if len(shared) < 4:
                continue
            ai = [ci.index(c) for c in shared]
            aj = [cj.index(c) for c in shared]
            Si = loaded[i][2][np.ix_(ai, ai)]
            Sj = loaded[j][2][np.ix_(aj, aj)]
            iu = np.triu_indices(len(shared), k=1)
            r, _ = spearmanr(Si[iu], Sj[iu])
            rho[i, j] = rho[j, i] = r
            n_overlap[i, j] = n_overlap[j, i] = len(shared)

    labels = [lbl for _, lbl, _, _ in loaded]

    # ---- Heatmap ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(max(11, n * 0.85), max(9, n * 0.7)))
    im = ax.imshow(rho, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(n):
        for j in range(n):
            if not np.isnan(rho[i, j]):
                v = rho[i, j]
                ax.text(j, i, f"{v:+.2f}",
                        ha="center", va="center", fontsize=7.5,
                        color="black" if abs(v) < 0.55 else "white",
                        fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.025)
    cbar.set_label("Spearman ρ on shared variety pairs", fontsize=10)

    ax.set_title(
        "Do different models agree on variety distances?\n"
        "Spearman ρ between similarity matrices on the overlap of varieties",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()

    out = PLOTS_DIR / "05_cross_method_agreement.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
