"""
03_dialect_radar.py — radar (spider) charts of every Italo-Romance
dialect's "lean" toward each non-dialect reference language.

Each axis is a reference language.  The plotted value is the *deviation*
of this dialect's similarity-to-reference from the mean similarity-to-
reference across all dialects (z-scored across dialects).

Why deviations and not raw cosine?  Translation-aware sentence encoders
(LaBSE, MiniLM, XLM-R) saturate raw cosine near 1 for every dialect-vs-
reference pair on a parallel corpus — every dialect looks equally close
to every language.  Subtracting the across-dialect mean per axis cancels
that shared content direction and reveals each dialect's individual lean.
This is exactly the same trick we apply at sentence level via the
content-subtracted centroid metric in evaluation/sentence_relations.py.

Two panels for direct contrast:
  LEFT  : raw cosine — visually saturated; the centroid-collapse problem
  RIGHT : mean-centered — actual dialect-level structure becomes visible

Output: plots/outputs/03_dialect_radar.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    PLOTS_DIR, REPO_ROOT,
    load_npz, cosine_similarity_matrix,
    ITALO_ROMANCE_DIALECTS, FAMILY_COLORS,
    color_of, display_of,
)


def _radar(ax, labels, values, *, color, title=None,
           rmin=None, rmax=None, fill_alpha=0.22):
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles_c = np.concatenate([angles, [angles[0]]])
    values_c = np.concatenate([values, [values[0]]])

    ax.plot(angles_c, values_c, color=color, linewidth=2.0, zorder=3)
    ax.fill(angles_c, values_c, color=color, alpha=fill_alpha, zorder=2)
    ax.scatter(angles, values, color=color, s=42,
               edgecolor="white", linewidth=1, zorder=4)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=7.8, fontweight="bold")
    for tick, lbl in zip(ax.get_xticklabels(), labels):
        tick.set_color({display_of(c): color_of(c) for c in REF_CODES}.get(lbl, "#444"))

    if rmin is not None and rmax is not None:
        ax.set_ylim(rmin, rmax)
    ax.set_yticklabels([])
    ax.grid(alpha=0.35)
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="bold",
                     color=color, pad=8)


# Will be filled at runtime
REF_CODES: list = []


def _panel(fig, gs_outer, panel_title, sims, dialects, refs, *, mode):
    rows = (len(dialects) + 3) // 4
    cols = min(len(dialects), 4)
    sub_gs = gs_outer.subgridspec(rows + 1, cols,
                                  height_ratios=[0.18] + [1.0] * rows,
                                  hspace=0.38, wspace=0.28)

    head_ax = fig.add_subplot(sub_gs[0, :])
    head_ax.axis("off")
    head_ax.text(0.5, 0.5, panel_title,
                 ha="center", va="center",
                 fontsize=13, fontweight="bold", color="#222222")

    if mode == "raw":
        rmin, rmax = sims.min(), 1.0
        rmin = max(0.0, rmin - 0.05)
    else:  # centered
        amax = max(abs(sims.min()), abs(sims.max()))
        rmin, rmax = -amax * 1.05, amax * 1.05

    ref_labels = [display_of(r) for r in refs]
    for k, dcode in enumerate(dialects):
        r, c = divmod(k, cols)
        ax = fig.add_subplot(sub_gs[r + 1, c], projection="polar")
        _radar(ax, ref_labels, sims[k],
               color=color_of(dcode),
               title=display_of(dcode),
               rmin=rmin, rmax=rmax,
               fill_alpha=0.22 if mode == "raw" else 0.32)
        if mode == "centered":
            ax.axhline(0, color="#888888", linewidth=0.6, alpha=0.6)


def main():
    src = REPO_ROOT / "analysis/multilingual_xlmr/old_experiments/flores/method_outputs/variety_vectors.npz"
    X, codes = load_npz(src)
    sim = cosine_similarity_matrix(X)
    code_to_idx = {c: i for i, c in enumerate(codes)}

    dialects = [c for c in codes if c in ITALO_ROMANCE_DIALECTS]
    refs = [c for c in codes if c not in ITALO_ROMANCE_DIALECTS]
    global REF_CODES
    REF_CODES = refs

    # raw[d_i, r_j] = cos(dialect_i, ref_j)
    raw = np.array([
        [sim[code_to_idx[d], code_to_idx[r]] for r in refs]
        for d in dialects
    ])
    # centered = raw - mean across dialects (per ref)
    centered = raw - raw.mean(axis=0, keepdims=True)

    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(1, 2, wspace=0.10)

    _panel(fig, gs[0], "Raw cosine similarity\n(translation-aware encoder saturates → no signal)",
           raw, dialects, refs, mode="raw")
    _panel(fig, gs[1], "Mean-centered (across dialects)\n(reveals each dialect's actual lean)",
           centered, dialects, refs, mode="centered")

    fig.suptitle(
        "Italo-Romance dialect profiles against reference languages — XLM-R / FLORES\n"
        "Why the new sentence_relations metrics matter: removing the shared content direction reveals real structure",
        fontsize=15, fontweight="bold", y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out = PLOTS_DIR / "03_dialect_radar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
