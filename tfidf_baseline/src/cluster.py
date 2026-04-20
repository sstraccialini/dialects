"""
Hierarchical clustering + silhouette score.

Tools:
- scipy.cluster.hierarchy: linkage + dendrogram
- sklearn.metrics.silhouette_score: cluster coherence vs external labels

External labels:
- family label: fine-grained (italo_romance, italian, romance, germanic,
  greek, semitic, slavic). Same as the 'group' column in config.VARIETIES.
- romance label: binary romance vs non-romance (coarser, easier to
  interpret as a first sanity check).

The silhouette score on a precomputed distance matrix
(metric='precomputed') tells us how close varieties labeled as the same
group are to each other relative to varieties in other groups.
Range: [-1, 1]; >0.2 is good, >0.5 is excellent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # no GUI needed
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
from pathlib import Path
from typing import List

from config import (
    results_subdir,
    VARIETY_GROUP,
    VARIETY_NAMES,
    GROUP_COLORS,
    GROUP_NAMES,
)


ROMANCE_GROUPS = {"italo_romance", "italian", "romance"}


def _family_labels(codes: List[str]) -> List[str]:
    return [VARIETY_GROUP[c] for c in codes]


def _romance_labels(codes: List[str]) -> List[int]:
    return [1 if VARIETY_GROUP[c] in ROMANCE_GROUPS else 0 for c in codes]


def silhouette_report(
    dist: np.ndarray,
    codes: List[str],
) -> dict:
    """Return silhouette scores for both family and romance-vs-rest labels."""
    family = _family_labels(codes)
    romance = _romance_labels(codes)

    sil_family = silhouette_score(dist, family, metric="precomputed")
    sil_romance = silhouette_score(dist, romance, metric="precomputed")

    return {
        "silhouette_family": float(sil_family),
        "silhouette_romance_vs_rest": float(sil_romance),
        "n_varieties": len(codes),
        "family_labels": family,
        "romance_labels": romance,
    }


def hierarchical_linkage(
    dist: np.ndarray,
    method: str = "average",
) -> np.ndarray:
    """
    Build the linkage matrix for hierarchical clustering from a distance
    matrix.

    method='average' is sensible with cosine distance.
    method='ward' requires Euclidean distances; we avoid it by default.
    """
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=method)
    return Z


def plot_dendrogram(
    Z: np.ndarray,
    codes: List[str],
    title: str,
    out_path: Path,
) -> Path:
    """
    Save a dendrogram where leaf labels are the full variety names
    (VARIETY_NAMES), colored by linguistic family.
    """
    leaf_labels = [VARIETY_NAMES[c] for c in codes]
    leaf_colors = {VARIETY_NAMES[c]: GROUP_COLORS.get(VARIETY_GROUP[c], "#333333")
                   for c in codes}

    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    dendrogram(
        Z,
        labels=leaf_labels,
        leaf_rotation=30,
        leaf_font_size=10,
        ax=ax,
        color_threshold=0,
        above_threshold_color="#666666",
    )
    # Color each leaf label according to its family group.
    for tick_label in ax.get_xmajorticklabels():
        text = tick_label.get_text()
        tick_label.set_color(leaf_colors.get(text, "#333333"))
        tick_label.set_fontweight("bold")

    ax.set_title(title)
    ax.set_ylabel("Cosine distance")

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=color,
                   label=GROUP_NAMES.get(group, group))
        for group, color in GROUP_COLORS.items()
    ]
    # Legend always outside the plot (upper-right), identical across MDS,
    # t-SNE and dendrogram. Never overlaps the data.
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=9, frameon=False, title="Family",
              borderaxespad=0.0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def cluster_pipeline(
    dist: np.ndarray,
    codes: List[str],
    pipeline: str,
) -> dict:
    """
    End-to-end wrap: silhouette + linkage + dendrogram.

    `pipeline` is 'word' or 'char'; the dendrogram is saved to
    results/<pipeline>/dendrogram.png.
    """
    report = silhouette_report(dist, codes)
    Z = hierarchical_linkage(dist, method="average")
    dendro_path = results_subdir(pipeline) / "dendrogram.png"
    plot_dendrogram(
        Z, codes,
        title=f"Hierarchical clustering ({pipeline} n-grams, cosine + average linkage)",
        out_path=dendro_path,
    )
    report["dendrogram_path"] = str(dendro_path)
    report["pipeline"] = pipeline
    return report


def save_silhouette_report(reports: List[dict]) -> Path:
    """
    Save a human-readable silhouette report in results/shared/silhouette_report.txt.
    """
    lines = ["Silhouette report", "=" * 60, ""]
    for r in reports:
        lines.append(f"Pipeline: {r['pipeline']}")
        lines.append(f"  n_varieties          = {r['n_varieties']}")
        lines.append(f"  silhouette (family)  = {r['silhouette_family']:+.4f}")
        lines.append(f"  silhouette (romance) = {r['silhouette_romance_vs_rest']:+.4f}")
        lines.append(f"  dendrogram           = {r['dendrogram_path']}")
        lines.append("")
    lines.append("Notes:")
    lines.append("  - family labels: italo_romance, italian, romance, germanic,")
    lines.append("    greek, semitic, slavic (from config.VARIETIES).")
    lines.append("  - romance vs rest: binary, romance = {italo_romance, italian, romance}.")
    lines.append("  - Silhouette range: [-1, 1]. >0.2 good, >0.5 excellent, ~0 no structure.")
    out = results_subdir("shared") / "silhouette_report.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


if __name__ == "__main__":
    from data_loader import load_all_varieties, build_variety_documents
    from vectorize import fit_transform_word, fit_transform_char
    from similarity import cosine_distance_matrix

    print("Loading + vectorizing (may take ~30-60s)...")
    data, _ = load_all_varieties(verbose=False)
    docs, codes = build_variety_documents(data)

    print("Word pipeline...")
    Xw, _ = fit_transform_word(docs)
    Dw = cosine_distance_matrix(Xw)
    rep_w = cluster_pipeline(Dw, codes, "word")

    print("Char pipeline...")
    Xc, _ = fit_transform_char(docs)
    Dc = cosine_distance_matrix(Xc)
    rep_c = cluster_pipeline(Dc, codes, "char")

    path = save_silhouette_report([rep_w, rep_c])
    print(f"\nSaved silhouette report: {path}")
    print(f"  Word: family silhouette = {rep_w['silhouette_family']:+.4f}, "
          f"romance = {rep_w['silhouette_romance_vs_rest']:+.4f}")
    print(f"  Char: family silhouette = {rep_c['silhouette_family']:+.4f}, "
          f"romance = {rep_c['silhouette_romance_vs_rest']:+.4f}")
