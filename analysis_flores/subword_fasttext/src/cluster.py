"""
Hierarchical clustering + silhouette score for the subword pipelines.

Identical semantics to the TF-IDF baseline so results are comparable:
    - average linkage (consistent with cosine distance)
    - silhouette on family labels + romance-vs-rest labels
    - dendrogram with leaves coloured by linguistic family
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

from config import (
    results_subdir,
    VARIETY_GROUP,
    VARIETY_NAMES,
    GROUP_COLORS,
    GROUP_NAMES,
    LINKAGE_METHOD,
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
    """Silhouette on family labels + binary romance-vs-rest labels."""
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
    method: str = LINKAGE_METHOD,
) -> np.ndarray:
    """
    Build linkage matrix for hierarchical clustering.

    'average' is the sensible choice with cosine distance. 'ward' would
    require Euclidean distances; we avoid it.
    """
    condensed = squareform(dist, checks=False)
    return linkage(condensed, method=method)


def plot_dendrogram(
    Z: np.ndarray,
    codes: List[str],
    title: str,
    out_path: Path,
) -> Path:
    """Dendrogram with full English labels, coloured by family."""
    leaf_labels = [VARIETY_NAMES[c] for c in codes]
    leaf_colors = {VARIETY_NAMES[c]: GROUP_COLORS.get(VARIETY_GROUP[c], "#333333")
                   for c in codes}

    fig, ax = plt.subplots(figsize=(13.0, 5.8))
    dendrogram(
        Z,
        labels=leaf_labels,
        leaf_rotation=30,
        leaf_font_size=10,
        ax=ax,
        color_threshold=0,
        above_threshold_color="#666666",
    )
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
    End-to-end wrapper: silhouette + linkage + dendrogram for one pipeline.

    `pipeline` is 'fasttext' or 'bpe'; dendrogram goes to
    results/<pipeline>/dendrogram.png.
    """
    report = silhouette_report(dist, codes)
    Z = hierarchical_linkage(dist)
    dendro_path = results_subdir(pipeline) / "dendrogram.png"
    plot_dendrogram(
        Z, codes,
        title=f"Hierarchical clustering ({pipeline}, cosine + {LINKAGE_METHOD} linkage)",
        out_path=dendro_path,
    )
    report["dendrogram_path"] = str(dendro_path)
    report["pipeline"] = pipeline
    return report


def save_silhouette_report(reports: List[dict]) -> Path:
    """Human-readable silhouette report in results/shared/silhouette_report.txt."""
    lines = ["Silhouette report (subword pipelines)", "=" * 60, ""]
    for r in reports:
        lines.append(f"Pipeline: {r['pipeline']}")
        lines.append(f"  n_varieties          = {r['n_varieties']}")
        lines.append(f"  silhouette (family)  = {r['silhouette_family']:+.4f}")
        lines.append(f"  silhouette (romance) = {r['silhouette_romance_vs_rest']:+.4f}")
        lines.append(f"  dendrogram           = {r['dendrogram_path']}")
        lines.append("")
    lines.append("Notes:")
    lines.append("  - family labels: italo_romance, italian, romance, germanic, english,")
    lines.append("    greek, semitic, slavic (from config.VARIETIES).")
    lines.append("  - romance vs rest: binary, romance = {italo_romance, italian, romance}.")
    lines.append("  - Silhouette range: [-1, 1]. >0.2 good, >0.5 excellent, ~0 no structure.")
    out = results_subdir("shared") / "silhouette_report.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
