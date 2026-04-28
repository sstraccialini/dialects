"""
Central evaluation entry point for all embedding methods in this project.

Every method (TF-IDF, Word2Vec, FastText, XLM-R, LaBSE, ...) must, at the
end of its pipeline, call ``run_evaluation`` with:

    - ``variety_vectors``  (N, D)   already aggregated per variety
    - ``variety_codes``    list[str] of length N
    - ``out_dir``          Path where evaluation artefacts are written

What we produce (always, in ``out_dir``):

    distances.csv             pairwise cosine distance matrix (N x N)
    nearest_neighbors.csv     top-k cosine neighbours for each variety
    dendrogram.png            hierarchical clustering (cosine + linkage)
    projection_mds.png        2D MDS projection (precomputed metric)
    projection_tsne.png       2D t-SNE projection (precomputed metric)
    silhouette_report.txt     silhouette scores (when family labels given)

If no taxonomy mapping is passed, plots fall back to plain codes and a
single neutral colour. Every method's ``run_*.py`` should depend ONLY on
this module for evaluation; the underlying logic lives here so that a
future change in the analysis (different linkage, new metrics, alternative
projections, ...) requires editing this single file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Set, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _l2_normalise(X: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(X, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return X / norm


def cosine_distance_matrix(X: np.ndarray) -> np.ndarray:
    """Symmetric, zero-diagonal cosine distance matrix in [0, 2]."""
    sim = cosine_similarity(X)
    sim = (sim + sim.T) / 2.0
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    np.clip(dist, 0.0, None, out=dist)
    return dist


def nearest_neighbors_table(
    dist: np.ndarray,
    codes: List[str],
    k: int = 3,
) -> pd.DataFrame:
    n = dist.shape[0]
    rows = []
    for i in range(n):
        d = dist[i].copy()
        d[i] = np.inf
        order = np.argsort(d)
        row = {"code": codes[i]}
        for rank, idx in enumerate(order[: min(k, n - 1)], start=1):
            row[f"nn_{rank}"] = codes[idx]
            row[f"dist_{rank}"] = float(dist[i, idx])
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #
DEFAULT_COLOR = "#444444"


def _legend_handles(family_colors: Mapping[str, str],
                    family_display_names: Mapping[str, str],
                    marker: str) -> List[plt.Line2D]:
    handles = []
    for family, color in family_colors.items():
        label = family_display_names.get(family, family)
        handles.append(plt.Line2D([0], [0], marker=marker, linestyle="",
                                  markersize=8, color=color, label=label))
    return handles


def _plot_dendrogram(
    Z: np.ndarray,
    codes: List[str],
    out_path: Path,
    *,
    title: str,
    leaf_labels: List[str],
    leaf_colors: List[str],
    family_colors: Optional[Mapping[str, str]],
    family_display_names: Mapping[str, str],
) -> Path:
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
    color_by_label = dict(zip(leaf_labels, leaf_colors))
    for tick_label in ax.get_xmajorticklabels():
        text = tick_label.get_text()
        tick_label.set_color(color_by_label.get(text, DEFAULT_COLOR))
        tick_label.set_fontweight("bold")
    ax.set_title(title)
    ax.set_ylabel("Cosine distance")
    if family_colors:
        ax.legend(handles=_legend_handles(family_colors, family_display_names, "s"),
                  loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  fontsize=9, frameon=False, title="Family",
                  borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_projection(
    coords: np.ndarray,
    leaf_labels: List[str],
    leaf_colors: List[str],
    out_path: Path,
    *,
    title: str,
    family_colors: Optional[Mapping[str, str]],
    family_display_names: Mapping[str, str],
) -> Path:
    fig, ax = plt.subplots(figsize=(11.0, 6.8))
    ax.scatter(coords[:, 0], coords[:, 1], c=leaf_colors, s=130,
               edgecolor="white", linewidth=1.0, zorder=2)
    for i, name in enumerate(leaf_labels):
        ax.annotate(
            name,
            (coords[i, 0], coords[i, 1]),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            color=leaf_colors[i],
        )
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    if family_colors:
        ax.legend(handles=_legend_handles(family_colors, family_display_names, "o"),
                  loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  fontsize=9, frameon=False, title="Family",
                  borderaxespad=0.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def run_evaluation(
    variety_vectors: np.ndarray,
    variety_codes: List[str],
    out_dir: Union[str, Path],
    *,
    method_label: str = "",
    family_groups: Optional[Mapping[str, str]] = None,
    family_colors: Optional[Mapping[str, str]] = None,
    family_display_names: Optional[Mapping[str, str]] = None,
    display_names: Optional[Mapping[str, str]] = None,
    romance_families: Optional[Set[str]] = None,
    linkage_method: str = "average",
    tsne_perplexity: float = 4.0,
    nearest_k: int = 3,
    random_state: int = 42,
    normalise: bool = True,
) -> Dict[str, object]:
    """
    Compute and dump all evaluation artefacts for a method.

    Parameters
    ----------
    variety_vectors
        (N, D) per-variety representations.
    variety_codes
        List of length N with stable variety identifiers (e.g. ISO codes
        or short names). Both vectors and codes must share the same order.
    out_dir
        Directory where artefacts are written; created if missing.
    method_label
        Human-readable name appended to plot titles (e.g. "TF-IDF char").
    family_groups
        Optional mapping ``code -> family``. Required for silhouette-by-family
        and for coloured plots. If not provided, plots use a neutral colour.
    family_colors
        Optional mapping ``family -> hex color``. Defaults to a single-tone
        plot when omitted.
    family_display_names
        Optional mapping ``family -> pretty name`` for the legend.
    display_names
        Optional mapping ``code -> pretty name`` for plot leaf labels;
        defaults to the code itself.
    romance_families
        Set of family ids treated as "Romance" for the binary
        romance-vs-rest silhouette. Skipped when ``None``.
    linkage_method
        Hierarchical linkage method (default ``"average"``, coherent with
        cosine distances).
    tsne_perplexity
        t-SNE perplexity. Small datasets (~16 varieties) need small values.
    nearest_k
        Number of nearest neighbours saved per row.
    random_state
        Seed for MDS / t-SNE reproducibility.
    normalise
        L2-normalise rows before computing cosine distances (recommended).

    Returns
    -------
    dict
        ``{path_*: str, silhouette_*: float | None, n_varieties: int}``
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.asarray(variety_vectors, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"variety_vectors must be 2D, got shape {X.shape}")
    if len(variety_codes) != X.shape[0]:
        raise ValueError(
            f"variety_codes length {len(variety_codes)} != n_rows {X.shape[0]}"
        )
    if normalise:
        X = _l2_normalise(X)

    codes = list(variety_codes)
    n = len(codes)

    leaf_labels = [
        (display_names.get(c, c) if display_names else c) for c in codes
    ]
    if family_groups and family_colors:
        leaf_colors = [
            family_colors.get(family_groups.get(c, ""), DEFAULT_COLOR)
            for c in codes
        ]
    else:
        leaf_colors = [DEFAULT_COLOR] * n

    fam_disp = dict(family_display_names) if family_display_names else {}

    # -- distances --
    dist = cosine_distance_matrix(X)
    dist_path = out_dir / "distances.csv"
    pd.DataFrame(dist, index=codes, columns=codes).to_csv(
        dist_path, float_format="%.6f"
    )

    # -- nearest neighbours --
    nn_path = out_dir / "nearest_neighbors.csv"
    nearest_neighbors_table(dist, codes, k=nearest_k).to_csv(
        nn_path, index=False
    )

    # -- silhouette --
    sil_family: Optional[float] = None
    sil_romance: Optional[float] = None
    if family_groups is not None:
        labels_family = [family_groups.get(c, "_unknown") for c in codes]
        if len(set(labels_family)) > 1:
            sil_family = float(silhouette_score(dist, labels_family,
                                                metric="precomputed"))
        if romance_families:
            labels_rom = [
                1 if family_groups.get(c) in romance_families else 0
                for c in codes
            ]
            if len(set(labels_rom)) > 1:
                sil_romance = float(silhouette_score(dist, labels_rom,
                                                    metric="precomputed"))

    sil_path = out_dir / "silhouette_report.txt"
    with sil_path.open("w", encoding="utf-8") as fh:
        fh.write(f"Silhouette report ({method_label})\n")
        fh.write("=" * 60 + "\n")
        fh.write(f"  n_varieties           = {n}\n")
        fh.write(f"  linkage_method        = {linkage_method}\n")
        fh.write(f"  silhouette (family)   = "
                 f"{f'{sil_family:+.4f}' if sil_family is not None else 'n/a'}\n")
        fh.write(f"  silhouette (romance)  = "
                 f"{f'{sil_romance:+.4f}' if sil_romance is not None else 'n/a'}\n")
        fh.write("\nNotes:\n")
        fh.write("  Silhouette range [-1, 1]. >0.2 good, >0.5 excellent, ~0 no structure.\n")

    # -- linkage + dendrogram --
    Z = linkage(squareform(dist, checks=False), method=linkage_method)
    title_dendro = (f"Hierarchical clustering"
                    f"{f' — {method_label}' if method_label else ''}"
                    f" (cosine + {linkage_method})")
    dendro_path = _plot_dendrogram(
        Z, codes, out_dir / "dendrogram.png",
        title=title_dendro,
        leaf_labels=leaf_labels,
        leaf_colors=leaf_colors,
        family_colors=family_colors,
        family_display_names=fam_disp,
    )

    # -- 2D projections --
    mds = MDS(n_components=2, dissimilarity="precomputed",
              random_state=random_state, normalized_stress="auto")
    coords_mds = mds.fit_transform(dist)
    mds_path = _plot_projection(
        coords_mds, leaf_labels, leaf_colors, out_dir / "projection_mds.png",
        title=f"MDS projection{f' — {method_label}' if method_label else ''}",
        family_colors=family_colors, family_display_names=fam_disp,
    )

    perplexity = max(2.0, min(tsne_perplexity, (n - 1) / 3.0))
    tsne = TSNE(n_components=2, metric="precomputed", init="random",
                perplexity=perplexity, random_state=random_state, max_iter=2000)
    coords_tsne = tsne.fit_transform(dist)
    tsne_path = _plot_projection(
        coords_tsne, leaf_labels, leaf_colors, out_dir / "projection_tsne.png",
        title=f"t-SNE projection (perp={perplexity:g})"
              f"{f' — {method_label}' if method_label else ''}",
        family_colors=family_colors, family_display_names=fam_disp,
    )

    return {
        "distances_path": str(dist_path),
        "nearest_neighbors_path": str(nn_path),
        "silhouette_path": str(sil_path),
        "dendrogram_path": str(dendro_path),
        "projection_mds_path": str(mds_path),
        "projection_tsne_path": str(tsne_path),
        "silhouette_family": sil_family,
        "silhouette_romance_vs_rest": sil_romance,
        "n_varieties": n,
        "method_label": method_label,
    }
