"""
Central evaluation entry point for all embedding methods in this project.

Every method (TF-IDF, Word2Vec, FastText, XLM-R, LaBSE, ...) must, at the
end of its pipeline, call ``run_evaluation`` with:

    - ``variety_vectors``  (N, D)   already aggregated per variety
    - ``variety_codes``    list[str] of length N
    - ``out_dir``          Path where evaluation artefacts are written

Artefacts written to ``out_dir``:

    distances.csv                  pairwise cosine distance matrix (N x N)
    similarity_matrix.csv          pairwise cosine similarity matrix (N x N)
    nearest_neighbors.csv          top-k cosine neighbours for each variety
    dendrogram.png                 hierarchical clustering (cosine + linkage)
    similarity_heatmap.png         annotated heatmap sorted by language family
    projection_mds.png             2D MDS projection (precomputed metric)
    projection_tsne.png            2D t-SNE projection (precomputed metric)
    projection_umap.png            2D UMAP projection (if umap-learn installed)
    per_variety_profiles.csv       per-variety ranked distances to all others
    per_variety_plots/<code>.png   bar chart per variety
    family_stats.csv               intra / inter-family distance statistics
    clustering_metrics.csv         DB, CH, ARI, NMI, cophenetic, per-var silhouette
    silhouette_report.txt          extended silhouette scores incl. per-variety

``run_sentence_evaluation`` provides sentence-level analysis when sentence
embeddings (not just variety centroids) are available.

If no taxonomy mapping is passed, plots fall back to plain codes and a
single neutral colour. Every method's ``run_*.py`` should depend ONLY on
this module for evaluation.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Set, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_samples,
    silhouette_score,
    v_measure_score,
)
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

try:
    from umap import UMAP as _UMAP
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #

def _l2_normalise(X: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(X, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return X / norm


def isotropy_correction(X: np.ndarray, top_k_pc: int = 1) -> np.ndarray:
    """
    Mu & Viswanath 2018 "all-but-the-top" post-processing for embedding
    spaces. Removes (i) the global mean, then (ii) the top-K principal
    components of the centered data. Mitigates the well-documented
    anisotropy of transformer-derived embeddings (Ethayarajh 2019).

    Parameters
    ----------
    X         (N, D) input matrix (rows are variety/sentence vectors).
    top_k_pc  Number of leading PCs to remove. Pass 0 for centering only.

    Returns
    -------
    (N, D) corrected matrix, same shape.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    if top_k_pc <= 0:
        return Xc
    n_components = min(top_k_pc, min(Xc.shape) - 1)
    if n_components <= 0:
        return Xc
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = Vt[:n_components]                            # (k, D)
    projection = (Xc @ pcs.T) @ pcs                    # (N, D)
    return Xc - projection


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
# Taxonomy / sorting helpers
# --------------------------------------------------------------------------- #

DEFAULT_COLOR = "#444444"

_FAMILY_ORDER = [
    "italo_romance", "italian", "romance",
    "germanic", "english", "greek", "semitic", "slavic", "_unknown",
]


def _family_sort_key(
    code: str,
    family_groups: Optional[Mapping[str, str]],
) -> Tuple[int, str]:
    fam = family_groups.get(code, "_unknown") if family_groups else "_unknown"
    try:
        fi = _FAMILY_ORDER.index(fam)
    except ValueError:
        fi = len(_FAMILY_ORDER)
    return (fi, code)


def _sorted_codes(
    codes: List[str],
    family_groups: Optional[Mapping[str, str]],
) -> List[str]:
    return sorted(codes, key=lambda c: _family_sort_key(c, family_groups))


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #

def _legend_handles(
    family_colors: Mapping[str, str],
    family_display_names: Mapping[str, str],
    marker: str,
) -> List[plt.Line2D]:
    handles = []
    for family, color in family_colors.items():
        label = family_display_names.get(family, family)
        handles.append(
            plt.Line2D([0], [0], marker=marker, linestyle="",
                       markersize=8, color=color, label=label)
        )
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
        tick_label.set_color(color_by_label.get(tick_label.get_text(), DEFAULT_COLOR))
        tick_label.set_fontweight("bold")
    ax.set_title(title)
    ax.set_ylabel("Cosine distance")
    if family_colors:
        ax.legend(
            handles=_legend_handles(family_colors, family_display_names, "s"),
            loc="upper left", bbox_to_anchor=(1.02, 1.0),
            fontsize=9, frameon=False, title="Family", borderaxespad=0.0,
        )
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
        ax.legend(
            handles=_legend_handles(family_colors, family_display_names, "o"),
            loc="upper left", bbox_to_anchor=(1.02, 1.0),
            fontsize=9, frameon=False, title="Family", borderaxespad=0.0,
        )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# New: similarity heatmap
# --------------------------------------------------------------------------- #

def _plot_similarity_heatmap(
    sim: np.ndarray,
    codes: List[str],
    out_path: Path,
    *,
    title: str,
    leaf_labels: List[str],
    leaf_colors: List[str],
    family_groups: Optional[Mapping[str, str]] = None,
    family_colors: Optional[Mapping[str, str]] = None,
    family_display_names: Optional[Mapping[str, str]] = None,
) -> Path:
    """Annotated cosine-similarity heatmap, rows/columns sorted by language family."""
    n = len(codes)
    if family_groups:
        sorted_c = _sorted_codes(codes, family_groups)
        order = [codes.index(c) for c in sorted_c]
    else:
        order = list(range(n))

    sim_s = sim[np.ix_(order, order)]
    labels_s = [leaf_labels[i] for i in order]
    colors_s = [leaf_colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(9, n * 0.65), max(7, n * 0.55)))
    im = ax.imshow(sim_s, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels_s, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels_s, fontsize=9)
    for i, cl in enumerate(colors_s):
        ax.get_xticklabels()[i].set_color(cl)
        ax.get_yticklabels()[i].set_color(cl)

    fs = 7 if n > 12 else 8
    for i in range(n):
        for j in range(n):
            val = sim_s[i, j]
            tc = "black" if 0.25 < val < 0.85 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=fs, color=tc)

    plt.colorbar(im, ax=ax, label="Cosine similarity", fraction=0.046, pad=0.04)
    ax.set_title(title)
    if family_colors and family_display_names:
        ax.legend(
            handles=_legend_handles(family_colors, dict(family_display_names), "s"),
            loc="upper left", bbox_to_anchor=(1.15, 1.0),
            fontsize=8, frameon=False, title="Family", borderaxespad=0.0,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# New: UMAP projection (optional)
# --------------------------------------------------------------------------- #

def _plot_umap(
    dist: np.ndarray,
    leaf_labels: List[str],
    leaf_colors: List[str],
    out_path: Path,
    *,
    title: str,
    family_colors: Optional[Mapping[str, str]],
    family_display_names: Mapping[str, str],
    n_neighbors: int = 5,
    random_state: int = 42,
) -> Optional[Path]:
    if not _HAS_UMAP:
        warnings.warn("umap-learn not installed; skipping UMAP projection.", stacklevel=3)
        return None
    n = dist.shape[0]
    n_neighbors = max(2, min(n_neighbors, n - 2))
    reducer = _UMAP(
        n_components=2,
        metric="precomputed",
        n_neighbors=n_neighbors,
        min_dist=0.3,
        random_state=random_state,
    )
    coords = reducer.fit_transform(dist)
    return _plot_projection(
        coords, leaf_labels, leaf_colors, out_path,
        title=title,
        family_colors=family_colors,
        family_display_names=family_display_names,
    )


# --------------------------------------------------------------------------- #
# New: per-variety bar charts
# --------------------------------------------------------------------------- #

def _plot_per_variety_bar(
    code: str,
    dist_row: np.ndarray,
    all_codes: List[str],
    out_path: Path,
    *,
    leaf_label: str,
    family_groups: Optional[Mapping[str, str]] = None,
    family_colors: Optional[Mapping[str, str]] = None,
    display_names: Optional[Mapping[str, str]] = None,
) -> Path:
    others_idx = [i for i, c in enumerate(all_codes) if c != code]
    dists = dist_row[others_idx]
    other_codes = [all_codes[i] for i in others_idx]
    order = np.argsort(dists)

    dists_s = dists[order]
    labels_s = [
        (display_names.get(other_codes[i], other_codes[i]) if display_names else other_codes[i])
        for i in order
    ]
    colors_s = []
    for i in order:
        fam = family_groups.get(other_codes[i], "") if family_groups else ""
        colors_s.append(family_colors.get(fam, DEFAULT_COLOR) if family_colors else DEFAULT_COLOR)

    fig, ax = plt.subplots(figsize=(max(8, len(others_idx) * 0.55), 5))
    ax.bar(range(len(dists_s)), dists_s, color=colors_s, edgecolor="white", linewidth=0.5)
    mean_d = float(np.mean(dists_s))
    ax.axhline(mean_d, color="#333333", linestyle="--", linewidth=1.0,
               label=f"mean = {mean_d:.3f}")
    ax.set_xticks(range(len(labels_s)))
    ax.set_xticklabels(labels_s, rotation=40, ha="right", fontsize=9)
    for i, cl in enumerate(colors_s):
        ax.get_xticklabels()[i].set_color(cl)
    ax.set_ylabel("Cosine distance")
    ax.set_title(f"Distance profile: {leaf_label}")
    ax.set_ylim(0.0, max(1.0, float(dists_s.max()) * 1.15))
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# New public analysis functions
# --------------------------------------------------------------------------- #

def per_variety_profiles(
    dist: np.ndarray,
    codes: List[str],
    out_dir: Path,
    *,
    family_groups: Optional[Mapping[str, str]] = None,
    family_colors: Optional[Mapping[str, str]] = None,
    family_display_names: Optional[Mapping[str, str]] = None,
    display_names: Optional[Mapping[str, str]] = None,
) -> Tuple[Path, Path]:
    """
    Per-variety distance profiles: a ranked CSV and one bar-chart PNG per variety.

    Returns (profiles_csv_path, plots_dir_path).
    """
    plots_dir = out_dir / "per_variety_plots"
    plots_dir.mkdir(exist_ok=True)

    rows = []
    for i, src in enumerate(codes):
        for j, tgt in enumerate(codes):
            if i == j:
                continue
            rows.append({
                "source": src,
                "target": tgt,
                "distance": float(dist[i, j]),
                "similarity": float(1.0 - dist[i, j]),
                "target_family": (family_groups.get(tgt, "") if family_groups else ""),
                "target_display": (display_names.get(tgt, tgt) if display_names else tgt),
            })
    df = pd.DataFrame(rows).sort_values(["source", "distance"])
    csv_path = out_dir / "per_variety_profiles.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")

    for i, code in enumerate(codes):
        leaf_lbl = display_names.get(code, code) if display_names else code
        _plot_per_variety_bar(
            code, dist[i], codes, plots_dir / f"{code}.png",
            leaf_label=leaf_lbl,
            family_groups=family_groups,
            family_colors=family_colors,
            display_names=display_names,
        )
    return csv_path, plots_dir


def family_cohesion_stats(
    dist: np.ndarray,
    codes: List[str],
    family_groups: Mapping[str, str],
    out_path: Path,
) -> pd.DataFrame:
    """
    Intra- vs inter-family cosine distance statistics.

    Columns: family, n_members, mean_intra_dist, std_intra_dist,
             mean_inter_dist, std_inter_dist, cohesion_ratio (inter/intra).
    """
    families = sorted(set(family_groups.values()))
    rows = []
    for fam in families:
        members = [i for i, c in enumerate(codes) if family_groups.get(c) == fam]
        non_mem = [i for i, c in enumerate(codes) if family_groups.get(c) != fam]

        if len(members) < 2:
            intra_mean = intra_std = float("nan")
        else:
            pairs = [dist[a, b] for i, a in enumerate(members) for b in members[i + 1:]]
            intra_mean = float(np.mean(pairs))
            intra_std = float(np.std(pairs))

        if members and non_mem:
            cross = [dist[a, b] for a in members for b in non_mem]
            inter_mean = float(np.mean(cross))
            inter_std = float(np.std(cross))
        else:
            inter_mean = inter_std = float("nan")

        cohesion = (
            inter_mean / intra_mean
            if (np.isfinite(intra_mean) and intra_mean > 0)
            else float("nan")
        )
        rows.append({
            "family": fam,
            "n_members": len(members),
            "mean_intra_dist": intra_mean,
            "std_intra_dist": intra_std,
            "mean_inter_dist": inter_mean,
            "std_inter_dist": inter_std,
            "cohesion_ratio": cohesion,
        })
    df = pd.DataFrame(rows).sort_values("cohesion_ratio", ascending=False)
    df.to_csv(out_path, index=False, float_format="%.6f")
    return df


def clustering_metrics_table(
    X: np.ndarray,
    dist: np.ndarray,
    codes: List[str],
    family_groups: Mapping[str, str],
    Z: np.ndarray,
    out_path: Path,
) -> pd.DataFrame:
    """
    Extended clustering metrics table.

    Global metrics: Davies-Bouldin, Calinski-Harabasz, cophenetic correlation,
    ARI / NMI / V-measure vs family labels (hierarchical cut and K-Means).
    Per-variety: silhouette score from precomputed distance matrix.

    Parameters
    ----------
    X    : (N, D) L2-normalised variety vectors  — needed for DB and CH
    dist : (N, N) precomputed cosine distance    — needed for silhouette, cophenetic
    Z    : linkage matrix from scipy             — needed for cophenetic + dendrogram cut
    """
    n = len(codes)
    family_labels = [family_groups.get(c, "_unknown") for c in codes]
    k = len(set(family_labels))

    # Per-variety silhouette (average over sentences belonging to each variety)
    if k > 1:
        sil_arr = silhouette_samples(dist, family_labels, metric="precomputed")
    else:
        sil_arr = np.zeros(n)

    per_var_df = pd.DataFrame({
        "variety": codes,
        "family": family_labels,
        "silhouette": sil_arr.tolist(),
    }).sort_values("silhouette", ascending=False)

    # Global metrics on raw X
    try:
        db = float(davies_bouldin_score(X, family_labels))
    except Exception:
        db = float("nan")
    try:
        ch = float(calinski_harabasz_score(X, family_labels))
    except Exception:
        ch = float("nan")

    # Cophenetic correlation
    try:
        coph_r = float(cophenet(Z, squareform(dist, checks=False))[0])
    except Exception:
        coph_r = float("nan")

    # Predicted labels: hierarchical cut at k clusters
    hc_labels = fcluster(Z, k, criterion="maxclust").tolist()
    ari_hc = float(adjusted_rand_score(family_labels, hc_labels))
    nmi_hc = float(normalized_mutual_info_score(family_labels, hc_labels))
    vm_hc = float(v_measure_score(family_labels, hc_labels))

    # Predicted labels: K-Means on raw X
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    km_labels = km.fit_predict(X).tolist()
    ari_km = float(adjusted_rand_score(family_labels, km_labels))
    nmi_km = float(normalized_mutual_info_score(family_labels, km_labels))
    vm_km = float(v_measure_score(family_labels, km_labels))

    global_rows = [
        {"metric": "silhouette_global",       "value": float(np.mean(sil_arr)), "clusterer": "family_labels"},
        {"metric": "davies_bouldin",           "value": db,                      "clusterer": "family_labels"},
        {"metric": "calinski_harabasz",        "value": ch,                      "clusterer": "family_labels"},
        {"metric": "cophenetic_correlation",   "value": coph_r,                  "clusterer": "hierarchical"},
        {"metric": "adjusted_rand_index",      "value": ari_hc,                  "clusterer": "hierarchical_cut"},
        {"metric": "normalized_mutual_info",   "value": nmi_hc,                  "clusterer": "hierarchical_cut"},
        {"metric": "v_measure",                "value": vm_hc,                   "clusterer": "hierarchical_cut"},
        {"metric": "adjusted_rand_index",      "value": ari_km,                  "clusterer": "kmeans"},
        {"metric": "normalized_mutual_info",   "value": nmi_km,                  "clusterer": "kmeans"},
        {"metric": "v_measure",                "value": vm_km,                   "clusterer": "kmeans"},
    ]
    global_df = pd.DataFrame(global_rows)

    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("# Global clustering metrics\n")
        global_df.to_csv(fh, index=False, float_format="%.6f")
        fh.write("\n# Per-variety silhouette scores\n")
        per_var_df.to_csv(fh, index=False, float_format="%.6f")

    return global_df


# --------------------------------------------------------------------------- #
# Sentence-level evaluation
# --------------------------------------------------------------------------- #

def run_sentence_evaluation(
    sentence_vectors: np.ndarray,
    sentence_labels: List[str],
    out_dir: Union[str, Path],
    *,
    method_label: str = "",
    family_groups: Optional[Mapping[str, str]] = None,
    family_colors: Optional[Mapping[str, str]] = None,
    family_display_names: Optional[Mapping[str, str]] = None,
    display_names: Optional[Mapping[str, str]] = None,
    romance_families: Optional[Set[str]] = None,
    normalise: bool = True,
    isotropy: bool = False,
    isotropy_top_k_pc: int = 1,
    random_state: int = 42,
    n_sample: Optional[int] = 5000,
    tsne_perplexity: float = 30.0,
    umap_n_neighbors: int = 15,
) -> Dict[str, object]:
    """
    Sentence-level evaluation: within/between-variety distances, silhouette,
    and UMAP / t-SNE projections of sentence embeddings.

    Parameters
    ----------
    sentence_vectors : (M, D) array — one row per sentence
    sentence_labels  : length-M list of variety codes (one per row)
    n_sample         : subsample for silhouette and projections (large M is slow)

    Outputs (in out_dir)
    --------------------
    sentence_within_between.csv      mean within/between cosine distance per variety
    sentence_silhouette_report.txt   silhouette at sentence level
    sentence_projection_tsne.png     t-SNE of (subsampled) sentence embeddings
    sentence_projection_umap.png     UMAP  (if umap-learn installed)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    V = np.asarray(sentence_vectors, dtype=np.float32)
    labels = list(sentence_labels)
    if isotropy:
        V = isotropy_correction(V, top_k_pc=isotropy_top_k_pc)
    if normalise:
        V = _l2_normalise(V)

    varieties = sorted(set(labels))
    label_arr = np.array(labels)

    # ---- within / between distances ----------------------------------------
    rows = []
    for var in varieties:
        idx = np.where(label_arr == var)[0]
        others_idx = np.where(label_arr != var)[0]

        if len(idx) < 2:
            within = float("nan")
        else:
            sub = V[idx]
            sim_sub = cosine_similarity(sub)
            ni = len(idx)
            tri = sim_sub[np.triu_indices(ni, k=1)]
            within = float(np.mean(1.0 - tri))

        between = (
            float(np.mean(1.0 - cosine_similarity(V[idx], V[others_idx])))
            if (len(idx) and len(others_idx))
            else float("nan")
        )
        sep = between / within if (np.isfinite(within) and within > 0) else float("nan")
        rows.append({
            "variety": var,
            "family": (family_groups.get(var, "") if family_groups else ""),
            "n_sentences": int(len(idx)),
            "mean_within_dist": within,
            "mean_between_dist": between,
            "separation_ratio": sep,
        })

    wb_df = pd.DataFrame(rows)
    wb_path = out_dir / "sentence_within_between.csv"
    wb_df.to_csv(wb_path, index=False, float_format="%.6f")

    # ---- subsample for silhouette / projections ----------------------------
    M = len(labels)
    if n_sample and M > n_sample:
        rng = np.random.default_rng(random_state)
        sub_idx = rng.choice(M, size=n_sample, replace=False)
        V_sub = V[sub_idx]
        labels_sub = [labels[i] for i in sub_idx]
    else:
        V_sub = V
        labels_sub = labels

    # ---- silhouette --------------------------------------------------------
    sil_sent: Optional[float] = None
    if len(set(labels_sub)) > 1:
        dist_sub = cosine_distances(V_sub)
        sil_sent = float(silhouette_score(dist_sub, labels_sub, metric="precomputed"))

    fam_disp = dict(family_display_names) if family_display_names else {}

    def _leaf_style(labs):
        lbl = [(display_names.get(c, c) if display_names else c) for c in labs]
        clr = []
        for c in labs:
            fam = family_groups.get(c, "") if family_groups else ""
            clr.append(family_colors.get(fam, DEFAULT_COLOR) if family_colors else DEFAULT_COLOR)
        return lbl, clr

    # ---- t-SNE projection --------------------------------------------------
    n_sub = len(labels_sub)
    perp = max(2.0, min(tsne_perplexity, (n_sub - 1) / 3.0))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=random_state, max_iter=1000)
    coords_tsne = tsne.fit_transform(V_sub)
    lbl_sub, clr_sub = _leaf_style(labels_sub)
    tsne_path = out_dir / "sentence_projection_tsne.png"
    _plot_projection(
        coords_tsne, lbl_sub, clr_sub, tsne_path,
        title=f"Sentence t-SNE (perp={perp:g}){f' — {method_label}' if method_label else ''}",
        family_colors=family_colors,
        family_display_names=fam_disp,
    )

    # ---- UMAP projection ---------------------------------------------------
    umap_path: Optional[Path] = None
    if _HAS_UMAP and n_sub >= 4:
        n_nbrs = max(2, min(umap_n_neighbors, n_sub - 2))
        reducer = _UMAP(n_components=2, n_neighbors=n_nbrs, min_dist=0.1,
                        random_state=random_state)
        coords_umap = reducer.fit_transform(V_sub)
        umap_path = out_dir / "sentence_projection_umap.png"
        _plot_projection(
            coords_umap, lbl_sub, clr_sub, umap_path,
            title=f"Sentence UMAP{f' — {method_label}' if method_label else ''}",
            family_colors=family_colors,
            family_display_names=fam_disp,
        )

    # ---- report ------------------------------------------------------------
    report_path = out_dir / "sentence_silhouette_report.txt"
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write(f"Sentence-level evaluation report ({method_label})\n")
        fh.write("=" * 60 + "\n")
        fh.write(f"  total sentences       = {M}\n")
        fh.write(f"  varieties             = {len(varieties)}\n")
        fh.write(f"  subsample for sil.    = {len(labels_sub)}\n")
        fh.write(f"  silhouette (variety)  = "
                 f"{f'{sil_sent:+.4f}' if sil_sent is not None else 'n/a'}\n\n")
        fh.write("Within / between distances per variety:\n")
        fh.write(wb_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        fh.write("\n")

    return {
        "within_between_path": str(wb_path),
        "silhouette_sentence": sil_sent,
        "sentence_report_path": str(report_path),
        "sentence_tsne_path": str(tsne_path),
        "sentence_umap_path": str(umap_path) if umap_path else None,
    }


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
    isotropy: bool = False,
    isotropy_top_k_pc: int = 1,
) -> Dict[str, object]:
    """
    Compute and dump all evaluation artefacts for a method.

    Parameters
    ----------
    variety_vectors
        (N, D) per-variety representations.
    variety_codes
        List of length N with stable variety identifiers.
    out_dir
        Directory where artefacts are written; created if missing.
    method_label
        Human-readable name appended to plot titles (e.g. "TF-IDF char").
    family_groups
        Optional mapping ``code -> family``. Required for silhouette-by-family
        and for coloured plots.
    family_colors
        Optional mapping ``family -> hex color``.
    family_display_names
        Optional mapping ``family -> pretty name`` for the legend.
    display_names
        Optional mapping ``code -> pretty name`` for plot leaf labels.
    romance_families
        Set of family ids treated as "Romance" for the binary romance-vs-rest
        silhouette. Skipped when ``None``.
    linkage_method
        Hierarchical linkage method (default ``"average"``).
    tsne_perplexity
        t-SNE perplexity. Small datasets (~16 varieties) need small values.
    nearest_k
        Number of nearest neighbours saved per row.
    random_state
        Seed for MDS / t-SNE / UMAP reproducibility.
    normalise
        L2-normalise rows before computing cosine distances (recommended).
    isotropy
        If True, apply Mu & Viswanath (2018) "all-but-the-top" post-processing
        before distance computation: subtract the global mean and remove the
        top-K principal components. Mitigates anisotropy of transformer-
        derived embeddings (Ethayarajh, 2019). Default False to preserve
        backward-compat with existing runs; turn on for any pretrained
        encoder evaluation (XLM-R, CANINE, Sentence-MiniLM, LaBSE).
    isotropy_top_k_pc
        Number of leading principal components to remove if ``isotropy=True``.
        1 is the standard choice. Larger values (2-3) over-correct.

    Returns
    -------
    dict
        Paths and scalar metrics for all artefacts produced.
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
    if isotropy:
        X = isotropy_correction(X, top_k_pc=isotropy_top_k_pc)
    if normalise:
        X = _l2_normalise(X)

    codes = list(variety_codes)
    n = len(codes)

    leaf_labels = [(display_names.get(c, c) if display_names else c) for c in codes]
    if family_groups and family_colors:
        leaf_colors = [
            family_colors.get(family_groups.get(c, ""), DEFAULT_COLOR) for c in codes
        ]
    else:
        leaf_colors = [DEFAULT_COLOR] * n

    fam_disp = dict(family_display_names) if family_display_names else {}

    # ---- distances & similarity -------------------------------------------
    dist = cosine_distance_matrix(X)
    dist_path = out_dir / "distances.csv"
    pd.DataFrame(dist, index=codes, columns=codes).to_csv(dist_path, float_format="%.6f")

    sim = 1.0 - dist
    sim_path = out_dir / "similarity_matrix.csv"
    pd.DataFrame(sim, index=codes, columns=codes).to_csv(sim_path, float_format="%.6f")

    # ---- nearest neighbours -----------------------------------------------
    nn_path = out_dir / "nearest_neighbors.csv"
    nearest_neighbors_table(dist, codes, k=nearest_k).to_csv(nn_path, index=False)

    # ---- similarity heatmap -----------------------------------------------
    heatmap_path = _plot_similarity_heatmap(
        sim, codes, out_dir / "similarity_heatmap.png",
        title=f"Cosine similarity{f' — {method_label}' if method_label else ''}",
        leaf_labels=leaf_labels,
        leaf_colors=leaf_colors,
        family_groups=family_groups,
        family_colors=family_colors,
        family_display_names=fam_disp,
    )

    # ---- linkage + dendrogram ---------------------------------------------
    Z = linkage(squareform(dist, checks=False), method=linkage_method)
    dendro_path = _plot_dendrogram(
        Z, codes, out_dir / "dendrogram.png",
        title=(f"Hierarchical clustering{f' — {method_label}' if method_label else ''}"
               f" (cosine + {linkage_method})"),
        leaf_labels=leaf_labels,
        leaf_colors=leaf_colors,
        family_colors=family_colors,
        family_display_names=fam_disp,
    )

    # ---- 2D projections ---------------------------------------------------
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
        title=(f"t-SNE projection (perp={perplexity:g})"
               f"{f' — {method_label}' if method_label else ''}"),
        family_colors=family_colors, family_display_names=fam_disp,
    )

    # ---- UMAP (optional) --------------------------------------------------
    umap_path = _plot_umap(
        dist, leaf_labels, leaf_colors, out_dir / "projection_umap.png",
        title=f"UMAP projection{f' — {method_label}' if method_label else ''}",
        family_colors=family_colors, family_display_names=fam_disp,
        n_neighbors=max(2, n // 3),
        random_state=random_state,
    )

    # ---- per-variety profiles ---------------------------------------------
    profiles_path, plots_dir = per_variety_profiles(
        dist, codes, out_dir,
        family_groups=family_groups,
        family_colors=family_colors,
        family_display_names=fam_disp,
        display_names=display_names,
    )

    # ---- family cohesion stats & clustering metrics -----------------------
    fam_stats_path: Optional[Path] = None
    cluster_path: Optional[Path] = None
    if family_groups and len(set(family_groups.get(c, "") for c in codes)) > 1:
        fam_stats_path = out_dir / "family_stats.csv"
        family_cohesion_stats(dist, codes, family_groups, fam_stats_path)

        cluster_path = out_dir / "clustering_metrics.csv"
        try:
            clustering_metrics_table(X, dist, codes, family_groups, Z, cluster_path)
        except Exception as exc:
            warnings.warn(f"clustering_metrics_table skipped: {exc}", stacklevel=2)
            cluster_path = None

    # ---- silhouette -------------------------------------------------------
    sil_family: Optional[float] = None
    sil_romance: Optional[float] = None
    sil_samples_arr = None
    labels_family = None

    if family_groups is not None:
        labels_family = [family_groups.get(c, "_unknown") for c in codes]
        if len(set(labels_family)) > 1:
            sil_family = float(silhouette_score(dist, labels_family, metric="precomputed"))
            sil_samples_arr = silhouette_samples(dist, labels_family, metric="precomputed")
        if romance_families:
            labels_rom = [1 if family_groups.get(c) in romance_families else 0 for c in codes]
            if len(set(labels_rom)) > 1:
                sil_romance = float(silhouette_score(dist, labels_rom, metric="precomputed"))

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
        fh.write("  Silhouette range [-1, 1].  >0.5 excellent, >0.2 good, ~0 no structure.\n")
        if sil_samples_arr is not None and labels_family is not None:
            fh.write("\nPer-variety silhouette scores (sorted descending):\n")
            fh.write(f"  {'variety':<20}  {'family':<16}  {'silhouette':>10}\n")
            fh.write("  " + "-" * 52 + "\n")
            for idx in np.argsort(sil_samples_arr)[::-1]:
                fh.write(
                    f"  {codes[idx]:<20}  {labels_family[idx]:<16}"
                    f"  {sil_samples_arr[idx]:>+10.4f}\n"
                )

    # ---- gold-matrix correlations -----------------------------------------
    # If gold matrices are present under gold/<family>/matrices/ at the
    # repo root, write a small per-experiment correlations CSV next to
    # the silhouette report.  Failures are swallowed so they cannot break
    # any existing run.py.
    gold_corr_path: Optional[Path] = None
    try:
        from evaluation._gold_correlation import write_per_experiment_gold_csv
        gold_corr_path = out_dir / "gold_correlations.csv"
        df_gold = write_per_experiment_gold_csv(
            gold_corr_path, dist, codes,
        )
        if df_gold is None:
            gold_corr_path = None
    except Exception as exc:
        warnings.warn(f"gold-correlation write skipped: {exc}", stacklevel=2)
        gold_corr_path = None

    return {
        "distances_path": str(dist_path),
        "similarity_path": str(sim_path),
        "nearest_neighbors_path": str(nn_path),
        "similarity_heatmap_path": str(heatmap_path),
        "dendrogram_path": str(dendro_path),
        "projection_mds_path": str(mds_path),
        "projection_tsne_path": str(tsne_path),
        "projection_umap_path": str(umap_path) if umap_path else None,
        "per_variety_profiles_path": str(profiles_path),
        "per_variety_plots_dir": str(plots_dir),
        "family_stats_path": str(fam_stats_path) if fam_stats_path else None,
        "clustering_metrics_path": str(cluster_path) if cluster_path else None,
        "silhouette_path": str(sil_path),
        "gold_correlations_path": str(gold_corr_path) if gold_corr_path else None,
        "silhouette_family": sil_family,
        "silhouette_romance_vs_rest": sil_romance,
        "n_varieties": n,
        "method_label": method_label,
    }
