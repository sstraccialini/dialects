"""
Parallel-corpus alignment analysis for FLORES+ and similar aligned datasets.

When all varieties share sentence-aligned text (FLORES+ has 2009 such sentences),
we can compute the mean cosine similarity between sentence i of variety A and
sentence i of variety B — a direct cross-lingual alignment score that is
independent of centroid distance.

This complements the centroid-based evaluation in ``evaluation.py`` by measuring
how well individual sentence translations overlap in the embedding space.

Usage
-----
    import numpy as np
    from evaluation.parallel_alignment import run_parallel_alignment

    # sentence_vecs[code] is shape (N_sentences, D), same N for every code
    sentence_vecs = {
        "veneto":   np.load("...veneto_sent.npy"),    # (2009, 768)
        "italiano": np.load("...italiano_sent.npy"),
        "inglese":  np.load("...inglese_sent.npy"),
        ...
    }
    run_parallel_alignment(
        sentence_vecs,
        out_dir="analysis/comparison/parallel_flores",
        method_label="XLM-R (FLORES+)",
        family_groups=...,
        family_colors=...,
    )

Outputs
-------
    parallel_alignment.csv              NxN mean sentence-pair cosine similarity
    parallel_alignment_heatmap.png      heatmap sorted by family
    parallel_alignment_pairs.csv        all variety pairs ranked by similarity
    parallel_alignment_report.txt       summary + per-variety stats
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

_DEFAULT_FAMILY_ORDER = [
    "italo_romance", "italian", "romance",
    "germanic", "english", "greek", "semitic", "slavic",
]
_DEFAULT_COLOR = "#444444"


def _l2_normalise(X: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / np.where(norm == 0, 1.0, norm)


def _sort_by_family(
    codes: List[str],
    family_groups: Optional[Mapping[str, str]],
) -> List[str]:
    order = {f: i for i, f in enumerate(_DEFAULT_FAMILY_ORDER)}
    return sorted(
        codes,
        key=lambda c: (order.get(family_groups.get(c, ""), 99) if family_groups else 0, c),
    )


def _legend_handles(
    family_colors: Mapping[str, str],
    family_display_names: Mapping[str, str],
) -> List[plt.Line2D]:
    return [
        plt.Line2D([0], [0], marker="s", linestyle="",
                   markersize=8, color=color,
                   label=family_display_names.get(fam, fam))
        for fam, color in family_colors.items()
    ]


def _heatmap(
    matrix: np.ndarray,
    codes: List[str],
    out_path: Path,
    *,
    title: str,
    cmap: str = "RdYlGn",
    vmin: float = 0.0,
    vmax: float = 1.0,
    colorbar_label: str = "Mean cosine similarity",
    family_groups: Optional[Mapping[str, str]] = None,
    family_colors: Optional[Mapping[str, str]] = None,
    family_display_names: Optional[Mapping[str, str]] = None,
    display_names: Optional[Mapping[str, str]] = None,
) -> Path:
    n = len(codes)
    if family_groups:
        sorted_c = _sort_by_family(codes, family_groups)
        order = [codes.index(c) for c in sorted_c]
    else:
        sorted_c = codes
        order = list(range(n))

    mat_s = matrix[np.ix_(order, order)]
    labels_s = [
        (display_names.get(sorted_c[i], sorted_c[i]) if display_names else sorted_c[i])
        for i in range(n)
    ]
    colors_s = []
    for c in sorted_c:
        fam = family_groups.get(c, "") if family_groups else ""
        colors_s.append(family_colors.get(fam, _DEFAULT_COLOR) if family_colors else _DEFAULT_COLOR)

    fig, ax = plt.subplots(figsize=(max(9, n * 0.65), max(7, n * 0.55)))
    im = ax.imshow(mat_s, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

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
            val = mat_s[i, j]
            if np.isfinite(val):
                tc = "black" if 0.25 < val < 0.85 else "white"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=fs, color=tc)

    plt.colorbar(im, ax=ax, label=colorbar_label, fraction=0.046, pad=0.04)
    ax.set_title(title)
    if family_colors and family_display_names:
        ax.legend(
            handles=_legend_handles(family_colors, dict(family_display_names)),
            loc="upper left", bbox_to_anchor=(1.15, 1.0),
            fontsize=8, frameon=False, title="Family", borderaxespad=0.0,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def run_parallel_alignment(
    sentence_vectors: Dict[str, np.ndarray],
    out_dir: Union[str, Path],
    *,
    method_label: str = "",
    family_groups: Optional[Mapping[str, str]] = None,
    family_colors: Optional[Mapping[str, str]] = None,
    family_display_names: Optional[Mapping[str, str]] = None,
    display_names: Optional[Mapping[str, str]] = None,
    normalise: bool = True,
) -> Dict[str, object]:
    """
    Sentence-pair cosine similarity analysis for a parallel corpus.

    For each variety pair (A, B) computes the mean cosine similarity between
    aligned sentence i in A and sentence i in B.  High values indicate that
    the model places translations close together in embedding space.

    Parameters
    ----------
    sentence_vectors
        Dict mapping variety code -> (N, D) float array.
        ALL arrays must have the same N (aligned sentences).
    out_dir
        Directory for output artefacts (created if missing).
    method_label
        Shown in plot titles (e.g. "XLM-R (FLORES+)").
    family_groups / family_colors / family_display_names / display_names
        Standard taxonomy mappings, same as in ``run_evaluation``.
    normalise
        L2-normalise sentence vectors before computing dot products.

    Returns
    -------
    dict with paths to all output artefacts.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    codes = list(sentence_vectors.keys())
    nv = len(codes)

    # Validate alignment
    lengths = {k: v.shape[0] for k, v in sentence_vectors.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) > 1:
        raise ValueError(
            f"All varieties must have the same number of aligned sentences.\n"
            f"Got: { {k: v for k, v in lengths.items()} }"
        )
    N = next(iter(unique_lengths))

    # Normalise
    mats: Dict[str, np.ndarray] = {}
    for code, V in sentence_vectors.items():
        V = np.asarray(V, dtype=np.float32)
        if normalise:
            V = _l2_normalise(V)
        mats[code] = V

    # ---- pairwise mean sentence-pair cosine similarity --------------------
    # For L2-normalised vectors: cosine_sim(x, y) = x · y
    sim_matrix = np.ones((nv, nv), dtype=np.float64)
    for i, c1 in enumerate(codes):
        for j in range(i + 1, nv):
            c2 = codes[j]
            # element-wise dot product over N sentences, then mean
            sim_vec = np.einsum("nd,nd->n", mats[c1], mats[c2])  # (N,)
            val = float(np.mean(sim_vec))
            sim_matrix[i, j] = sim_matrix[j, i] = val

    sim_df = pd.DataFrame(sim_matrix, index=codes, columns=codes)
    csv_path = out_dir / "parallel_alignment.csv"
    sim_df.to_csv(csv_path, float_format="%.6f")

    # ---- heatmap ----------------------------------------------------------
    heatmap_path = _heatmap(
        sim_matrix, codes, out_dir / "parallel_alignment_heatmap.png",
        title=f"Parallel sentence alignment{f' — {method_label}' if method_label else ''}",
        colorbar_label="Mean sentence-pair cosine similarity",
        family_groups=family_groups,
        family_colors=family_colors,
        family_display_names=family_display_names,
        display_names=display_names,
    )

    # ---- pair-level ranking -----------------------------------------------
    pair_rows = []
    for i in range(nv):
        for j in range(i + 1, nv):
            pair_rows.append({
                "variety_A": codes[i],
                "variety_B": codes[j],
                "mean_similarity": float(sim_matrix[i, j]),
                "family_A": (family_groups.get(codes[i], "") if family_groups else ""),
                "family_B": (family_groups.get(codes[j], "") if family_groups else ""),
            })
    pair_df = (
        pd.DataFrame(pair_rows)
        .sort_values("mean_similarity", ascending=False)
        .reset_index(drop=True)
    )
    pairs_csv = out_dir / "parallel_alignment_pairs.csv"
    pair_df.to_csv(pairs_csv, index=False, float_format="%.6f")

    # ---- per-variety stats ------------------------------------------------
    per_var_rows = []
    for i, code in enumerate(codes):
        others = [sim_matrix[i, j] for j in range(nv) if j != i]
        per_var_rows.append({
            "variety": code,
            "family": (family_groups.get(code, "") if family_groups else ""),
            "mean_sim_to_others": float(np.mean(others)),
            "max_sim": float(np.max(others)),
            "min_sim": float(np.min(others)),
            "std_sim": float(np.std(others)),
        })
    per_var_df = (
        pd.DataFrame(per_var_rows)
        .sort_values("mean_sim_to_others", ascending=False)
        .reset_index(drop=True)
    )

    # ---- dialect-specific sub-analysis ------------------------------------
    # Identify "dialect" varieties via DIALECT_FAMILIES from the shared registry,
    # falling back to the legacy "italo_romance" label for backward compatibility.
    try:
        from analysis._shared.varieties import DIALECT_FAMILIES as _DIAL_FAMS
    except Exception:
        _DIAL_FAMS = {"italo_romance"}
    _DIAL_FAMS = set(_DIAL_FAMS) | {"italo_romance"}  # always include legacy name

    dialect_rows = []
    if family_groups:
        dialect_codes = [c for c in codes if family_groups.get(c) in _DIAL_FAMS]
        ref_codes     = [c for c in codes if family_groups.get(c) not in _DIAL_FAMS]
        for d in dialect_codes:
            di = codes.index(d)
            for r in ref_codes:
                ri = codes.index(r)
                dialect_rows.append({
                    "dialect": d,
                    "reference": r,
                    "ref_family": family_groups.get(r, ""),
                    "mean_similarity": float(sim_matrix[di, ri]),
                })
    if dialect_rows:
        dialect_df = (
            pd.DataFrame(dialect_rows)
            .sort_values(["dialect", "mean_similarity"], ascending=[True, False])
            .reset_index(drop=True)
        )
    else:
        dialect_df = pd.DataFrame(
            columns=["dialect", "reference", "ref_family", "mean_similarity"]
        )
    dialect_csv = out_dir / "parallel_alignment_dialects.csv"
    dialect_df.to_csv(dialect_csv, index=False, float_format="%.6f")

    # ---- report -----------------------------------------------------------
    report_path = out_dir / "parallel_alignment_report.txt"
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write(f"Parallel alignment report ({method_label})\n")
        fh.write("=" * 60 + "\n")
        fh.write(f"  n_varieties  = {nv}\n")
        fh.write(f"  n_sentences  = {N}\n\n")

        fh.write("Top 20 most-aligned variety pairs:\n")
        fh.write(pair_df.head(20).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        fh.write("\n\nBottom 10 least-aligned pairs:\n")
        fh.write(pair_df.tail(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        fh.write("\n\nPer-variety mean alignment to all others (desc.):\n")
        fh.write(per_var_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        if not dialect_df.empty:
            fh.write("\n\nDialect-to-reference alignment:\n")
            fh.write(dialect_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        fh.write("\n")

    return {
        "alignment_csv": str(csv_path),
        "alignment_heatmap": str(heatmap_path),
        "alignment_pairs_csv": str(pairs_csv),
        "alignment_dialects_csv": str(dialect_csv),
        "alignment_report": str(report_path),
        "n_sentences": N,
        "n_varieties": nv,
    }
