"""
Cross-method comparison: compare distance matrices and embedding spaces
from different embedding methods (TF-IDF, Word2Vec, FastText, XLM-R, …).

Metrics
-------
Spearman rank correlation
    How similarly do two methods rank variety pairs by distance?
    Range [-1, 1]; higher = more consistent distance orderings.

Mantel test
    Permutation test of the Pearson correlation between two distance
    matrices.  Returns (r, p-value).  p < 0.05 means the correlation
    is unlikely by chance.

Procrustes disparity
    Finds the best orthogonal rotation aligning space A onto space B,
    then reports the normalised residual.  0 = perfect alignment,
    1 = completely unrelated geometries.  Requires variety-vector files.

Linear CKA (Centered Kernel Alignment)
    Rotation- and scale-invariant similarity between representation matrices.
    Range [0, 1]; 1 = identical geometry up to orthogonal transform + scaling.
    Requires variety-vector files.

Usage
-----
    from evaluation.compare_methods import run_cross_method_comparison

    run_cross_method_comparison(
        {
            "TF-IDF (char)": "analysis/tfidf/flores/evaluation_results/char",
            "Word2Vec":       "analysis/word2vec/flores/evaluation_results",
            "FastText":       "analysis/fasttext/flores/evaluation_results/fasttext",
            "XLM-R":         "analysis/multilingual_xlmr/flores/evaluation_results",
            "Sentence-MiniLM":"analysis/sentence_baseline/flores/evaluation_results/sentence",
        },
        out_dir="analysis/comparison/flores",
        variety_vector_paths={
            "Word2Vec": "analysis/word2vec/flores/method_outputs/variety_vectors.npz",
            "XLM-R":    "analysis/multilingual_xlmr/flores/method_outputs/variety_vectors.npz",
        },
    )
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

def load_distance_matrix(path: Union[str, Path]) -> Tuple[np.ndarray, List[str]]:
    """Load a ``distances.csv`` produced by ``run_evaluation``."""
    df = pd.read_csv(path, index_col=0)
    codes = list(df.index)
    return df.values.astype(np.float64), codes


def load_variety_vectors(path: Union[str, Path]) -> Tuple[np.ndarray, List[str]]:
    """
    Load variety vectors from ``.npz`` (keys: ``matrix``, ``labels``)
    or ``.csv`` (row index = variety code, columns = dimensions).
    """
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        matrix = data["matrix"].astype(np.float64)
        labels = [str(x) for x in data["labels"]]
    elif path.suffix in (".csv", ".tsv"):
        sep = "\t" if path.suffix == ".tsv" else ","
        df = pd.read_csv(path, index_col=0, sep=sep)
        labels = list(df.index)
        matrix = df.values.astype(np.float64)
    else:
        raise ValueError(f"Unsupported vector format: {path.suffix}. Use .npz or .csv")
    return matrix, labels


# --------------------------------------------------------------------------- #
# Pairwise metrics
# --------------------------------------------------------------------------- #

def rank_correlation(dist1: np.ndarray, dist2: np.ndarray) -> float:
    """Spearman rank correlation of the upper triangles of two distance matrices."""
    n = dist1.shape[0]
    idx = np.triu_indices(n, k=1)
    r, _ = spearmanr(dist1[idx], dist2[idx])
    return float(r)


def mantel_test(
    dist1: np.ndarray,
    dist2: np.ndarray,
    permutations: int = 999,
    random_state: int = 42,
) -> Tuple[float, float]:
    """
    Mantel test: permutation test of Pearson correlation between two distance
    matrices.

    Returns
    -------
    (r, p_value)
        r is the observed Pearson correlation of the upper triangles.
        p_value is the one-tailed p (fraction of permuted r >= observed r).
    """
    rng = np.random.default_rng(random_state)
    n = dist1.shape[0]
    if dist2.shape[0] != n:
        raise ValueError("Distance matrices must have the same shape")
    idx = np.triu_indices(n, k=1)
    v1 = dist1[idx]
    v2 = dist2[idx]
    r_obs = float(np.corrcoef(v1, v2)[0, 1])

    count = 0
    for _ in range(permutations):
        perm = rng.permutation(n)
        v2p = dist2[np.ix_(perm, perm)][idx]
        if np.corrcoef(v1, v2p)[0, 1] >= r_obs:
            count += 1

    p = (count + 1) / (permutations + 1)
    return r_obs, p


def procrustes_disparity(
    X1: np.ndarray,
    codes1: List[str],
    X2: np.ndarray,
    codes2: List[str],
) -> float:
    """
    Orthogonal Procrustes: find rotation R minimising ||X1 R - X2||_F
    on the shared subset of varieties.

    Returns the normalised Frobenius disparity after alignment.
    Lower = more similar embedding geometries.
    """
    shared = sorted(set(codes1) & set(codes2))
    if len(shared) < 2:
        raise ValueError(f"Only {len(shared)} shared variety codes; need at least 2")
    i1 = [codes1.index(c) for c in shared]
    i2 = [codes2.index(c) for c in shared]
    A = X1[i1].copy().astype(np.float64)
    B = X2[i2].copy().astype(np.float64)
    # L2-normalise rows
    A /= np.linalg.norm(A, axis=1, keepdims=True).clip(1e-12)
    B /= np.linalg.norm(B, axis=1, keepdims=True).clip(1e-12)
    R, _ = orthogonal_procrustes(A, B)
    A_aligned = A @ R
    denom = np.linalg.norm(B, "fro")
    return float(np.linalg.norm(A_aligned - B, "fro") / denom) if denom > 0 else float("nan")


def cka_score(
    X1: np.ndarray,
    codes1: List[str],
    X2: np.ndarray,
    codes2: List[str],
) -> float:
    """
    Linear CKA between two representation matrices on shared varieties.

    CKA ∈ [0, 1]; 1 = identical geometry up to orthogonal transform + scaling.
    """
    shared = sorted(set(codes1) & set(codes2))
    if len(shared) < 2:
        return float("nan")
    i1 = [codes1.index(c) for c in shared]
    i2 = [codes2.index(c) for c in shared]
    A = X1[i1].astype(np.float64)
    B = X2[i2].astype(np.float64)

    def _hsic(Ka, Kb):
        n = Ka.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        Kac = H @ Ka @ H
        Kbc = H @ Kb @ H
        return np.trace(Kac @ Kbc)

    Ka = A @ A.T
    Kb = B @ B.T
    num = _hsic(Ka, Kb)
    denom = np.sqrt(_hsic(Ka, Ka) * _hsic(Kb, Kb))
    return float(num / denom) if denom > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Internal heatmap helper
# --------------------------------------------------------------------------- #

def _square_heatmap(
    matrix: np.ndarray,
    labels: List[str],
    out_path: Path,
    *,
    title: str,
    cmap: str = "RdYlGn",
    vmin: float = -1.0,
    vmax: float = 1.0,
    colorbar_label: str = "",
) -> None:
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(7, n * 0.8), max(6, n * 0.7)))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    fs = 7 if n > 10 else 9
    for i in range(n):
        for j in range(n):
            v = matrix[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=fs)
    plt.colorbar(im, ax=ax, label=colorbar_label, fraction=0.046, pad=0.04)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def run_cross_method_comparison(
    method_eval_dirs: Dict[str, Union[str, Path]],
    out_dir: Union[str, Path],
    *,
    variety_vector_paths: Optional[Dict[str, Union[str, Path]]] = None,
    mantel_permutations: int = 999,
    random_state: int = 42,
) -> Dict[str, object]:
    """
    Full cross-method comparison using distance matrices and (optionally)
    raw variety vectors.

    Parameters
    ----------
    method_eval_dirs
        Dict mapping method name -> path to its ``evaluation_results/``
        directory.  Each directory must contain ``distances.csv``.
    out_dir
        Where comparison outputs are written.
    variety_vector_paths
        Optional dict mapping method name -> path to ``variety_vectors.npz``
        or ``variety_vectors.csv``.  Required for Procrustes and CKA.
    mantel_permutations
        Permutations for the Mantel test (default 999; 0 skips the test).

    Outputs
    -------
    method_rank_correlation.csv          Spearman ρ matrix between all methods
    method_rank_correlation_heatmap.png
    method_mantel.csv                    Mantel r + p-value for every pair
    method_procrustes.csv                Procrustes disparity (if vectors given)
    method_procrustes_heatmap.png
    method_cka.csv                       Linear CKA (if vectors given)
    method_cka_heatmap.png
    comparison_report.txt                Human-readable summary
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load distance matrices ------------------------------------------
    dists: Dict[str, np.ndarray] = {}
    codes_map: Dict[str, List[str]] = {}
    for name, d in method_eval_dirs.items():
        dist_csv = Path(d) / "distances.csv"
        if not dist_csv.exists():
            warnings.warn(f"{name}: distances.csv not found at {dist_csv} — skipping", stacklevel=2)
            continue
        mat, cds = load_distance_matrix(dist_csv)
        dists[name] = mat
        codes_map[name] = cds

    methods = list(dists.keys())
    m = len(methods)
    if m < 2:
        raise RuntimeError(f"Need at least 2 methods with valid distances.csv; found {m}")

    # ---- Spearman rank correlation ----------------------------------------
    rho_mat = np.ones((m, m))
    for i, m1 in enumerate(methods):
        for j in range(i + 1, m):
            m2 = methods[j]
            shared = sorted(set(codes_map[m1]) & set(codes_map[m2]))
            if len(shared) < 3:
                rho_mat[i, j] = rho_mat[j, i] = float("nan")
                continue
            i1 = [codes_map[m1].index(c) for c in shared]
            i2 = [codes_map[m2].index(c) for c in shared]
            r = rank_correlation(dists[m1][np.ix_(i1, i1)], dists[m2][np.ix_(i2, i2)])
            rho_mat[i, j] = rho_mat[j, i] = r

    rho_df = pd.DataFrame(rho_mat, index=methods, columns=methods)
    rho_path = out_dir / "method_rank_correlation.csv"
    rho_df.to_csv(rho_path, float_format="%.6f")

    _square_heatmap(
        rho_mat, methods, out_dir / "method_rank_correlation_heatmap.png",
        title="Method distance-rank correlation (Spearman ρ)",
        cmap="RdYlGn", vmin=-1.0, vmax=1.0,
        colorbar_label="Spearman ρ",
    )

    # ---- Mantel test ------------------------------------------------------
    mantel_rows = []
    if mantel_permutations > 0:
        for i, m1 in enumerate(methods):
            for j in range(i + 1, m):
                m2 = methods[j]
                shared = sorted(set(codes_map[m1]) & set(codes_map[m2]))
                if len(shared) < 3:
                    continue
                i1 = [codes_map[m1].index(c) for c in shared]
                i2 = [codes_map[m2].index(c) for c in shared]
                r, p = mantel_test(
                    dists[m1][np.ix_(i1, i1)],
                    dists[m2][np.ix_(i2, i2)],
                    permutations=mantel_permutations,
                    random_state=random_state,
                )
                mantel_rows.append({
                    "method_A": m1, "method_B": m2,
                    "mantel_r": r, "p_value": p,
                    "n_varieties": len(shared),
                    "significant_005": p < 0.05,
                })
    mantel_df = pd.DataFrame(mantel_rows)
    if not mantel_df.empty:
        mantel_df = mantel_df.sort_values("mantel_r", ascending=False)
    mantel_path = out_dir / "method_mantel.csv"
    mantel_df.to_csv(mantel_path, index=False, float_format="%.6f")

    # ---- Procrustes + CKA (require variety vectors) ----------------------
    proc_path: Optional[Path] = None
    cka_path: Optional[Path] = None
    proc_mat = np.full((m, m), float("nan"))
    cka_arr = np.full((m, m), float("nan"))
    np.fill_diagonal(proc_mat, 0.0)
    np.fill_diagonal(cka_arr, 1.0)

    if variety_vector_paths:
        vecs: Dict[str, Tuple[np.ndarray, List[str]]] = {}
        for name, p in variety_vector_paths.items():
            if name not in dists:
                continue
            try:
                vecs[name] = load_variety_vectors(p)
            except Exception as exc:
                warnings.warn(f"{name}: could not load vectors from {p} — {exc}", stacklevel=2)

        vec_names = [mn for mn in methods if mn in vecs]
        for i, m1 in enumerate(methods):
            for j in range(i + 1, m):
                m2 = methods[j]
                if m1 not in vecs or m2 not in vecs:
                    continue
                X1, c1 = vecs[m1]
                X2, c2 = vecs[m2]
                try:
                    disp = procrustes_disparity(X1, c1, X2, c2)
                    proc_mat[i, j] = proc_mat[j, i] = disp
                except Exception as exc:
                    warnings.warn(f"Procrustes {m1}/{m2}: {exc}", stacklevel=2)
                try:
                    cka = cka_score(X1, c1, X2, c2)
                    cka_arr[i, j] = cka_arr[j, i] = cka
                except Exception as exc:
                    warnings.warn(f"CKA {m1}/{m2}: {exc}", stacklevel=2)

        if len(vec_names) >= 2:
            proc_df = pd.DataFrame(proc_mat, index=methods, columns=methods)
            proc_path = out_dir / "method_procrustes.csv"
            proc_df.to_csv(proc_path, float_format="%.6f")
            _square_heatmap(
                proc_mat, methods, out_dir / "method_procrustes_heatmap.png",
                title="Procrustes disparity (lower = more similar spaces)",
                cmap="RdYlGn_r", vmin=0.0, vmax=1.0,
                colorbar_label="Normalised Frobenius disparity",
            )

            cka_df = pd.DataFrame(cka_arr, index=methods, columns=methods)
            cka_path = out_dir / "method_cka.csv"
            cka_df.to_csv(cka_path, float_format="%.6f")
            _square_heatmap(
                cka_arr, methods, out_dir / "method_cka_heatmap.png",
                title="Linear CKA (higher = more similar spaces)",
                cmap="RdYlGn", vmin=0.0, vmax=1.0,
                colorbar_label="Linear CKA",
            )

    # ---- Text report ------------------------------------------------------
    report_path = out_dir / "comparison_report.txt"
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("Cross-method comparison report\n")
        fh.write("=" * 60 + "\n\n")
        fh.write(f"Methods compared ({m}):\n")
        for mn in methods:
            fh.write(f"  {mn}\n")
        fh.write("\n")

        fh.write("Spearman rank correlation (distance orderings):\n")
        fh.write(rho_df.to_string(float_format=lambda x: f"{x:+.4f}"))
        fh.write("\n\n")

        if not mantel_df.empty:
            fh.write("Mantel test results (r, p-value):\n")
            fh.write(mantel_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            fh.write("\n\n")

        if proc_path:
            fh.write("Procrustes disparity (0=identical, 1=unrelated):\n")
            fh.write(proc_df.to_string(float_format=lambda x: f"{x:.4f}"))
            fh.write("\n\n")

        if cka_path:
            fh.write("Linear CKA (0=unrelated, 1=identical):\n")
            fh.write(cka_df.to_string(float_format=lambda x: f"{x:.4f}"))
            fh.write("\n")

    return {
        "rank_correlation_csv": str(rho_path),
        "rank_correlation_heatmap": str(out_dir / "method_rank_correlation_heatmap.png"),
        "mantel_csv": str(mantel_path),
        "procrustes_csv": str(proc_path) if proc_path else None,
        "procrustes_heatmap": str(out_dir / "method_procrustes_heatmap.png") if proc_path else None,
        "cka_csv": str(cka_path) if cka_path else None,
        "cka_heatmap": str(out_dir / "method_cka_heatmap.png") if cka_path else None,
        "report": str(report_path),
        "methods": methods,
        "n_methods": m,
    }
