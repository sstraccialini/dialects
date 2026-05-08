"""
For every (model, gold) pair: Spearman ρ + Mantel test on distance matrices.

Both metrics are computed on the upper triangle of the pairwise distance
matrices, restricted to the varieties shared between the model and the
gold (gold may have NaN entries — see ASJP).  We rely on:

    evaluation.compare_methods.rank_correlation  (Spearman ρ)
    evaluation.compare_methods.mantel_test       (Pearson r + permutation p)

Output: ``edoardo/results/correlation_with_gold.csv`` with columns
    method, root_kind, experiment, variant_path, model_id,
    gold_family, gold_name, n_shared,
    spearman_rho, mantel_r, mantel_p,
    significant_005

Plus a wide pivot ``correlation_with_gold_pivot.csv`` keyed on model_id
with one column per gold (Spearman ρ).

Run from the repo root:
    python -m edoardo.analysis.correlate_with_gold
    python -m edoardo.analysis.correlate_with_gold --include-old
    python -m edoardo.analysis.correlate_with_gold --permutations 9999
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from edoardo.analysis.load_gold import discover_golds, load_gold
from edoardo.analysis.load_models import discover_models, restrict_to_codes
from evaluation.compare_methods import mantel_test, rank_correlation


RESULTS_DIR_DEFAULT = Path(__file__).resolve().parents[1] / "results"


def _drop_nan_pairs(model_d: np.ndarray, gold_d: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Return matched flat upper-triangle vectors after NaN filtering."""
    n = model_d.shape[0]
    iu = np.triu_indices(n, k=1)
    a = model_d[iu]
    b = gold_d[iu]
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def _spearman_safe(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or b.size < 3:
        return float("nan")
    from scipy.stats import spearmanr
    r, _ = spearmanr(a, b)
    return float(r)


def _mantel_safe(model_d: np.ndarray, gold_d: np.ndarray,
                 permutations: int, random_state: int) -> tuple[float, float]:
    # Mantel needs square matrices; if any NaN remains after restrict_to_codes,
    # we drop the rows/cols of any all-NaN entry.
    n = model_d.shape[0]
    finite_rows = np.array([
        np.isfinite(gold_d[i]).any() and np.isfinite(model_d[i]).any()
        for i in range(n)
    ])
    if finite_rows.sum() < 3:
        return float("nan"), float("nan")
    keep = np.where(finite_rows)[0]
    a = model_d[np.ix_(keep, keep)].copy()
    b = gold_d[np.ix_(keep, keep)].copy()
    # Replace remaining NaN with column means (rare; only triggers for
    # sparse golds like ASJP).
    for M in (a, b):
        col_mean = np.nanmean(M, axis=0)
        idx = np.where(np.isnan(M))
        M[idx] = col_mean[idx[1]]
        if np.isnan(M).any():
            return float("nan"), float("nan")
    try:
        r, p = mantel_test(a, b, permutations=permutations, random_state=random_state)
    except Exception as exc:
        warnings.warn(f"Mantel failed: {exc}", stacklevel=2)
        return float("nan"), float("nan")
    return float(r), float(p)


def correlate_one(model, gold, target_codes: List[str],
                  permutations: int, random_state: int) -> Dict:
    model_mat, model_labels = model.load_distances()
    gold_mat, gold_labels, _ = load_gold(gold)

    shared = [c for c in target_codes if c in model_labels and c in gold_labels]
    if len(shared) < 4:
        return {
            "n_shared": len(shared),
            "spearman_rho": float("nan"),
            "mantel_r": float("nan"),
            "mantel_p": float("nan"),
            "significant_005": False,
        }

    md, _ = restrict_to_codes(model_mat, model_labels, shared)
    gd, _ = restrict_to_codes(gold_mat, gold_labels, shared)

    a, b = _drop_nan_pairs(md, gd)
    rho = _spearman_safe(a, b)
    mr, mp = _mantel_safe(md, gd, permutations=permutations, random_state=random_state)

    return {
        "n_shared": len(shared),
        "spearman_rho": rho,
        "mantel_r": mr,
        "mantel_p": mp,
        "significant_005": bool(np.isfinite(mp) and mp < 0.05),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR_DEFAULT)
    ap.add_argument("--gold-dir", type=Path, default=None,
                    help="Override the directory with gold .npz matrices.")
    ap.add_argument("--include-old", action="store_true",
                    help="Also compare models from old_experiments/.")
    ap.add_argument("--permutations", type=int, default=999)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    models = discover_models(include_old=args.include_old)
    golds = discover_golds(args.gold_dir)
    if not models:
        print("No models found (no distances.csv under analysis/).", file=sys.stderr)
        return 1
    if not golds:
        print("No gold matrices found.  Run "
              "`python -m edoardo.gold_references.build_all` first.",
              file=sys.stderr)
        return 1

    print(f"Models : {len(models)}")
    print(f"Golds  : {[g.name for g in golds]}")
    print(f"Pairs  : {len(models) * len(golds)}\n")

    # Use the union of codes seen across all models (typically the canonical 13)
    seen_codes = set()
    for m in models:
        _, lbls = m.load_distances()
        seen_codes.update(lbls)
    target = sorted(seen_codes)

    rows = []
    for m in models:
        for g in golds:
            res = correlate_one(m, g, target,
                                permutations=args.permutations,
                                random_state=args.seed)
            rows.append({
                "method": m.method,
                "root_kind": m.root_kind,
                "experiment": m.experiment,
                "variant_path": m.variant_path,
                "model_id": m.short_id,
                "gold_family": g.family,
                "gold_name": g.name,
                **res,
            })

    df = pd.DataFrame(rows)
    long_path = args.out_dir / "correlation_with_gold.csv"
    df.to_csv(long_path, index=False, float_format="%.6f")
    print(f"  long format → {long_path}")

    # Wide pivot: rows=model_id, cols=gold_name, values=spearman_rho
    wide = df.pivot_table(
        index=["method", "experiment", "variant_path", "model_id"],
        columns="gold_name",
        values="spearman_rho",
        aggfunc="first",
    ).reset_index()
    wide_path = args.out_dir / "correlation_with_gold_pivot.csv"
    wide.to_csv(wide_path, index=False, float_format="%.4f")
    print(f"  pivot       → {wide_path}")

    # Console preview: best gold-alignment per model
    print("\nBest-aligned gold per model (highest Spearman ρ):")
    best = (df.dropna(subset=["spearman_rho"])
              .sort_values("spearman_rho", ascending=False)
              .groupby("model_id")
              .head(1)
              [["model_id", "gold_name", "spearman_rho", "mantel_p", "n_shared"]])
    print(best.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
