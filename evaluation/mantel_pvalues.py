"""
Mantel permutation test on every (model × gold) pair already produced by
the evaluation pipeline.

Reads:
  * ``analysis/<method>/experiments/<exp>/.../distances.csv`` (17×17 model
    distance matrix produced by the centroid evaluation).
  * Every ``*.npz`` gold matrix under ``--gold-dir``.

For each pair, permutes the variety labels of the model matrix
``--n-perm`` times (default 10000), recomputes both Spearman scores
(full triangle + dialect↔external block), and reports a two-sided p-value:

    p = (#{|ρ_perm| ≥ |ρ_obs|} + 1) / (n_perm + 1)

Two-sided because some models have negative ρ (e.g. LaBSE), and a
one-sided "≥ ρ_obs" test would mis-classify those as non-significant
when they are in fact significantly *anti*-correlated with the gold.

Writes one CSV per gold to ``--out-dir``:

    correlation_<gold_name>_with_pvalue.csv
        method, experiment, variant, model_id,
        Spearman ρ (full matrix),
        Mantel p (full matrix),
        Spearman ρ (dialect ↔ external),
        Mantel p (dialect ↔ external)

CLI:
    python -m evaluation.mantel_pvalues \\
        --gold-dir gold/lexicostatistical/matrices gold/geographic/matrices \\
        --analysis-root analysis \\
        --out-dir gold/_correlations \\
        --n-perm 10000 --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from evaluation._gold_correlation import (
    COL_RHO_DIAEXT,
    COL_RHO_FULL,
    _restrict_to,
    _spearman_safe,
    default_roles,
    load_gold,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

COL_P_FULL   = "Mantel p (full matrix)"
COL_P_DIAEXT = "Mantel p (dialect ↔ external)"


def _iter_distance_csvs(analysis_root: Path) -> List[Path]:
    out: List[Path] = []
    for method_dir in sorted(p for p in analysis_root.iterdir() if p.is_dir()):
        if method_dir.name.startswith("_"):
            continue
        er_root = method_dir / "experiments"
        if not er_root.exists():
            continue
        out.extend(sorted(er_root.rglob("distances.csv")))
    return out


def _parse_model_path(distances_csv: Path, analysis_root: Path) -> Dict[str, str]:
    rel = distances_csv.relative_to(analysis_root).parts
    method = rel[0]
    experiment = rel[2] if len(rel) > 2 else "?"
    try:
        er_idx = rel.index("evaluation_results")
        variant = "/".join(rel[er_idx + 1 : -1])
    except ValueError:
        variant = ""
    short_id = f"{method}__{experiment}"
    if variant:
        short_id += "__" + variant.replace("/", "_")
    return {
        "method": method, "experiment": experiment,
        "variant": variant, "model_id": short_id,
    }


def _load_distance_matrix(path: Path) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(path, index_col=0)
    return df.values.astype(np.float64), [str(x) for x in df.index]


def mantel_pvalues(
    model_mat: np.ndarray, model_labels: List[str],
    gold_mat:  np.ndarray, gold_labels:  List[str],
    dialect_codes: List[str], external_codes: List[str],
    n_perm: int, rng: np.random.Generator,
) -> Tuple[float, float, float, float, int]:
    """Return (rho_full, p_full, rho_dia, p_dia, n_shared)."""
    shared = [c for c in gold_labels if c in model_labels]
    n = len(shared)
    if n < 4:
        return (float("nan"),) * 4 + (n,)

    md, _ = _restrict_to(model_mat, model_labels, shared)
    gd, _ = _restrict_to(gold_mat,  gold_labels,  shared)

    iu = np.triu_indices(n, k=1)
    gold_full = gd[iu]
    rho_full_obs = _spearman_safe(md[iu], gold_full)

    d_set = [c for c in dialect_codes if c in shared]
    e_set = [c for c in external_codes if c in shared]
    use_block = bool(d_set and e_set)
    if use_block:
        d_idx = np.asarray([shared.index(c) for c in d_set])
        e_idx = np.asarray([shared.index(c) for c in e_set])
        gold_block = gd[np.ix_(d_idx, e_idx)].flatten()
        rho_dia_obs = _spearman_safe(
            md[np.ix_(d_idx, e_idx)].flatten(), gold_block,
        )
    else:
        rho_dia_obs = float("nan")

    # Permutations: shuffle the model matrix labels, keep gold and roles fixed.
    abs_obs_full = abs(rho_full_obs) if np.isfinite(rho_full_obs) else 0.0
    abs_obs_dia  = abs(rho_dia_obs)  if np.isfinite(rho_dia_obs)  else 0.0
    cnt_full = 0
    cnt_dia  = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        mdp = md[np.ix_(perm, perm)]
        r_full = _spearman_safe(mdp[iu], gold_full)
        if np.isfinite(r_full) and abs(r_full) >= abs_obs_full:
            cnt_full += 1
        if use_block and np.isfinite(rho_dia_obs):
            r_dia = _spearman_safe(
                mdp[np.ix_(d_idx, e_idx)].flatten(), gold_block,
            )
            if np.isfinite(r_dia) and abs(r_dia) >= abs_obs_dia:
                cnt_dia += 1

    p_full = (cnt_full + 1) / (n_perm + 1) if np.isfinite(rho_full_obs) else float("nan")
    p_dia  = (cnt_dia  + 1) / (n_perm + 1) if (use_block and np.isfinite(rho_dia_obs)) else float("nan")
    return rho_full_obs, p_full, rho_dia_obs, p_dia, n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold-dir", type=Path, nargs="+", required=True,
                    help="One or more folders containing <name>.npz gold matrices.")
    ap.add_argument("--analysis-root", type=Path, default=REPO_ROOT / "analysis")
    ap.add_argument("--out-dir",       type=Path, required=True)
    ap.add_argument("--n-perm",        type=int,  default=10_000)
    ap.add_argument("--seed",          type=int,  default=42)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dialect_codes, external_codes = default_roles()
    rng = np.random.default_rng(args.seed)

    gold_paths: List[Path] = []
    for d in args.gold_dir:
        gold_paths.extend(sorted(d.glob("*.npz")))
    if not gold_paths:
        print(f"No .npz gold found in {args.gold_dir}", file=sys.stderr)
        return 1

    dist_csvs = _iter_distance_csvs(args.analysis_root)
    print(f"Golds  : {[p.stem for p in gold_paths]}")
    print(f"Models : {len(dist_csvs)} distances.csv under {args.analysis_root}")
    print(f"n_perm : {args.n_perm}")

    for gp in gold_paths:
        gold_mat, gold_labels, _ = load_gold(gp)

        rows: List[Dict] = []
        for dc in dist_csvs:
            try:
                model_mat, model_labels = _load_distance_matrix(dc)
            except Exception as exc:
                print(f"  skip {dc} — {exc}", file=sys.stderr)
                continue
            info = _parse_model_path(dc, args.analysis_root)
            rho_f, p_f, rho_d, p_d, _ = mantel_pvalues(
                model_mat, model_labels,
                gold_mat,  gold_labels,
                dialect_codes, external_codes,
                n_perm=args.n_perm, rng=rng,
            )
            rows.append({
                "method":     info["method"],
                "experiment": info["experiment"],
                "variant":    info["variant"],
                "model_id":   info["model_id"],
                COL_RHO_FULL:   rho_f,
                COL_P_FULL:     p_f,
                COL_RHO_DIAEXT: rho_d,
                COL_P_DIAEXT:   p_d,
            })

        if not rows:
            continue
        df = pd.DataFrame(rows).sort_values(COL_RHO_FULL, ascending=False)
        out_csv = args.out_dir / f"correlation_{gp.stem}_with_pvalue.csv"
        df.to_csv(out_csv, index=False, float_format="%.4f")
        print(f"\n→ {out_csv}  ({len(df)} rows)")
        print(df.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
