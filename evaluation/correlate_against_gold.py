"""
Correlate every embedding-method output under ``analysis/`` against the
team-curated lexicostatistical LDND gold matrix.

Two correlation metrics are reported per (model, gold) pair:

    rho   — Spearman rank correlation on the FULL upper triangle of the
            shared distance matrix.  Captures ALL pairwise relationships
            between varieties.
    rho2  — Spearman correlation restricted to the cross-block of pairs
            where one variety is a "dialect" and the other is an "external"
            (non-Italian, non-dialect) language.  Excludes intra-dialect
            pairs and pairs involving standard Italian.  Tells you how
            well the model recovers the *dialect ↔ external-language*
            relationships specifically — the hardest signal because it
            crosses the genealogical boundary.

Roles are read from ``gold/lexicostatistical/varieties.py``.  When the
variety set grows (more Wikipedia languages added), update that file and
the script adapts automatically.

Models are discovered by globbing ``analysis/*/experiments/*/evaluation_results/**/distances.csv``.
We work on the SHARED label set between each (model, gold), so model
runs that include extra varieties (e.g. 16-variety legacy runs) are
restricted to the gold's labels before correlation.

Output: a single CSV ``<out>/correlation_with_gold_pivot.csv`` with one
row per (model, gold) pair and columns:
    model_id, method, experiment, variant, gold_name,
    n_shared_full, rho, n_shared_dial_ext, rho2

CLI:
    python -m evaluation.correlate_against_gold \\
        --gold-dir gold/lexicostatistical/matrices \\
        --analysis-root analysis \\
        --out gold/lexicostatistical/results/correlation_with_gold_pivot.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

def _iter_distance_csvs(analysis_root: Path) -> List[Path]:
    """Find every model's distances.csv under analysis/<method>/experiments/."""
    out: List[Path] = []
    for method_dir in sorted(p for p in analysis_root.iterdir() if p.is_dir()):
        if method_dir.name.startswith("_"):
            continue
        for er_kind in ("experiments", "old_experiments"):
            er_root = method_dir / er_kind
            if not er_root.exists():
                continue
            out.extend(sorted(er_root.rglob("distances.csv")))
    return out


def _parse_model_path(distances_csv: Path, analysis_root: Path) -> Dict[str, str]:
    """Extract method / experiment / variant identifiers from a distances.csv path.

    Path layout:
        analysis/<method>/(experiments|old_experiments)/<experiment>/.../<...>/evaluation_results/[<variant>/]distances.csv
    """
    rel = distances_csv.relative_to(analysis_root).parts
    method = rel[0]
    root_kind = rel[1] if len(rel) > 1 and rel[1] in ("experiments", "old_experiments") else "?"
    experiment = rel[2] if root_kind != "?" and len(rel) > 2 else "?"
    try:
        er_idx = rel.index("evaluation_results")
        variant = "/".join(rel[er_idx + 1 : -1])
    except ValueError:
        variant = ""
    short_id = f"{method}__{experiment}"
    if variant:
        short_id += "__" + variant.replace("/", "_")
    return {
        "method": method,
        "root_kind": root_kind,
        "experiment": experiment,
        "variant": variant,
        "model_id": short_id,
    }


def _load_distance_matrix(path: Path) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(path, index_col=0)
    return df.values.astype(np.float64), [str(x) for x in df.index]


def _load_gold(npz_path: Path) -> Tuple[np.ndarray, List[str], Dict]:
    data = np.load(npz_path, allow_pickle=True)
    mat = np.asarray(data["matrix"], dtype=np.float64)
    labels = [str(x) for x in data["labels"]]
    try:
        meta = json.loads(str(data["meta"][0]))
    except Exception:
        meta = {}
    return mat, labels, meta


# --------------------------------------------------------------------------- #
# Correlation
# --------------------------------------------------------------------------- #

def _restrict(matrix: np.ndarray, labels: List[str], target: List[str]
              ) -> Tuple[np.ndarray, List[str]]:
    keep = [c for c in target if c in labels]
    idx = [labels.index(c) for c in keep]
    return matrix[np.ix_(idx, idx)], keep


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or b.size < 3:
        return float("nan")
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    r, _ = spearmanr(a[mask], b[mask])
    return float(r) if r is not None else float("nan")


def _full_triangle(matrix: np.ndarray) -> np.ndarray:
    """Return the upper triangle (i<j) as a flat array."""
    n = matrix.shape[0]
    iu = np.triu_indices(n, k=1)
    return matrix[iu]


def _cross_block(matrix: np.ndarray, labels: List[str],
                 row_codes: List[str], col_codes: List[str]) -> np.ndarray:
    """Flatten the cross-block matrix[row_codes, col_codes].

    row_codes ∩ col_codes is assumed empty (e.g. dialect × external).
    Returns a 1-D array of len(row_codes) * len(col_codes) values, or an
    empty array if either set is empty in the labels.
    """
    rows = [labels.index(c) for c in row_codes if c in labels]
    cols = [labels.index(c) for c in col_codes if c in labels]
    if not rows or not cols:
        return np.array([], dtype=np.float64)
    return matrix[np.ix_(rows, cols)].flatten()


def correlate_one(model_dist: np.ndarray, model_labels: List[str],
                  gold_dist: np.ndarray,  gold_labels:  List[str],
                  dialect_codes: List[str],
                  external_codes: List[str]) -> Dict[str, float]:
    """Compute rho (full) and rho2 (dialect × external) for one (model, gold)."""
    shared = [c for c in gold_labels if c in model_labels]
    if len(shared) < 4:
        return {"n_shared_full": len(shared), "rho": float("nan"),
                "n_shared_dial_ext": 0, "rho2": float("nan")}

    md, _ = _restrict(model_dist, model_labels, shared)
    gd, _ = _restrict(gold_dist,  gold_labels,  shared)

    # rho: full upper triangle
    a_full = _full_triangle(md)
    b_full = _full_triangle(gd)
    rho = _spearman(a_full, b_full)

    # rho2: dialect × external cross-block
    d_set = [c for c in dialect_codes if c in shared]
    e_set = [c for c in external_codes if c in shared]
    a_cross = _cross_block(md, shared, d_set, e_set)
    b_cross = _cross_block(gd, shared, d_set, e_set)
    rho2 = _spearman(a_cross, b_cross)

    return {
        "n_shared_full": len(shared),
        "rho": rho,
        "n_shared_dial_ext": int(min(len(d_set), len(e_set))) * (
            int(max(len(d_set), len(e_set))) if (d_set and e_set) else 0),
        "rho2": rho2,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold-dir", type=Path, required=True,
                    help="Folder with one or more <name>.npz gold matrices.")
    ap.add_argument("--analysis-root", type=Path, default=REPO_ROOT / "analysis",
                    help="Root of the analysis/ tree to scan for models.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Path of the pivot CSV to write.")
    ap.add_argument("--varieties-module", type=str,
                    default="gold.lexicostatistical.varieties",
                    help="Module exporting DIALECT_CODES + EXTERNAL_CODES.")
    args = ap.parse_args(argv)

    # Roles for rho2
    import importlib
    roles = importlib.import_module(args.varieties_module)
    dialect_codes: List[str] = list(roles.DIALECT_CODES)
    external_codes: List[str] = list(roles.EXTERNAL_CODES)

    # Discover golds
    gold_paths = sorted(args.gold_dir.glob("*.npz"))
    if not gold_paths:
        print(f"No .npz gold found in {args.gold_dir}", file=sys.stderr)
        return 1

    # Discover models
    dist_csvs = _iter_distance_csvs(args.analysis_root)
    print(f"Golds  : {[p.stem for p in gold_paths]}")
    print(f"Models : {len(dist_csvs)} distances.csv files found under {args.analysis_root}")

    rows: List[Dict] = []
    for gp in gold_paths:
        gold_mat, gold_labels, _ = _load_gold(gp)
        for dc in dist_csvs:
            try:
                model_mat, model_labels = _load_distance_matrix(dc)
            except Exception as exc:
                print(f"  skip {dc} — {exc}", file=sys.stderr)
                continue
            info = _parse_model_path(dc, args.analysis_root)
            res = correlate_one(model_mat, model_labels,
                                gold_mat, gold_labels,
                                dialect_codes, external_codes)
            rows.append({**info, "gold_name": gp.stem, **res,
                         "distances_csv": str(dc.relative_to(REPO_ROOT))})

    df = pd.DataFrame(rows)
    if df.empty:
        print("No (model, gold) pairs computed.", file=sys.stderr)
        return 1

    # Order columns
    cols = ["gold_name", "method", "experiment", "variant", "model_id",
            "n_shared_full", "rho",
            "n_shared_dial_ext", "rho2",
            "root_kind", "distances_csv"]
    df = df[cols].sort_values(["gold_name", "rho"], ascending=[True, False])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, float_format="%.4f")
    print(f"\nWrote {len(df)} rows → {args.out}")

    # Console preview: top by rho per gold
    for gn, sub in df.groupby("gold_name"):
        print(f"\n--- {gn} : top 10 by rho ---")
        cols_pre = ["model_id", "n_shared_full", "rho", "n_shared_dial_ext", "rho2"]
        print(sub[cols_pre].head(10).to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
