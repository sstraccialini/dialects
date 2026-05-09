"""
Correlate every embedding-method output under ``analysis/`` against each
gold-reference distance matrix found under ``--gold-dir``.

Two correlation metrics are reported per (model, gold) pair, written
into separate CSVs (one per gold):

    Spearman ρ (full matrix)       — Spearman rank correlation on the
        FULL upper triangle of the shared distance matrix.  Captures ALL
        pairwise relationships between the varieties present in both
        the model output and the gold (so for our 13-variety setup, on
        the 13×12/2 = 78 unordered pairs).

    Spearman ρ (dialect ↔ external) — Spearman correlation restricted to
        the cross-block of pairs where one variety is a "dialect" and the
        other is an "external" (non-Italian, non-dialect) language.
        Excludes intra-dialect pairs (e.g. lij↔vec) and pairs involving
        standard Italian.  This is the harder signal because it crosses
        the genealogical boundary; for our 6 dialects × 6 external
        languages it is computed on 36 pairs.

Roles are read from ``gold/lexicostatistical/varieties.py``.  When the
variety set grows (more Wikipedia languages added), update that file
and this script adapts automatically.

Models are discovered by globbing
``analysis/*/experiments/*/evaluation_results/**/distances.csv``.
For each (model, gold), the analysis works on the SHARED label set so
that legacy 16-variety runs are correctly restricted to the gold's
codes before correlation.

Output: under ``--out-dir`` one CSV per gold:
    correlation_<gold_name>.csv     columns:
        method, experiment, variant, model_id,
        Spearman ρ (full matrix),
        Spearman ρ (dialect ↔ external)
plus a ``README.md`` explaining the metrics.

CLI:
    python -m evaluation.correlate_against_gold \\
        --gold-dir gold/lexicostatistical/matrices \\
        --analysis-root analysis \\
        --out-dir gold/lexicostatistical/results
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


# Column names — descriptive, used directly as CSV headers.
COL_RHO_FULL   = "Spearman ρ (full matrix)"
COL_RHO_DIAEXT = "Spearman ρ (dialect ↔ external)"


def correlate_one(model_dist: np.ndarray, model_labels: List[str],
                  gold_dist: np.ndarray,  gold_labels:  List[str],
                  dialect_codes: List[str],
                  external_codes: List[str]) -> Dict[str, float]:
    """Compute the two correlation metrics for one (model, gold) pair."""
    shared = [c for c in gold_labels if c in model_labels]
    if len(shared) < 4:
        return {COL_RHO_FULL: float("nan"), COL_RHO_DIAEXT: float("nan")}

    md, _ = _restrict(model_dist, model_labels, shared)
    gd, _ = _restrict(gold_dist,  gold_labels,  shared)

    rho_full = _spearman(_full_triangle(md), _full_triangle(gd))

    d_set = [c for c in dialect_codes if c in shared]
    e_set = [c for c in external_codes if c in shared]
    rho_diaext = _spearman(
        _cross_block(md, shared, d_set, e_set),
        _cross_block(gd, shared, d_set, e_set),
    )

    return {COL_RHO_FULL: rho_full, COL_RHO_DIAEXT: rho_diaext}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

README_TEMPLATE = """\
# Correlations against gold reference matrices

This folder contains one CSV per gold reference matrix.  Each CSV reports
how well every model output (under ``analysis/<method>/experiments/<exp>/...``)
matches that gold, using two Spearman rank correlations:

| Column | What it measures |
|---|---|
| `Spearman ρ (full matrix)` | Spearman correlation on the FULL upper triangle of the shared distance matrix.  For 13 varieties this is 78 unordered pairs; it summarises how well the model recovers ALL pairwise relationships at once. |
| `Spearman ρ (dialect ↔ external)` | Spearman restricted to the cross-block of (dialect × external-non-Italian) pairs.  For 6 dialects × 6 external languages this is 36 pairs.  It excludes intra-dialect pairs and pairs involving standard Italian, isolating the harder genealogical-boundary-crossing signal. |

Range for both: −1 (anti-correlated) … 0 (random) … +1 (identical ordering).
ρ ≥ 0.7 is strong, 0.4–0.7 moderate, < 0.3 weak / noise.

Sub-variety roles are defined in ``gold/lexicostatistical/varieties.py``:
``dialect`` ∈ {{fur, lij, lmo, sc, scn, vec}};
``italian`` = {{ita}} (excluded from the dialect↔external column);
``external`` ∈ {{fra, spa, cat, deu, slv, eng}}.

When the variety set grows (more Wikipedia languages added), update
``varieties.py``, regenerate the gold matrices via the ``rebuild_*.slurm``
jobs, and rerun ``correlate_against_gold`` — the metric definitions and
column names stay the same.

## Files

{file_list}

## How to interpret a result

Look at one model row.  If `Spearman ρ (full matrix)` is high (e.g. 0.8)
the model captures the OVERALL similarity structure well.  If
`Spearman ρ (dialect ↔ external)` is also high, the model handles the
dialect-to-external-language relations specifically.  A model can score
high on the full matrix but low on the dialect↔external column when it
is good at intra-dialect distinctions but poor at relating dialects to
the standard languages around them — that is precisely the kind of
imbalance worth flagging in the paper.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold-dir", type=Path, required=True,
                    help="Folder with one or more <name>.npz gold matrices.")
    ap.add_argument("--analysis-root", type=Path, default=REPO_ROOT / "analysis",
                    help="Root of the analysis/ tree to scan for models.")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Folder where one CSV per gold is written.")
    ap.add_argument("--varieties-module", type=str,
                    default="gold.lexicostatistical.varieties",
                    help="Module exporting DIALECT_CODES + EXTERNAL_CODES.")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Roles for the dialect↔external metric
    import importlib
    roles = importlib.import_module(args.varieties_module)
    dialect_codes: List[str] = list(roles.DIALECT_CODES)
    external_codes: List[str] = list(roles.EXTERNAL_CODES)

    gold_paths = sorted(args.gold_dir.glob("*.npz"))
    if not gold_paths:
        print(f"No .npz gold found in {args.gold_dir}", file=sys.stderr)
        return 1
    dist_csvs = _iter_distance_csvs(args.analysis_root)
    print(f"Golds  : {[p.stem for p in gold_paths]}")
    print(f"Models : {len(dist_csvs)} distances.csv files found under {args.analysis_root}")

    written: List[str] = []
    for gp in gold_paths:
        gold_mat, gold_labels, _ = _load_gold(gp)
        rows: List[Dict] = []
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
            rows.append({
                "method":     info["method"],
                "experiment": info["experiment"],
                "variant":    info["variant"],
                "model_id":   info["model_id"],
                COL_RHO_FULL:   res[COL_RHO_FULL],
                COL_RHO_DIAEXT: res[COL_RHO_DIAEXT],
            })

        if not rows:
            print(f"  no valid (model, gold) pairs for {gp.stem}", file=sys.stderr)
            continue

        df = pd.DataFrame(rows)
        df = df.sort_values(COL_RHO_FULL, ascending=False)
        out_csv = args.out_dir / f"correlation_{gp.stem}.csv"
        df.to_csv(out_csv, index=False, float_format="%.4f")
        written.append(out_csv.name)
        print(f"\n→ {out_csv}  ({len(df)} rows)")
        print(df.head(10).to_string(index=False))

    # Write a README in the out dir explaining the columns
    readme_path = args.out_dir / "README.md"
    file_list = "\n".join(f"* `{n}`" for n in sorted(written))
    readme_path.write_text(README_TEMPLATE.format(file_list=file_list or "(none)"))
    print(f"\n→ {readme_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
