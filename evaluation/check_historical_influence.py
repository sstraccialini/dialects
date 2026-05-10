"""
Evaluate every model under ``analysis/`` against the team-curated
historical-influence gold (set-based, NOT a distance matrix).

For each (model, dialect) pair we compute the model's top-3 closest
non-dialect varieties and check overlap with the documented top-3
historical influences in ``gold/historical_influence/influences.csv``.

Metrics
-------
* Per (model, dialect):
    overlap_count  — |gold ∩ predicted| ∈ {0, 1, 2, 3}
    precision_at_3 — overlap_count / 3 ∈ {0.0, 0.33, 0.67, 1.0}
* Per model (averaged over the dialects with valid gold rows):
    Mean Precision@3 (historical influence)

Random-chance baseline: with 7 non-dialect candidates and 3 picks,
expected overlap is 3/7 ≈ 0.429 per dialect.

Output
------
* ``--out-dir/historical_influence_summary.csv``
    method, experiment, variant, model_id,
    Mean Precision@3 (historical influence)
* ``--out-dir/historical_influence_detail.csv``
    method, experiment, variant, model_id, dialect,
    gold_top3, model_top3, overlap_count, precision_at_3

CLI:
    python -m evaluation.check_historical_influence \\
        --gold-csv gold/historical_influence/influences.csv \\
        --analysis-root analysis \\
        --out-dir gold/historical_influence/results
"""
from __future__ import annotations

import argparse
import csv
import importlib
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

COL_MEAN_P3 = "Mean Precision@3 (historical influence)"


# --------------------------------------------------------------------------- #
# I/O helpers (reused conceptually from correlate_against_gold.py)
# --------------------------------------------------------------------------- #

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
        "experiment": experiment,
        "variant": variant,
        "model_id": short_id,
    }


def _load_distance_matrix(path: Path) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(path, index_col=0)
    return df.values.astype(np.float64), [str(x) for x in df.index]


# --------------------------------------------------------------------------- #
# Gold loading
# --------------------------------------------------------------------------- #

def _load_gold(csv_path: Path) -> Dict[str, Set[str]]:
    """Return ``{dialect: {influence_1, influence_2, influence_3}}``."""
    out: Dict[str, Set[str]] = {}
    with csv_path.open(encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        for row in rd:
            d = row["dialect"].strip()
            inf = {row.get(k, "").strip()
                   for k in ("influence_1", "influence_2", "influence_3")}
            inf.discard("")
            if not d or not inf:
                continue
            out[d] = inf
    return out


# --------------------------------------------------------------------------- #
# Per-model evaluation
# --------------------------------------------------------------------------- #

def _model_top3(model_dist: np.ndarray,
                model_labels: List[str],
                dialect: str,
                candidate_codes: List[str]) -> List[str]:
    """Top-3 closest candidates to ``dialect`` from ``candidate_codes``,
    using the model's distance row.  Returns at most 3 codes; fewer if
    the model is missing some candidates from its label set.
    """
    if dialect not in model_labels:
        return []
    i = model_labels.index(dialect)
    candidates_present = [c for c in candidate_codes if c in model_labels]
    if len(candidates_present) < 3:
        return candidates_present
    dist_to_cand = [(c, model_dist[i, model_labels.index(c)]) for c in candidates_present]
    dist_to_cand.sort(key=lambda x: x[1])
    return [c for c, _ in dist_to_cand[:3]]


def evaluate_model(model_dist: np.ndarray, model_labels: List[str],
                   gold: Dict[str, Set[str]],
                   dialect_codes: List[str],
                   candidate_codes: List[str]
                   ) -> Tuple[float, List[Dict]]:
    """Return (mean_precision_at_3, per_dialect_rows)."""
    rows: List[Dict] = []
    precisions: List[float] = []
    for d in dialect_codes:
        if d not in gold or d not in model_labels:
            continue
        gold_set = gold[d]
        pred = _model_top3(model_dist, model_labels, d, candidate_codes)
        if not pred:
            continue
        overlap = gold_set & set(pred)
        prec = len(overlap) / 3.0
        precisions.append(prec)
        rows.append({
            "dialect": d,
            "gold_top3": ",".join(sorted(gold_set)),
            "model_top3": ",".join(sorted(pred)),
            "overlap_count": len(overlap),
            "precision_at_3": prec,
        })
    mean_p3 = float(np.mean(precisions)) if precisions else float("nan")
    return mean_p3, rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold-csv", type=Path,
                    default=REPO_ROOT / "gold" / "historical_influence" / "influences.csv")
    ap.add_argument("--analysis-root", type=Path, default=REPO_ROOT / "analysis")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "gold" / "historical_influence" / "results")
    ap.add_argument("--varieties-module", type=str,
                    default="gold.lexicostatistical.varieties",
                    help="Module exporting DIALECT_CODES + EXTERNAL_CODES.")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    roles = importlib.import_module(args.varieties_module)
    dialect_codes: List[str] = list(roles.DIALECT_CODES)
    # Italian is intentionally EXCLUDED from the candidate pool — it is not
    # listed among any dialect's documented historical influences, and we
    # do not want a model to score on the trivial "ita is closest to every
    # dialect" answer.  Candidates are only the non-Italian external codes.
    candidate_codes: List[str] = list(roles.EXTERNAL_CODES)

    gold = _load_gold(args.gold_csv)
    if not gold:
        print(f"Empty gold at {args.gold_csv}", file=sys.stderr)
        return 1
    print(f"Gold dialects: {sorted(gold.keys())}")
    print(f"Candidate codes: {candidate_codes}")
    print(f"Random-chance baseline (3 picks of {len(candidate_codes)}): "
          f"{3.0 / len(candidate_codes):.3f}")

    dist_csvs = _iter_distance_csvs(args.analysis_root)
    print(f"Models: {len(dist_csvs)} distances.csv files")

    summary_rows: List[Dict] = []
    detail_rows: List[Dict] = []

    for dc in dist_csvs:
        try:
            model_dist, model_labels = _load_distance_matrix(dc)
        except Exception as exc:
            print(f"  skip {dc} — {exc}", file=sys.stderr)
            continue
        info = _parse_model_path(dc, args.analysis_root)
        mean_p3, per_dialect = evaluate_model(model_dist, model_labels,
                                              gold, dialect_codes, candidate_codes)
        summary_rows.append({
            "method":     info["method"],
            "experiment": info["experiment"],
            "variant":    info["variant"],
            "model_id":   info["model_id"],
            COL_MEAN_P3:  mean_p3,
        })
        for r in per_dialect:
            detail_rows.append({
                "method":     info["method"],
                "experiment": info["experiment"],
                "variant":    info["variant"],
                "model_id":   info["model_id"],
                **r,
            })

    if not summary_rows:
        print("No models could be evaluated.", file=sys.stderr)
        return 1

    sdf = pd.DataFrame(summary_rows).sort_values(COL_MEAN_P3, ascending=False)
    out_summary = args.out_dir / "historical_influence_summary.csv"
    sdf.to_csv(out_summary, index=False, float_format="%.4f")
    print(f"\n→ {out_summary}")
    print(sdf.head(10).to_string(index=False))

    ddf = pd.DataFrame(detail_rows)
    out_detail = args.out_dir / "historical_influence_detail.csv"
    ddf.to_csv(out_detail, index=False, float_format="%.4f")
    print(f"\n→ {out_detail}  ({len(ddf)} rows)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
