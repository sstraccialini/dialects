"""
Batch evaluator: correlate every model output found under
``analysis/<method>/experiments/<exp>/.../distances.csv`` against every
gold matrix in the directories given by ``--gold-dir``.

Two correlation metrics per (model, gold) pair, defined and computed in
``evaluation/_gold_correlation.py`` (so the per-experiment evaluation
in ``evaluation/run_evaluation`` uses identical logic):

    Spearman ρ (full matrix)        — full upper-triangle correlation.
    Spearman ρ (dialect ↔ external) — restricted to the dialect × external
                                      cross-block (italian and intra-dialect
                                      pairs are excluded).

Output: under ``--out-dir`` one CSV per gold:
    correlation_<gold_name>.csv
        method, experiment, variant, model_id,
        Spearman ρ (full matrix),
        Spearman ρ (dialect ↔ external)
plus a ``README.md`` explaining the metrics and how to read them per
gold type (lex vs geo).

By default no row filtering is applied; pass ``--min-shared-fraction``
> 0 to drop legacy experiments that share fewer than that fraction of
the gold's varieties (those produce fake-high ρ on tiny subsets).

CLI:
    python -m evaluation.correlate_against_gold \\
        --gold-dir gold/lexicostatistical/matrices gold/geographic/matrices \\
        --analysis-root analysis \\
        --out-dir gold/_correlations
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from evaluation._gold_correlation import (
    COL_RHO_DIAEXT,
    COL_RHO_FULL,
    correlate_against_gold,
    default_roles,
    load_gold,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Discovery
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
# README
# --------------------------------------------------------------------------- #

README_TEMPLATE = """\
# Correlations against gold reference matrices

One CSV per gold reference matrix.  Columns:

| Column | Meaning |
|---|---|
| `Spearman ρ (full matrix)` | Spearman on the full upper triangle of the shared distance matrix. |
| `Spearman ρ (dialect ↔ external)` | Spearman on the cross-block (dialect × external-non-Italian) only. |

Range −1 … +1.  Per-experiment versions of the same metrics are written
inside every method's `evaluation_results/.../gold_correlations.csv`.

## Reading per gold type

* **Lexicostatistical (LDND on Swadesh-207).**  High ρ = the model
  recovers lexical-cognate similarity.  Central metric for language
  similarity.

* **Geographic (Haversine).**  A *language-aware* model is expected to
  score MODERATELY on the full matrix and NEAR-ZERO or NEGATIVELY on the
  dialect↔external block — Slovenian is geographically near Veneto but
  linguistically Slavic.  A negative ρ on dialect↔external for the geo
  gold is a positive signal that the model captures language rather
  than geography.

## Sub-variety roles

Defined in ``gold/lexicostatistical/varieties.py``:
``dialect`` ∈ {{fur, lij, lmo, sc, scn, vec}};
``italian`` = {{ita}} (excluded from the dialect↔external column);
``external`` ∈ {{fra, spa, cat, deu, slv, eng}}.

## Files

{file_list}
"""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold-dir", type=Path, nargs="+", required=True,
                    help="One or more folders containing <name>.npz gold matrices.")
    ap.add_argument("--analysis-root", type=Path, default=REPO_ROOT / "analysis")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--min-shared-fraction", type=float, default=0.0,
                    help="If > 0, drop model rows whose shared-variety count "
                         "is below this fraction of the gold's variety count. "
                         "Default 0.0 = no filtering.")
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dialect_codes, external_codes = default_roles()

    gold_paths: List[Path] = []
    for d in args.gold_dir:
        gold_paths.extend(sorted(d.glob("*.npz")))
    if not gold_paths:
        print(f"No .npz gold found in {args.gold_dir}", file=sys.stderr)
        return 1

    dist_csvs = _iter_distance_csvs(args.analysis_root)
    print(f"Golds  : {[p.stem for p in gold_paths]}")
    print(f"Models : {len(dist_csvs)} distances.csv files under {args.analysis_root}")

    written: List[str] = []
    for gp in gold_paths:
        gold_mat, gold_labels, _ = load_gold(gp)
        n_gold = len(gold_labels)
        min_shared = max(4, int(round(n_gold * args.min_shared_fraction))) \
            if args.min_shared_fraction > 0 else 0

        rows: List[Dict] = []
        n_dropped = 0
        for dc in dist_csvs:
            try:
                model_mat, model_labels = _load_distance_matrix(dc)
            except Exception as exc:
                print(f"  skip {dc} — {exc}", file=sys.stderr)
                continue
            info = _parse_model_path(dc, args.analysis_root)
            rho_full, rho_dia, n_sh = correlate_against_gold(
                model_mat, model_labels,
                gold_mat,  gold_labels,
                dialect_codes, external_codes,
            )
            if min_shared and n_sh < min_shared:
                n_dropped += 1
                continue
            rows.append({
                "method":     info["method"],
                "experiment": info["experiment"],
                "variant":    info["variant"],
                "model_id":   info["model_id"],
                COL_RHO_FULL:   rho_full,
                COL_RHO_DIAEXT: rho_dia,
            })

        if not rows:
            print(f"  no valid (model, gold) pairs for {gp.stem}", file=sys.stderr)
            continue

        df = pd.DataFrame(rows).sort_values(COL_RHO_FULL, ascending=False)
        out_csv = args.out_dir / f"correlation_{gp.stem}.csv"
        df.to_csv(out_csv, index=False, float_format="%.4f")
        written.append(out_csv.name)
        suffix = f"  ({n_dropped} dropped for n_shared < {min_shared}/{n_gold})" if min_shared else ""
        print(f"\n→ {out_csv}  ({len(df)} rows){suffix}")
        print(df.head(10).to_string(index=False))

    readme_path = args.out_dir / "README.md"
    file_list = "\n".join(f"* `{n}`" for n in sorted(written))
    readme_path.write_text(README_TEMPLATE.format(file_list=file_list or "(none)"))
    print(f"\n→ {readme_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
