"""
Merge per-experiment ``bootstrap_results.csv`` files with the Mantel
p-values into the final cross-method tables.

Reads:
  * ``analysis/<method>/experiments/<exp>/evaluation_results/flores/centroid/bootstrap_results.csv``
    (one per experiment; produced by ``analysis/<method>/core/bootstrap.py``)
  * ``gold/_correlations/correlation_<gold>_with_pvalue.csv``
    (produced by ``evaluation/mantel_pvalues.py``)

Writes, one per gold:
  ``gold/_correlations/correlation_<gold>_with_bootstrap.csv`` with columns
    method, experiment, variant, model_id,
    Spearman ρ (full matrix),
    CI 2.5% (full), CI 97.5% (full),
    Mantel p (full matrix),
    Spearman ρ (dialect ↔ external),
    CI 2.5% (d↔e), CI 97.5% (d↔e),
    Mantel p (dialect ↔ external),
    n_boot

CLI:
    python -m evaluation.aggregate_bootstrap \\
        --analysis-root analysis \\
        --pvalue-dir gold/_correlations \\
        --out-dir gold/_correlations
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _iter_bootstrap_csvs(analysis_root: Path) -> List[Path]:
    out: List[Path] = []
    for method_dir in sorted(p for p in analysis_root.iterdir() if p.is_dir()):
        if method_dir.name.startswith("_"):
            continue
        er_root = method_dir / "experiments"
        if not er_root.exists():
            continue
        out.extend(sorted(er_root.rglob(
            "evaluation_results/flores/centroid/bootstrap_results.csv"
        )))
    return out


def _parse_model_path(bootstrap_csv: Path, analysis_root: Path) -> Dict[str, str]:
    rel = bootstrap_csv.relative_to(analysis_root).parts
    method     = rel[0]
    experiment = rel[2] if len(rel) > 2 else "?"
    try:
        er_idx = rel.index("evaluation_results")
        variant = "/".join(rel[er_idx + 1 : -1])
    except ValueError:
        variant = ""
    short_id = f"{method}__{experiment}"
    if variant:
        short_id += "__" + variant.replace("/", "_")
    return {"method": method, "experiment": experiment,
            "variant": variant, "model_id": short_id}


def _wide_row(boot_df: pd.DataFrame, info: Dict[str, str], gold: str) -> Dict | None:
    """Pivot the long boot_df for one (experiment, gold) into one wide row."""
    sub = boot_df[boot_df["gold"] == gold]
    if sub.empty:
        return None
    full = sub[sub["block"] == "full"]
    dia  = sub[sub["block"] == "dialect_external"]
    if full.empty or dia.empty:
        return None
    f = full.iloc[0]; d = dia.iloc[0]
    return {
        **info,
        "Spearman ρ (full matrix)":          f["rho_observed"],
        "CI 2.5% (full)":                    f["rho_lo"],
        "CI 97.5% (full)":                   f["rho_hi"],
        "Spearman ρ (dialect ↔ external)":   d["rho_observed"],
        "CI 2.5% (d↔e)":                     d["rho_lo"],
        "CI 97.5% (d↔e)":                    d["rho_hi"],
        "n_boot":                            int(f["n_boot"]),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis-root", type=Path, default=REPO_ROOT / "analysis")
    ap.add_argument("--pvalue-dir",    type=Path, default=REPO_ROOT / "gold" / "_correlations",
                    help="Folder containing correlation_<gold>_with_pvalue.csv")
    ap.add_argument("--out-dir",       type=Path, default=REPO_ROOT / "gold" / "_correlations")
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    csvs = _iter_bootstrap_csvs(args.analysis_root)
    if not csvs:
        print("No bootstrap_results.csv found. "
              "Run analysis/<method>/core/bootstrap.py first.", file=sys.stderr)
        return 1

    # Discover golds via the long-format bootstrap files themselves.
    sample = pd.read_csv(csvs[0])
    golds = sorted(sample["gold"].unique())
    print(f"Models : {len(csvs)} bootstrap_results.csv under {args.analysis_root}")
    print(f"Golds  : {golds}")

    for gold in golds:
        rows: List[Dict] = []
        for cp in csvs:
            try:
                bdf = pd.read_csv(cp)
            except Exception as exc:
                print(f"  skip {cp} — {exc}", file=sys.stderr)
                continue
            info = _parse_model_path(cp, args.analysis_root)
            r = _wide_row(bdf, info, gold)
            if r is not None:
                rows.append(r)
        if not rows:
            continue
        df = pd.DataFrame(rows)

        # Merge in Mantel p-values if available.
        pval_csv = args.pvalue_dir / f"correlation_{gold}_with_pvalue.csv"
        if pval_csv.exists():
            p = pd.read_csv(pval_csv)[[
                "model_id",
                "Mantel p (full matrix)",
                "Mantel p (dialect ↔ external)",
            ]]
            df = df.merge(p, on="model_id", how="left")
        else:
            df["Mantel p (full matrix)"] = float("nan")
            df["Mantel p (dialect ↔ external)"] = float("nan")
            print(f"  WARNING: {pval_csv.name} not found — Mantel cols left NaN",
                  file=sys.stderr)

        # Column order.
        cols = [
            "method", "experiment", "variant", "model_id",
            "Spearman ρ (full matrix)",
            "CI 2.5% (full)", "CI 97.5% (full)",
            "Mantel p (full matrix)",
            "Spearman ρ (dialect ↔ external)",
            "CI 2.5% (d↔e)", "CI 97.5% (d↔e)",
            "Mantel p (dialect ↔ external)",
            "n_boot",
        ]
        df = df[cols].sort_values("Spearman ρ (full matrix)", ascending=False)
        out_csv = args.out_dir / f"correlation_{gold}_with_bootstrap.csv"
        df.to_csv(out_csv, index=False, float_format="%.4f")
        print(f"\n→ {out_csv}  ({len(df)} rows)")
        print(df.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
