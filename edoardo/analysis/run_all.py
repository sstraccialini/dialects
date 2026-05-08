"""
Full ``edoardo/`` pipeline runner.

Steps:
    1. inventory       — list every model output under analysis/
    2. correlate       — Spearman ρ + Mantel test against every gold
    3. cluster_ari     — ARI / NMI / V-measure vs gold genealogical labels
    4. cka             — linear CKA + Procrustes for every (baseline, ft) pair
    5. shift           — Δρ per gold for every (A, B) pair within a method
    6. summary         — write a leaderboard combining everything

Run from the repo root:
    python -m edoardo.analysis.run_all                 # standard
    python -m edoardo.analysis.run_all --include-old   # include old_experiments/
    python -m edoardo.analysis.run_all --skip cka      # skip step
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import pandas as pd

from edoardo.analysis import (
    cka_baseline_vs_finetuned,
    cluster_agreement,
    correlate_with_gold,
    inventory_models,
    shift_analysis,
)


RESULTS_DIR_DEFAULT = Path(__file__).resolve().parents[1] / "results"

STEPS = ["inventory", "correlate", "cluster", "cka", "shift", "summary"]


def _run(step_name: str, fn, args_list: list[str]) -> int:
    print(f"\n[{step_name}]")
    try:
        rc = fn(args_list)
        return rc or 0
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0
    except Exception:
        traceback.print_exc()
        return 1


def _write_summary(out_dir: Path) -> None:
    """
    Produce a single human-readable leaderboard that combines:
      - best gold alignment per model (from correlation_with_gold.csv)
      - genealogical ARI (from cluster_agreement.csv)
      - shift signature vs first available baseline experiment per method
    """
    corr_csv = out_dir / "correlation_with_gold.csv"
    clust_csv = out_dir / "cluster_agreement.csv"
    if not corr_csv.exists() or not clust_csv.exists():
        print("  (summary skipped — missing inputs)")
        return

    corr = pd.read_csv(corr_csv)
    clust = pd.read_csv(clust_csv)

    # Best gold per model
    best = (corr.dropna(subset=["spearman_rho"])
                .sort_values("spearman_rho", ascending=False)
                .groupby("model_id").head(1)
                .rename(columns={"gold_name": "best_gold",
                                 "spearman_rho": "best_rho"})
                [["model_id", "method", "experiment", "variant_path",
                  "best_gold", "best_rho"]])

    # Pivot of all rho by gold
    rho_pivot = corr.pivot_table(index="model_id",
                                 columns="gold_name",
                                 values="spearman_rho",
                                 aggfunc="first")

    # Genealogical ARI
    g4 = clust[clust["gold_label_set"] == "genealogical_4way"][["model_id", "ari"]]
    g4 = g4.rename(columns={"ari": "ari_genealogical_4way"})
    rb = clust[clust["gold_label_set"] == "romance_binary"][["model_id", "ari"]]
    rb = rb.rename(columns={"ari": "ari_romance_binary"})

    summary = (best
               .merge(rho_pivot.reset_index(), on="model_id", how="left")
               .merge(g4, on="model_id", how="left")
               .merge(rb, on="model_id", how="left"))

    out_path = out_dir / "summary_leaderboard.csv"
    summary.to_csv(out_path, index=False, float_format="%.4f")
    print(f"  → {out_path}")

    # Console preview
    cols_preview = ["model_id", "best_gold", "best_rho",
                    "ari_genealogical_4way", "ari_romance_binary"]
    cols_preview = [c for c in cols_preview if c in summary.columns]
    print("\nLeaderboard preview:")
    print(summary[cols_preview].to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR_DEFAULT)
    ap.add_argument("--include-old", action="store_true")
    ap.add_argument("--baseline-experiment", type=str, default=None)
    ap.add_argument("--permutations", type=int, default=999)
    ap.add_argument("--skip", nargs="+", choices=STEPS, default=[])
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    common = []
    if args.include_old:
        common.append("--include-old")

    errors = 0
    if "inventory" not in args.skip:
        errors += _run("inventory", inventory_models.main,
                       common + ["--csv", str(args.out_dir / "hpc_inventory.csv")])
    if "correlate" not in args.skip:
        errors += _run("correlate", correlate_with_gold.main,
                       common + ["--out-dir", str(args.out_dir),
                                 "--permutations", str(args.permutations)])
    if "cluster" not in args.skip:
        errors += _run("cluster", cluster_agreement.main,
                       common + ["--out-dir", str(args.out_dir)])
    if "cka" not in args.skip:
        cka_args = common + ["--out-dir", str(args.out_dir)]
        if args.baseline_experiment:
            cka_args += ["--baseline-experiment", args.baseline_experiment]
        errors += _run("cka", cka_baseline_vs_finetuned.main, cka_args)
    if "shift" not in args.skip:
        sh_args = ["--out-dir", str(args.out_dir),
                   "--in-csv", str(args.out_dir / "correlation_with_gold.csv")]
        if args.baseline_experiment:
            sh_args += ["--baseline-experiment", args.baseline_experiment]
        errors += _run("shift", shift_analysis.main, sh_args)
    if "summary" not in args.skip:
        print("\n[summary]")
        _write_summary(args.out_dir)

    print(f"\nDone.  {len(STEPS) - len(args.skip) - errors}/"
          f"{len(STEPS) - len(args.skip)} steps succeeded.")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
