"""
Shift analysis: how does each fine-tuning recipe move the embedding's
alignment with each gold proxy?

Reads ``edoardo/results/correlation_with_gold.csv`` (produced by
``correlate_with_gold.py``), groups by method, and for every pair
(experiment_A, experiment_B) reports

    Δρ_g = ρ(B, gold_g) - ρ(A, gold_g)

for every gold ``g``.  A useful interpretation:
    * positive Δρ_genetic  ⇒ B captures genealogy better than A
    * negative Δρ_lexical  ⇒ B captures lexical surface less than A
    * the *signature* of Δρ across golds is the dimensional shift.

Output: ``edoardo/results/shift_analysis.csv``.

Optional baseline pinning: if ``--baseline-experiment`` is given,
only A=baseline rows are emitted.

Run:
    python -m edoardo.analysis.shift_analysis
    python -m edoardo.analysis.shift_analysis --baseline-experiment mlm_wiki_to_flores_oldi
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd


RESULTS_DIR_DEFAULT = Path(__file__).resolve().parents[1] / "results"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-csv", type=Path,
                    default=RESULTS_DIR_DEFAULT / "correlation_with_gold.csv")
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR_DEFAULT)
    ap.add_argument("--baseline-experiment", type=str, default=None)
    args = ap.parse_args(argv)

    if not args.in_csv.exists():
        print(f"Missing {args.in_csv}.  Run "
              f"`python -m edoardo.analysis.correlate_with_gold` first.",
              file=sys.stderr)
        return 1

    df = pd.read_csv(args.in_csv)
    rows = []
    for method, sub in df.groupby("method"):
        # rows of (experiment, variant) per method
        keys = list(sub.groupby(["experiment", "variant_path"]).groups.keys())
        if len(keys) < 2:
            continue

        if args.baseline_experiment:
            base_keys = [k for k in keys if k[0] == args.baseline_experiment]
            other_keys = [k for k in keys if k[0] != args.baseline_experiment]
            if not base_keys:
                continue
            pairs = [(b, o) for b in base_keys for o in other_keys]
        else:
            pairs = list(combinations(keys, 2))

        for (a_exp, a_var), (b_exp, b_var) in pairs:
            a = sub[(sub["experiment"] == a_exp) & (sub["variant_path"] == a_var)]
            b = sub[(sub["experiment"] == b_exp) & (sub["variant_path"] == b_var)]
            joined = a.merge(b, on="gold_name", suffixes=("_A", "_B"))
            for _, r in joined.iterrows():
                rho_a = r["spearman_rho_A"]
                rho_b = r["spearman_rho_B"]
                rows.append({
                    "method": method,
                    "A_experiment": a_exp, "A_variant": a_var,
                    "B_experiment": b_exp, "B_variant": b_var,
                    "gold_name": r["gold_name"],
                    "rho_A": rho_a,
                    "rho_B": rho_b,
                    "delta_rho": (rho_b - rho_a) if pd.notna(rho_a) and pd.notna(rho_b) else float("nan"),
                })

    if not rows:
        print("No fine-tunable pairs found.", file=sys.stderr)
        pd.DataFrame(rows).to_csv(args.out_dir / "shift_analysis.csv", index=False)
        return 0

    out_df = pd.DataFrame(rows).sort_values(["method", "A_experiment", "B_experiment", "gold_name"])
    out_path = args.out_dir / "shift_analysis.csv"
    out_df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"  → {out_path}")

    # Human-readable: top losses and gains
    print("\nLargest gains (Δρ > 0): fine-tuning *increased* alignment with gold")
    gains = out_df.dropna(subset=["delta_rho"]).sort_values("delta_rho", ascending=False).head(15)
    print(gains[["method", "A_experiment", "B_experiment", "gold_name",
                 "rho_A", "rho_B", "delta_rho"]].to_string(index=False))
    print("\nLargest drops (Δρ < 0): fine-tuning *decreased* alignment with gold")
    drops = out_df.dropna(subset=["delta_rho"]).sort_values("delta_rho").head(15)
    print(drops[["method", "A_experiment", "B_experiment", "gold_name",
                 "rho_A", "rho_B", "delta_rho"]].to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
