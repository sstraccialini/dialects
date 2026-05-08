"""
Linear CKA between baseline and fine-tuned variants of the same model family.

Question this answers: when we MLM/TLM fine-tune XLM-R or CANINE, how much
of the original embedding geometry survives?  CKA ∈ [0, 1]: 1 = identical
up to orthogonal transform + scaling, 0 = unrelated.

Pairs are detected by ``method``: any method with ≥ 2 experiments has each
non-baseline experiment compared against every other (we don't know which
"is" the baseline a priori — the user picks via ``--baseline-experiment``,
otherwise every pair is reported and downstream code can pick).

Inputs : ``analysis/<method>/experiments/<exp>/method_outputs/[<variant>/]variety_vectors.npz``
Outputs: ``edoardo/results/cka_baseline_vs_finetuned.csv`` with rows
    method, A_experiment, A_variant, B_experiment, B_variant,
    n_shared_varieties, linear_cka, procrustes_disparity

Run from the repo root:
    python -m edoardo.analysis.cka_baseline_vs_finetuned
    python -m edoardo.analysis.cka_baseline_vs_finetuned --baseline-experiment baseline
"""
from __future__ import annotations

import argparse
import sys
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from edoardo.analysis.load_models import ModelOutput, discover_models
from evaluation.compare_methods import cka_score, procrustes_disparity


RESULTS_DIR_DEFAULT = Path(__file__).resolve().parents[1] / "results"


def _can_compare(a: ModelOutput, b: ModelOutput) -> bool:
    return (a.variety_vectors_npz is not None
            and b.variety_vectors_npz is not None)


def _maybe_compute_pair(a: ModelOutput, b: ModelOutput) -> dict | None:
    if not _can_compare(a, b):
        return None
    try:
        Xa, ca = a.load_vectors()
        Xb, cb = b.load_vectors()
    except Exception as exc:
        warnings.warn(f"{a.short_id} vs {b.short_id}: load failed ({exc})", stacklevel=2)
        return None

    shared = sorted(set(ca) & set(cb))
    if len(shared) < 3:
        return None

    try:
        cka = cka_score(Xa, ca, Xb, cb)
    except Exception:
        cka = float("nan")
    try:
        disp = procrustes_disparity(Xa, ca, Xb, cb)
    except Exception:
        disp = float("nan")
    return {
        "method": a.method,
        "A_experiment": a.experiment,
        "A_variant":    a.variant_path,
        "B_experiment": b.experiment,
        "B_variant":    b.variant_path,
        "A_short_id":   a.short_id,
        "B_short_id":   b.short_id,
        "n_shared_varieties": len(shared),
        "linear_cka":   cka,
        "procrustes_disparity": disp,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR_DEFAULT)
    ap.add_argument("--include-old", action="store_true")
    ap.add_argument("--baseline-experiment", type=str, default=None,
                    help="If given, only compare each fine-tuned experiment to "
                         "the experiment whose name == this value (per method).")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    models = discover_models(include_old=args.include_old)
    if not models:
        print("No models found.", file=sys.stderr)
        return 1

    by_method: dict[str, List[ModelOutput]] = defaultdict(list)
    for m in models:
        by_method[m.method].append(m)

    rows: list[dict] = []
    for method, ms in by_method.items():
        if len(ms) < 2:
            continue
        if args.baseline_experiment:
            base = [m for m in ms if m.experiment == args.baseline_experiment]
            others = [m for m in ms if m.experiment != args.baseline_experiment]
            if not base:
                warnings.warn(
                    f"{method}: no experiment named '{args.baseline_experiment}'",
                    stacklevel=2)
                continue
            for b in base:
                for o in others:
                    pair = _maybe_compute_pair(b, o)
                    if pair:
                        rows.append(pair)
        else:
            for a, b in combinations(ms, 2):
                pair = _maybe_compute_pair(a, b)
                if pair:
                    rows.append(pair)

    if not rows:
        print("No comparable pairs (need ≥2 experiments per method "
              "with variety_vectors.npz).", file=sys.stderr)
        # Still write an empty CSV so downstream tooling sees the file.
        pd.DataFrame(rows).to_csv(args.out_dir / "cka_baseline_vs_finetuned.csv",
                                  index=False)
        return 0

    df = pd.DataFrame(rows).sort_values(["method", "linear_cka"], ascending=[True, False])
    out_path = args.out_dir / "cka_baseline_vs_finetuned.csv"
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"  → {out_path}")

    print("\nLow-CKA pairs (most-changed embedding spaces):")
    cols = ["method", "A_experiment", "A_variant",
            "B_experiment", "B_variant",
            "linear_cka", "procrustes_disparity"]
    print(df.sort_values("linear_cka").head(15)[cols].to_string(index=False))

    print("\nHigh-CKA pairs (least-changed):")
    print(df.sort_values("linear_cka", ascending=False).head(10)[cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
