#!/usr/bin/env python3
"""
bootstrap_significance.py
=========================
Paired bootstrap significance testing for chrF++ (and, optionally, BLEU)
across (source -> dialect) translation systems that share the same N
sentence references.

For each target dialect, every available source language is compared
pairwise. Methodology follows Koehn (2004), *Statistical Significance Tests
for Machine Translation Evaluation*:

    1. Compute observed corpus chrF++ for system A and system B on the
       full N-sentence set; record delta = score_A - score_B.
    2. For each of B bootstrap iterations:
         a. Sample N sentence indices with replacement.
         b. Compute corpus chrF++ for system A on that subset.
         c. Compute corpus chrF++ for system B on the SAME subset
            (paired -- same indices for both systems).
         d. Record delta_b = score_A_b - score_B_b.
    3. Two-sided p-value:
            p_one = mean( sign(delta_b) != sign(delta_obs) )
            p_two = min(2 * p_one, 1.0)
    4. 95% CI on the delta from the bootstrap percentiles (2.5, 97.5).

We also compute Bonferroni-corrected p-values per (target, metric), since
a single target generates many pairwise tests.

File / naming convention mirrors compute_metrics.py:
    {SOURCE}-to-{DIALECT}.txt

Run:
    python bootstrap_significance.py \
        --pred_dir path/to/predictions \
        --gold     path/to/gold_table.csv \
        --out      bootstrap_chrf_significance.csv \
        --n_bootstrap 1000 \
        --n_jobs -1 \
        --seed 42
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Reuse loading / cleaning helpers from compute_metrics.py (same folder).
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from compute_metrics import (   # noqa: E402
    parse_filename,
    load_predictions,
    load_gold_table,
)

try:
    from sacrebleu.metrics import BLEU, CHRF
except ImportError:
    sys.exit("ERROR: sacrebleu not installed.  pip install sacrebleu")

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    def tqdm(it, **kw):  # type: ignore
        return it

try:
    import joblib
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False
    joblib = None  # type: ignore


# --------------------------------------------------------------------------- #
# Bootstrap primitives
# --------------------------------------------------------------------------- #
def _make_metric(metric: str):
    if metric == "chrfpp":
        return CHRF(word_order=2)
    if metric == "bleu":
        return BLEU()
    raise ValueError(f"unknown metric '{metric}'")


def _corpus_score(m, preds: List[str], refs: List[str]) -> float:
    return m.corpus_score(preds, [refs]).score


def _one_bootstrap_iter(
    preds_a: List[str],
    preds_b: List[str],
    refs: List[str],
    metric: str,
    iter_seed: int,
) -> float:
    """Single paired bootstrap iteration; returns delta = score_A - score_B
    on a resampled (with replacement) index set of size N. Recreates the
    metric object inside so it is picklable / safe across worker processes."""
    n = len(refs)
    rng = np.random.default_rng(iter_seed)
    idx = rng.integers(0, n, size=n)
    pa = [preds_a[j] for j in idx]
    pb = [preds_b[j] for j in idx]
    rr = [refs[j]    for j in idx]
    m = _make_metric(metric)
    sa = _corpus_score(m, pa, rr)
    sb = _corpus_score(m, pb, rr)
    return float(sa - sb)


def paired_bootstrap(
    preds_a: List[str],
    preds_b: List[str],
    refs: List[str],
    *,
    metric: str = "chrfpp",
    n_iter: int = 1000,
    seed: int = 42,
    n_jobs: int = 1,
) -> Dict[str, float]:
    """
    Paired bootstrap test on a corpus-level metric (chrF++ or BLEU).

    n_jobs : 1 = sequential; -1 = all CPU cores; k = k cores. Requires joblib.
    """
    n = len(refs)
    if not (len(preds_a) == len(preds_b) == n):
        raise ValueError(
            f"Length mismatch: |preds_a|={len(preds_a)}, "
            f"|preds_b|={len(preds_b)}, |refs|={n}"
        )

    m = _make_metric(metric)
    score_a = _corpus_score(m, preds_a, refs)
    score_b = _corpus_score(m, preds_b, refs)
    delta_obs = score_a - score_b

    rng = np.random.default_rng(seed)
    iter_seeds = rng.integers(0, 2**31 - 1, size=n_iter).tolist()

    use_parallel = (n_jobs != 1) and _HAS_JOBLIB
    if use_parallel:
        # Joblib loky backend handles process spawning + pickling.
        deltas = joblib.Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            joblib.delayed(_one_bootstrap_iter)(
                preds_a, preds_b, refs, metric, int(s)
            )
            for s in iter_seeds
        )
    else:
        if n_jobs != 1 and not _HAS_JOBLIB:
            print("[warn] joblib not installed; falling back to sequential. "
                  "pip install joblib for parallel bootstrap.", file=sys.stderr)
        deltas = []
        iterator = iter_seeds
        if _HAS_TQDM:
            iterator = tqdm(iter_seeds, desc=f"  bootstrap[{metric}]",
                            leave=False)
        for s in iterator:
            deltas.append(_one_bootstrap_iter(
                preds_a, preds_b, refs, metric, int(s)))

    deltas = np.asarray(deltas, dtype=np.float64)
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])

    if delta_obs > 0:
        p_one = float(np.mean(deltas <= 0))
    elif delta_obs < 0:
        p_one = float(np.mean(deltas >= 0))
    else:
        p_one = 0.5
    p_two = min(2.0 * p_one, 1.0)

    return {
        "score_a":     float(score_a),
        "score_b":     float(score_b),
        "delta":       float(delta_obs),
        "ci_low":      float(ci_low),
        "ci_high":     float(ci_high),
        "p_value":     float(p_two),
        "n_sentences": int(n),
        "n_bootstrap": int(n_iter),
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _align_to_min(systems: Dict[str, List[str]], refs: List[str]
                  ) -> Tuple[Dict[str, List[str]], List[str], int]:
    n = min(len(refs), *(len(v) for v in systems.values()))
    if any(len(v) != n for v in systems.values()) or len(refs) != n:
        systems = {k: v[:n] for k, v in systems.items()}
        refs = refs[:n]
    return systems, refs, n


def _drop_empty_pairs(
    systems: Dict[str, List[str]], refs: List[str]
) -> Tuple[Dict[str, List[str]], List[str]]:
    n = len(refs)
    keep = [i for i in range(n)
            if refs[i] and all(systems[s][i] for s in systems)]
    if len(keep) == n:
        return systems, refs
    systems = {k: [v[i] for i in keep] for k, v in systems.items()}
    refs = [refs[i] for i in keep]
    return systems, refs


def _bonferroni(p_values: List[float]) -> List[float]:
    k = len(p_values)
    return [min(p * k, 1.0) for p in p_values]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="Paired bootstrap significance testing for chrF++/BLEU "
                    "across source languages, per target dialect.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pred_dir", required=True, type=Path,
                   help="Folder containing {SOURCE}-to-{DIALECT}.txt files.")
    p.add_argument("--gold", required=True, type=Path,
                   help="Reference table (CSV / TSV / XLSX).")
    p.add_argument("--out", default="bootstrap_chrf_significance.csv", type=Path,
                   help="Output CSV.")
    p.add_argument("--n_bootstrap", type=int, default=1000,
                   help="Number of bootstrap resamples per pair.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (a per-pair seed is derived from this).")
    p.add_argument("--metrics", nargs="+", choices=["chrfpp", "bleu"],
                   default=["chrfpp"],
                   help="Which metric(s) to bootstrap. chrF++ is primary.")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Significance threshold for the printed flags.")
    p.add_argument("--on_mismatch", choices=["skip", "truncate"], default="truncate",
                   help="How to reconcile differing lengths across systems "
                        "and the reference column for one dialect.")
    p.add_argument("--include_pairs", nargs="*", default=None,
                   help="Optional explicit list of source pairs to test, "
                        "format src_a:src_b. If omitted, ALL pairs are run.")
    p.add_argument("--restrict_targets", nargs="*", default=None,
                   help="Optional list of target dialects to restrict to "
                        "(default: every target found in pred_dir).")
    p.add_argument("--n_jobs", type=int, default=1,
                   help="CPU cores for parallel bootstrap. 1 = sequential, "
                        "-1 = all cores. Requires joblib.")
    args = p.parse_args()

    if args.n_jobs != 1 and not _HAS_JOBLIB:
        print("[warn] --n_jobs requested but joblib not installed; sequential.",
              file=sys.stderr)

    # ---- gold ---------------------------------------------------------- #
    if not args.gold.exists():
        sys.exit(f"ERROR: gold not found: {args.gold}")
    gold_df = load_gold_table(args.gold)
    print(f"[info] gold table       = {args.gold}  shape={gold_df.shape}")
    print(f"[info] gold columns     = {list(gold_df.columns)}")
    print(f"[info] n_bootstrap      = {args.n_bootstrap}")
    print(f"[info] n_jobs           = {args.n_jobs}"
          f"{'  (joblib OK)' if _HAS_JOBLIB else '  (joblib MISSING)'}")

    # ---- group prediction files by target dialect ---------------------- #
    if not args.pred_dir.is_dir():
        sys.exit(f"ERROR: pred_dir not a folder: {args.pred_dir}")
    files = sorted(args.pred_dir.glob("*.txt"))
    if not files:
        sys.exit(f"ERROR: no .txt files in {args.pred_dir}")

    by_target: Dict[str, Dict[str, List[str]]] = {}
    for fpath in files:
        parsed = parse_filename(fpath)
        if parsed is None:
            print(f"[warn] skipping {fpath.name} (filename pattern)",
                  file=sys.stderr)
            continue
        src, tgt = parsed
        if args.restrict_targets and tgt not in args.restrict_targets:
            continue
        if tgt not in gold_df.columns:
            print(f"[warn] skipping {fpath.name}: '{tgt}' not in gold columns",
                  file=sys.stderr)
            continue
        try:
            preds = load_predictions(fpath)
        except Exception as e:
            print(f"[warn] skipping {fpath.name}: read error {e}",
                  file=sys.stderr)
            continue
        by_target.setdefault(tgt, {})[src] = preds

    if not by_target:
        sys.exit("ERROR: no usable prediction files found.")

    explicit_pairs = None
    if args.include_pairs:
        explicit_pairs = set()
        for s in args.include_pairs:
            try:
                a, b = s.split(":")
            except ValueError:
                sys.exit(f"ERROR: bad --include_pairs entry: {s}")
            explicit_pairs.add(frozenset({a.strip().lower(), b.strip().lower()}))

    # ---- run bootstrap per (target, src_a, src_b) --------------------- #
    rows: List[dict] = []
    base_rng = np.random.default_rng(args.seed)

    for tgt in sorted(by_target):
        sources = sorted(by_target[tgt])
        if len(sources) < 2:
            print(f"[info] target {tgt}: only {len(sources)} source(s) — skipping.")
            continue

        refs_full = [str(r).strip() for r in gold_df[tgt].tolist()]
        sys_dict = {s: by_target[tgt][s] for s in sources}

        n_pre = len(refs_full)
        sys_dict, refs, n = _align_to_min(sys_dict, refs_full)
        if any(len(by_target[tgt][s]) != n_pre for s in sources) or len(refs_full) != n_pre:
            if args.on_mismatch == "skip":
                print(f"[warn] target {tgt}: length mismatch — skipping.",
                      file=sys.stderr)
                continue
            print(f"[warn] target {tgt}: length mismatch reconciled → N={n}.",
                  file=sys.stderr)
        sys_dict, refs = _drop_empty_pairs(sys_dict, refs)
        n = len(refs)
        if n < 5:
            print(f"[warn] target {tgt}: only {n} non-empty rows — skipping.",
                  file=sys.stderr)
            continue

        print(f"\n=== TARGET: {tgt}   sources={sources}   N={n} ===")

        pair_list = list(combinations(sources, 2))
        if explicit_pairs is not None:
            pair_list = [p for p in pair_list if frozenset(p) in explicit_pairs]
        if not pair_list:
            print(f"  (no pairs after --include_pairs filtering)")
            continue

        per_target_rows: List[dict] = []
        for src_a, src_b in pair_list:
            for metric in args.metrics:
                pair_seed = int(base_rng.integers(0, 2**31 - 1))
                print(f"  [{metric}] {src_a} vs {src_b}  (seed={pair_seed})")
                res = paired_bootstrap(
                    preds_a=sys_dict[src_a],
                    preds_b=sys_dict[src_b],
                    refs=refs,
                    metric=metric,
                    n_iter=args.n_bootstrap,
                    seed=pair_seed,
                    n_jobs=args.n_jobs,
                )
                row = {
                    "target_dialect": tgt,
                    "metric":         metric,
                    "source_a":       src_a,
                    "source_b":       src_b,
                    "score_a":        round(res["score_a"], 4),
                    "score_b":        round(res["score_b"], 4),
                    "delta":          round(res["delta"], 4),
                    "ci_low":         round(res["ci_low"], 4),
                    "ci_high":        round(res["ci_high"], 4),
                    "p_value":        round(res["p_value"], 6),
                    "n_sentences":    res["n_sentences"],
                    "n_bootstrap":    res["n_bootstrap"],
                    "seed":           pair_seed,
                }
                per_target_rows.append(row)
                ci_str = f"[{res['ci_low']:+.2f}, {res['ci_high']:+.2f}]"
                star = "*" if res["p_value"] < args.alpha else " "
                print(f"     A={res['score_a']:6.2f}  B={res['score_b']:6.2f}  "
                      f"Δ={res['delta']:+6.2f}  CI95={ci_str}  "
                      f"p={res['p_value']:.4f} {star}")

        for metric in args.metrics:
            sub = [r for r in per_target_rows if r["metric"] == metric]
            if not sub:
                continue
            ps = [r["p_value"] for r in sub]
            ps_bonf = _bonferroni(ps)
            for r, pb in zip(sub, ps_bonf):
                r["p_value_bonferroni"] = round(pb, 6)
                r["significant_alpha"]  = bool(r["p_value"] < args.alpha)
                r["significant_bonf"]   = bool(pb < args.alpha)

        rows.extend(per_target_rows)

    if not rows:
        sys.exit("\nERROR: no comparisons were run.")

    df = pd.DataFrame(rows)
    df = df.sort_values(["target_dialect", "metric", "source_a", "source_b"]
                        ).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"\n[ok] wrote {len(df)} rows -> {args.out}")

    print("\n" + "=" * 96)
    print(f"PAIRED BOOTSTRAP SUMMARY  (alpha={args.alpha}, "
          f"B={args.n_bootstrap}, paired by sentence index)")
    print("=" * 96)
    cols = ["target_dialect", "metric", "source_a", "source_b",
            "score_a", "score_b", "delta", "ci_low", "ci_high",
            "p_value", "p_value_bonferroni",
            "significant_alpha", "significant_bonf"]
    cols = [c for c in cols if c in df.columns]
    with pd.option_context("display.max_rows", None,
                           "display.max_columns", None,
                           "display.width", 220):
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
