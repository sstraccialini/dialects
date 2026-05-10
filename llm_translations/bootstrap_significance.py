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

We also compute Bonferroni-corrected p-values per dialect, since a single
target generates many pairwise tests.

File / naming convention mirrors compute_metrics.py:
    {SOURCE}-to-{DIALECT}.txt

Run:
    python bootstrap_significance.py \
        --pred_dir path/to/predictions \
        --gold     path/to/gold_table.csv \
        --out      bootstrap_chrf_significance.csv \
        --n_bootstrap 1000 \
        --seed 42

Important: by design we only compare systems that share the same target
dialect (same reference column).  Cross-target comparisons are not valid.
"""
from __future__ import annotations

import argparse
import sys
import concurrent.futures
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


# --------------------------------------------------------------------------- #
# Paired bootstrap
# --------------------------------------------------------------------------- #
def _corpus_chrf(metric: CHRF, preds: List[str], refs: List[str]) -> float:
    return metric.corpus_score(preds, [refs]).score


def _corpus_bleu(metric: BLEU, preds: List[str], refs: List[str]) -> float:
    return metric.corpus_score(preds, [refs]).score


def _bootstrap_worker(args):
    seed, n_iter, preds_a, preds_b, refs, metric = args
    n = len(refs)
    if metric == "chrfpp":
        m = CHRF(word_order=2)
        score_fn = lambda p, r: _corpus_chrf(m, p, r)
    elif metric == "bleu":
        m = BLEU()
        score_fn = lambda p, r: _corpus_bleu(m, p, r)
    else:
        raise ValueError(f"unknown metric '{metric}'")

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        pa = [preds_a[j] for j in idx]
        pb = [preds_b[j] for j in idx]
        rr = [refs[j]    for j in idx]
        sa = score_fn(pa, rr)
        sb = score_fn(pb, rr)
        deltas[i] = sa - sb
    return deltas


def paired_bootstrap(
    preds_a: List[str],
    preds_b: List[str],
    refs: List[str],
    *,
    metric: str = "chrfpp",
    n_iter: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Paired bootstrap test on a corpus-level metric (chrF++ or BLEU).

    Parameters
    ----------
    preds_a, preds_b : aligned predictions for two systems on the SAME refs.
    refs             : reference sentences (length N).
    metric           : "chrfpp" or "bleu".
    n_iter           : number of bootstrap resamples (default 1000).
    seed             : RNG seed.

    Returns
    -------
    dict with keys:
        score_a        observed corpus score of system A on full N
        score_b        observed corpus score of system B on full N
        delta          score_a - score_b
        ci_low, ci_high  95 % percentile CI on delta
        p_value        two-sided bootstrap p-value (Koehn 2004)
        n_sentences    N
        n_bootstrap    n_iter
    """
    n = len(refs)
    if not (len(preds_a) == len(preds_b) == n):
        raise ValueError(
            f"Length mismatch: |preds_a|={len(preds_a)}, "
            f"|preds_b|={len(preds_b)}, |refs|={n}"
        )

    if metric == "chrfpp":
        m = CHRF(word_order=2)
        score_fn = lambda p, r: _corpus_chrf(m, p, r)
    elif metric == "bleu":
        m = BLEU()
        score_fn = lambda p, r: _corpus_bleu(m, p, r)
    else:
        raise ValueError(f"unknown metric '{metric}'")

    score_a = score_fn(preds_a, refs)
    score_b = score_fn(preds_b, refs)
    delta_obs = score_a - score_b

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_iter, dtype=np.float64)

    iterator = range(n_iter)
    if _HAS_TQDM:
        iterator = tqdm(iterator, desc=f"  bootstrap[{metric}]", leave=False)

    for i in iterator:
        idx = rng.integers(0, n, size=n)        # with replacement
        pa = [preds_a[j] for j in idx]
        pb = [preds_b[j] for j in idx]
        rr = [refs[j]    for j in idx]
        sa = score_fn(pa, rr)
        sb = score_fn(pb, rr)
        deltas[i] = sa - sb

    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])

    # Two-sided p-value (Koehn 2004): fraction of bootstrap deltas with the
    # opposite sign from the observed delta, doubled (capped at 1.0).
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
    """If systems / refs differ in length, truncate everyone to the min."""
    n = min(len(refs), *(len(v) for v in systems.values()))
    if any(len(v) != n for v in systems.values()) or len(refs) != n:
        systems = {k: v[:n] for k, v in systems.items()}
        refs = refs[:n]
    return systems, refs, n


def _drop_empty_pairs(
    systems: Dict[str, List[str]], refs: List[str]
) -> Tuple[Dict[str, List[str]], List[str]]:
    """Drop sentence indices where any system pred OR the ref is empty.

    This guarantees every (system, ref) pair used in bootstrap is non-empty,
    which keeps BLEU well-defined and keeps chrF++ on a comparable basis.
    """
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
                   help="Which metric(s) to bootstrap.  chrF++ is primary; "
                        "BLEU is added as a secondary cross-check.")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Significance threshold for the printed flags.")
    p.add_argument("--on_mismatch", choices=["skip", "truncate"], default="truncate",
                   help="How to reconcile differing lengths across systems "
                        "and the reference column for one dialect.")
    p.add_argument("--include_pairs", nargs="*", default=None,
                   help="Optional explicit list of source pairs to test, "
                        "format src_a:src_b.  If omitted, ALL pairs are run.")
    p.add_argument("--restrict_targets", nargs="*", default=None,
                   help="Optional list of target dialects to restrict to "
                        "(default: every target found in pred_dir).")
    args = p.parse_args()

    # ---- gold ---------------------------------------------------------- #
    if not args.gold.exists():
        sys.exit(f"ERROR: gold not found: {args.gold}")
    gold_df = load_gold_table(args.gold)
    print(f"[info] gold table       = {args.gold}  shape={gold_df.shape}")
    print(f"[info] gold columns     = {list(gold_df.columns)}")

    # ---- group prediction files by target dialect ---------------------- #
    if not args.pred_dir.is_dir():
        sys.exit(f"ERROR: pred_dir not a folder: {args.pred_dir}")
    files = sorted(args.pred_dir.glob("*.txt"))
    if not files:
        sys.exit(f"ERROR: no .txt files in {args.pred_dir}")

    # by_target[tgt][src] = List[str] of cleaned predictions
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

    # Optional restriction of pairs
    explicit_pairs = None
    if args.include_pairs:
        explicit_pairs = set()
        for s in args.include_pairs:
            try:
                a, b = s.split(":")
            except ValueError:
                sys.exit(f"ERROR: bad --include_pairs entry: {s} "
                         f"(expected 'src_a:src_b')")
            a, b = a.strip().lower(), b.strip().lower()
            explicit_pairs.add(frozenset({a, b}))

    # ---- run bootstrap per (target, src_a, src_b) --------------------- #
    rows: List[dict] = []
    base_rng = np.random.default_rng(args.seed)

    for tgt in sorted(by_target):
        sources = sorted(by_target[tgt])
        if len(sources) < 2:
            print(f"[info] target {tgt}: only {len(sources)} source(s) — "
                  f"no pair to compare, skipping.")
            continue

        refs_full = [str(r).strip() for r in gold_df[tgt].tolist()]
        sys_dict = {s: by_target[tgt][s] for s in sources}

        # Length reconciliation across {refs, every system for this target}
        n_pre = len(refs_full)
        sys_dict, refs, n = _align_to_min(sys_dict, refs_full)
        if any(len(by_target[tgt][s]) != n_pre for s in sources) or len(refs_full) != n_pre:
            if args.on_mismatch == "skip":
                print(f"[warn] target {tgt}: length mismatch (refs={n_pre}, "
                      f"systems={[len(by_target[tgt][s]) for s in sources]}) "
                      f"— skipping (--on_mismatch=skip).", file=sys.stderr)
                continue
            else:
                print(f"[warn] target {tgt}: length mismatch reconciled by "
                      f"truncation to N={n}.", file=sys.stderr)
        # Drop empty rows (any system) so all comparisons share the same n.
        sys_dict, refs = _drop_empty_pairs(sys_dict, refs)
        n = len(refs)
        if n < 5:
            print(f"[warn] target {tgt}: only {n} non-empty rows — skipping.",
                  file=sys.stderr)
            continue

        print(f"\n=== TARGET: {tgt}   sources={sources}   N={n} ===")

        # Pairs
        pair_list = list(combinations(sources, 2))
        if explicit_pairs is not None:
            pair_list = [p for p in pair_list if frozenset(p) in explicit_pairs]
        if not pair_list:
            print(f"  (no pairs after --include_pairs filtering)")
            continue

        # Pre-collect rows for this target so we can apply Bonferroni later.
        per_target_rows: List[dict] = []

        for src_a, src_b in pair_list:
            for metric in args.metrics:
                # Per-pair seed derived from the global one for full reproducibility
                pair_seed = int(base_rng.integers(0, 2**31 - 1))
                print(f"  [{metric}] {src_a} vs {src_b}  (seed={pair_seed})")
                res = paired_bootstrap(
                    preds_a=sys_dict[src_a],
                    preds_b=sys_dict[src_b],
                    refs=refs,
                    metric=metric,
                    n_iter=args.n_bootstrap,
                    seed=pair_seed,
                )
                row = {
                    "target_dialect": tgt,
                    "metric":         metric,
                    "source_a":       src_a,
                    "source_b":       src_b,
                    f"score_a":       round(res["score_a"], 4),
                    f"score_b":       round(res["score_b"], 4),
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

        # ---- Bonferroni correction PER (target, metric) ---------------- #
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

    # ---- summary -------------------------------------------------------- #
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

    # Compact "winners" table for chrF++
    if "chrfpp" in args.metrics:
        chrf_rows = df[df["metric"] == "chrfpp"]
        print("\n" + "-" * 96)
        print("chrF++ winners per pair  (>=  means significant at Bonferroni-alpha)")
        print("-" * 96)
        for _, r in chrf_rows.iterrows():
            d = r["delta"]
            if r.get("significant_bonf", False):
                rel = ">=" if d > 0 else "<="
            elif r.get("significant_alpha", False):
                rel = ">"  if d > 0 else "<"
            else:
                rel = "~"
            print(f"  {r['target_dialect']:>4}: "
                  f"{r['source_a']:>3} ({r['score_a']:5.2f})  "
                  f"{rel}  "
                  f"{r['source_b']:>3} ({r['score_b']:5.2f})    "
                  f"Δ={d:+5.2f}   p={r['p_value']:.4f}   "
                  f"p_bonf={r.get('p_value_bonferroni', float('nan')):.4f}")

    print("\nLegend: '>=' / '<=' Bonferroni-significant, "
          "'>' / '<' significant at alpha (uncorrected), '~' not significant.")


if __name__ == "__main__":
    main()
