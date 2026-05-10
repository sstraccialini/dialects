#!/usr/bin/env python3
"""
cross_llm_bootstrap.py
======================
Head-to-head paired bootstrap significance test between TWO LLM systems
(e.g. ChatGPT vs Gemini) on the SAME translation files.

For every file present in BOTH prediction folders with the same filename
(e.g. ita-to-scn.txt), the two LLMs are compared sentence-by-sentence
against the same gold reference column. Pairing is at sentence-index
level: each bootstrap iteration draws N indices with replacement, and
both LLMs are re-scored on the SAME indices against the SAME refs.

Methodology (Koehn 2004), identical to bootstrap_significance.py but
the "two systems" being compared are now {LLM_A on file F} vs
{LLM_B on file F}, instead of {src_a -> dialect} vs {src_b -> dialect}.

Run:
    python cross_llm_bootstrap.py \
        --pred_dir_a  data-chatgpt \
        --pred_dir_b  data-gemini \
        --label_a     chatgpt \
        --label_b     gemini \
        --gold        data/gold_table.csv \
        --out         cross_llm_chatgpt_vs_gemini.csv \
        --n_bootstrap 1000 \
        --n_jobs      -1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from compute_metrics import (   # noqa: E402
    parse_filename,
    load_predictions,
    load_gold_table,
)
from llm_translations.bootstrap_significance import paired_bootstrap, _bonferroni  # noqa: E402

try:
    from sacrebleu.metrics import BLEU, CHRF  # noqa: F401
except ImportError:
    sys.exit("ERROR: sacrebleu not installed.  pip install sacrebleu")


# --------------------------------------------------------------------------- #
def _index_pred_dir(pred_dir: Path
                    ) -> Dict[Tuple[str, str], Tuple[Path, List[str]]]:
    """Return { (source, target) -> (filepath, predictions) }."""
    out: Dict[Tuple[str, str], Tuple[Path, List[str]]] = {}
    if not pred_dir.is_dir():
        sys.exit(f"ERROR: pred_dir not a folder: {pred_dir}")
    for fpath in sorted(pred_dir.glob("*.txt")):
        parsed = parse_filename(fpath)
        if parsed is None:
            print(f"[warn] {pred_dir.name}: bad filename {fpath.name}",
                  file=sys.stderr)
            continue
        try:
            preds = load_predictions(fpath)
        except Exception as e:
            print(f"[warn] {pred_dir.name}: cannot read {fpath.name} ({e})",
                  file=sys.stderr)
            continue
        out[parsed] = (fpath, preds)
    return out


def _reconcile(
    preds_a: List[str], preds_b: List[str], refs: List[str],
    on_mismatch: str,
) -> Tuple[List[str], List[str], List[str]]:
    """Return (preds_a, preds_b, refs) after aligning lengths and dropping
    rows where any of the three is empty. None if nothing remains."""
    n_a, n_b, n_r = len(preds_a), len(preds_b), len(refs)
    if not (n_a == n_b == n_r):
        if on_mismatch == "skip":
            raise ValueError(f"length mismatch: A={n_a} B={n_b} R={n_r}")
        n = min(n_a, n_b, n_r)
        preds_a, preds_b, refs = preds_a[:n], preds_b[:n], refs[:n]
    keep = [i for i in range(len(refs))
            if refs[i] and preds_a[i] and preds_b[i]]
    return ([preds_a[i] for i in keep],
            [preds_b[i] for i in keep],
            [refs[i]    for i in keep])


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="Paired bootstrap chrF++ comparison between two LLMs "
                    "(e.g. ChatGPT vs Gemini) on identically-named files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pred_dir_a", required=True, type=Path,
                   help="Folder for system A (e.g. data-chatgpt).")
    p.add_argument("--pred_dir_b", required=True, type=Path,
                   help="Folder for system B (e.g. data-gemini).")
    p.add_argument("--label_a", default="A",
                   help="Short label for system A in the output (e.g. chatgpt).")
    p.add_argument("--label_b", default="B",
                   help="Short label for system B in the output (e.g. gemini).")
    p.add_argument("--gold", required=True, type=Path,
                   help="Reference table (CSV / TSV / XLSX).")
    p.add_argument("--out", default="cross_llm_bootstrap.csv", type=Path,
                   help="Output CSV.")
    p.add_argument("--n_bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--metrics", nargs="+", choices=["chrfpp", "bleu"],
                   default=["chrfpp"])
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--on_mismatch", choices=["skip", "truncate"],
                   default="truncate")
    p.add_argument("--n_jobs", type=int, default=1,
                   help="CPU cores for parallel bootstrap. 1=seq, -1=all.")
    p.add_argument("--restrict_targets", nargs="*", default=None,
                   help="Optional list of target dialects to restrict to.")
    p.add_argument("--restrict_sources", nargs="*", default=None,
                   help="Optional list of source languages to restrict to.")
    args = p.parse_args()

    # ---- gold ---------------------------------------------------------- #
    if not args.gold.exists():
        sys.exit(f"ERROR: gold not found: {args.gold}")
    gold_df = load_gold_table(args.gold)
    print(f"[info] gold table       = {args.gold}  shape={gold_df.shape}")
    print(f"[info] system A         = {args.label_a}  ({args.pred_dir_a})")
    print(f"[info] system B         = {args.label_b}  ({args.pred_dir_b})")
    print(f"[info] n_bootstrap      = {args.n_bootstrap}")
    print(f"[info] n_jobs           = {args.n_jobs}")

    # ---- index both pred dirs ------------------------------------------ #
    idx_a = _index_pred_dir(args.pred_dir_a)
    idx_b = _index_pred_dir(args.pred_dir_b)
    print(f"[info] {args.label_a}: {len(idx_a)} files")
    print(f"[info] {args.label_b}: {len(idx_b)} files")

    # Files present in BOTH
    shared = sorted(set(idx_a.keys()) & set(idx_b.keys()))
    only_a = sorted(set(idx_a.keys()) - set(idx_b.keys()))
    only_b = sorted(set(idx_b.keys()) - set(idx_a.keys()))
    if only_a:
        print(f"[warn] only in {args.label_a}: {only_a}", file=sys.stderr)
    if only_b:
        print(f"[warn] only in {args.label_b}: {only_b}", file=sys.stderr)
    if not shared:
        sys.exit("ERROR: no (source,target) files shared between the two folders.")

    # Optional restrictions
    if args.restrict_targets:
        shared = [(s, t) for (s, t) in shared if t in args.restrict_targets]
    if args.restrict_sources:
        shared = [(s, t) for (s, t) in shared if s in args.restrict_sources]
    if not shared:
        sys.exit("ERROR: no shared files left after restrictions.")

    print(f"[info] shared comparisons = {len(shared)}: {shared}")

    # ---- run bootstrap per shared file --------------------------------- #
    rows: List[dict] = []
    base_rng = np.random.default_rng(args.seed)

    for src, tgt in shared:
        if tgt not in gold_df.columns:
            print(f"[warn] no gold column '{tgt}' for {src}-to-{tgt} — skipping.",
                  file=sys.stderr)
            continue

        path_a, preds_a = idx_a[(src, tgt)]
        path_b, preds_b = idx_b[(src, tgt)]
        refs_full = [str(r).strip() for r in gold_df[tgt].tolist()]

        try:
            preds_a, preds_b, refs = _reconcile(
                preds_a, preds_b, refs_full, args.on_mismatch
            )
        except ValueError as e:
            print(f"[warn] {src}-to-{tgt}: {e} — skipping.", file=sys.stderr)
            continue

        n = len(refs)
        if n < 5:
            print(f"[warn] {src}-to-{tgt}: only {n} usable rows — skipping.",
                  file=sys.stderr)
            continue

        print(f"\n=== {src} → {tgt}   N={n} ===")

        for metric in args.metrics:
            pair_seed = int(base_rng.integers(0, 2**31 - 1))
            print(f"  [{metric}] {args.label_a} vs {args.label_b}  (seed={pair_seed})")
            res = paired_bootstrap(
                preds_a=preds_a,
                preds_b=preds_b,
                refs=refs,
                metric=metric,
                n_iter=args.n_bootstrap,
                seed=pair_seed,
                n_jobs=args.n_jobs,
            )
            rows.append({
                "source_language":  src,
                "target_dialect":   tgt,
                "metric":           metric,
                "system_a":         args.label_a,
                "system_b":         args.label_b,
                "score_a":          round(res["score_a"], 4),
                "score_b":          round(res["score_b"], 4),
                "delta":            round(res["delta"], 4),
                "ci_low":           round(res["ci_low"], 4),
                "ci_high":          round(res["ci_high"], 4),
                "p_value":          round(res["p_value"], 6),
                "n_sentences":      res["n_sentences"],
                "n_bootstrap":      res["n_bootstrap"],
                "seed":             pair_seed,
            })
            ci_str = f"[{res['ci_low']:+.2f}, {res['ci_high']:+.2f}]"
            star = "*" if res["p_value"] < args.alpha else " "
            print(f"     {args.label_a}={res['score_a']:6.2f}  "
                  f"{args.label_b}={res['score_b']:6.2f}  "
                  f"Δ={res['delta']:+6.2f}  CI95={ci_str}  "
                  f"p={res['p_value']:.4f} {star}")

    if not rows:
        sys.exit("\nERROR: no comparisons were run.")

    # ---- Bonferroni correction PER metric (across all files) ----------- #
    df = pd.DataFrame(rows)
    for metric in args.metrics:
        sub_idx = df.index[df["metric"] == metric].tolist()
        if not sub_idx:
            continue
        ps = df.loc[sub_idx, "p_value"].tolist()
        ps_bonf = _bonferroni(ps)
        for i, pb in zip(sub_idx, ps_bonf):
            df.loc[i, "p_value_bonferroni"] = round(pb, 6)
        df.loc[sub_idx, "significant_alpha"] = (
            df.loc[sub_idx, "p_value"] < args.alpha
        )
        df.loc[sub_idx, "significant_bonf"] = (
            df.loc[sub_idx, "p_value_bonferroni"] < args.alpha
        )

    df = df.sort_values(["target_dialect", "metric", "source_language"]
                        ).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"\n[ok] wrote {len(df)} rows -> {args.out}")

    # ---- summary -------------------------------------------------------- #
    print("\n" + "=" * 96)
    print(f"CROSS-LLM SUMMARY: {args.label_a} vs {args.label_b}  "
          f"(alpha={args.alpha}, B={args.n_bootstrap})")
    print("=" * 96)
    cols = ["target_dialect", "source_language", "metric",
            "score_a", "score_b", "delta", "ci_low", "ci_high",
            "p_value", "p_value_bonferroni",
            "significant_alpha", "significant_bonf"]
    cols = [c for c in cols if c in df.columns]
    with pd.option_context("display.max_rows", None,
                           "display.max_columns", None,
                           "display.width", 220):
        print(df[cols].to_string(index=False))

    # Compact win count
    print("\n" + "-" * 96)
    print(f"Win counts (chrF++):")
    print("-" * 96)
    chrf = df[df["metric"] == "chrfpp"] if "chrfpp" in args.metrics else df.head(0)
    if not chrf.empty:
        wins_a = ((chrf["delta"] > 0) & (chrf.get("significant_bonf",
                  chrf.get("significant_alpha", False)))).sum()
        wins_b = ((chrf["delta"] < 0) & (chrf.get("significant_bonf",
                  chrf.get("significant_alpha", False)))).sum()
        ties   = len(chrf) - wins_a - wins_b
        print(f"  {args.label_a:<10} significant wins : {wins_a}")
        print(f"  {args.label_b:<10} significant wins : {wins_b}")
        print(f"  ties / not significant         : {ties}")


if __name__ == "__main__":
    main()
