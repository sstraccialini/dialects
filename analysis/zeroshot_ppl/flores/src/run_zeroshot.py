"""
End-to-end orchestrator for zero-shot pseudo-perplexity evaluation.

Run from repo root (with env activated):
    python experiments/analysis/zeroshot_ppl/flores/src/run_zeroshot.py

Outputs (in analysis/zeroshot_ppl/flores/{method_outputs,evaluation_results}/):
    ppl_matrix.csv          (n_varieties x n_models matrix of pseudo-PPL)
    ppl_long.csv            (long format: variety, model, ppl, unk_rate)
    nearest_lm_per_variety.csv   (top-3 LM with lowest PPL per variety)

The script saves a per-row CSV after each (variety, model) pair, so a
crash does not lose progress: re-running picks up where it left off.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import pandas as pd

from config import (
    MODELS, VARIETIES, FLORES_DIR,
    outputs_subdir, evaluation_subdir,
    N_SENTENCES, RANDOM_SEED, MASK_RATIO, MAX_LENGTH,
)
from compute_ppl import evaluate_model_on_varieties

# Method outputs vs. evaluation outputs:
#   ppl_long.csv           — raw incremental log (per-row ppl), method-level
#   ppl_matrix.csv         — aggregated wide table, evaluation-level
#   unk_rate_matrix.csv    — aggregated wide table, evaluation-level
#   nearest_lm_per_variety — analysis derived from ppl matrix, evaluation-level
LONG_CSV = outputs_subdir() / "ppl_long.csv"


def load_flores_variety(name: str, n: int) -> list[str]:
    path = FLORES_DIR / f"{name}.txt"
    with open(path, encoding="utf-8") as f:
        sents = [line.strip() for line in f if line.strip()]
    return sents[:n]


def load_existing_long() -> dict:
    """Returns {model_alias: {variety: ppl}} from existing ppl_long.csv if any."""
    out: dict = {}
    if not LONG_CSV.exists():
        return out
    df = pd.read_csv(LONG_CSV)
    for _, row in df.iterrows():
        out.setdefault(row["model"], {})[row["variety"]] = row["ppl"]
    return out


def append_long(variety: str, model: str, ppl: float, unk_rate: float):
    """Append one row to ppl_long.csv (creating header if missing)."""
    new_file = not LONG_CSV.exists()
    with open(LONG_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["variety", "model", "ppl", "unk_rate"])
        w.writerow([variety, model, f"{ppl:.6f}", f"{unk_rate:.6f}"])


def main():
    print("Zero-shot pseudo-PPL on FLORES+")
    print("=" * 60)
    print(f"  varieties     = {VARIETIES}")
    print(f"  models        = {list(MODELS.keys())}")
    print(f"  n_sentences   = {N_SENTENCES}")
    print(f"  mask_ratio    = {MASK_RATIO}")
    print(f"  max_length    = {MAX_LENGTH}")
    print()

    # Load all variety sentences once
    print("Loading variety sentences ...")
    variety_sents = {}
    for var in VARIETIES:
        variety_sents[var] = load_flores_variety(var, N_SENTENCES)
        print(f"  {var:12s}  loaded {len(variety_sents[var])} sents")
    print()

    # Pick up where we left off
    existing = load_existing_long()
    if existing:
        print(f"Resume: found existing partial results for {list(existing.keys())}")
        print()

    # Loop over models, evaluate each on all varieties
    matrix: dict[str, dict[str, float]] = {}
    unk_matrix: dict[str, dict[str, float]] = {}

    for model_alias, model_path in MODELS.items():
        print(f"--- Evaluating model: {model_alias} ({model_path}) ---")
        skip = existing.get(model_alias, {})

        # If every variety is already done for this model, skip loading
        # the model entirely. This avoids wasting a GPU/disk re-download
        # just to skip every iteration.
        if len(skip) >= len(VARIETIES):
            print(f"  all {len(VARIETIES)} varieties already done, skipping model load")
            matrix[model_alias] = skip
            unk_matrix[model_alias] = {v: 0.0 for v in VARIETIES}
            print()
            continue

        if skip:
            print(f"  resume: skipping {len(skip)} already-done varieties")

        def cb(var, ppl, unk, _alias=model_alias):
            append_long(var, _alias, ppl, unk)

        ppl_per_variety, unk_per_variety = evaluate_model_on_varieties(
            model_path, variety_sents,
            mask_ratio=MASK_RATIO,
            max_length=MAX_LENGTH,
            seed=RANDOM_SEED,
            skip_existing=skip,
            save_callback=cb,
        )
        matrix[model_alias] = ppl_per_variety
        unk_matrix[model_alias] = unk_per_variety
        for var in VARIETIES:
            ppl = ppl_per_variety.get(var, float("nan"))
            unk = unk_per_variety.get(var, 0.0)
            print(f"  {var:12s}  ppl={ppl:8.3f}  unk_rate={unk:.3f}")
        print()

    # ---- Save ppl_matrix.csv (rows=varieties, cols=models) ----
    rows = []
    for var in VARIETIES:
        row = {"variety": var}
        for model_alias in MODELS:
            row[model_alias] = matrix[model_alias].get(var, float("nan"))
        rows.append(row)
    df = pd.DataFrame(rows).set_index("variety")
    out_csv = evaluation_subdir() / "ppl_matrix.csv"
    df.to_csv(out_csv)
    print(f"Saved: {out_csv}")

    # ---- Save unk_rate_matrix.csv ----
    rows = []
    for var in VARIETIES:
        row = {"variety": var}
        for model_alias in MODELS:
            row[model_alias] = unk_matrix[model_alias].get(var, float("nan"))
        rows.append(row)
    unk_df = pd.DataFrame(rows).set_index("variety")
    unk_csv = evaluation_subdir() / "unk_rate_matrix.csv"
    unk_df.to_csv(unk_csv)
    print(f"Saved: {unk_csv}")

    # ---- Build long_df from ppl_long.csv (already saved incrementally) ----
    long_df = pd.read_csv(LONG_CSV)

    # ---- Save nearest_lm_per_variety.csv (treating UNK-heavy as nan) ----
    # Filter rows where unk_rate > 0.5 (mostly UNK, unreliable PPL)
    valid = long_df[long_df["unk_rate"] < 0.5].copy()
    nearest_rows = []
    for var in VARIETIES:
        sub = valid[valid["variety"] == var].sort_values("ppl")
        if sub.empty:
            nearest_rows.append({"variety": var, "nn_1_lm": "(all UNK-heavy)", "nn_1_ppl": ""})
            continue
        top1 = sub.iloc[0]
        top2 = sub.iloc[1] if len(sub) > 1 else None
        top3 = sub.iloc[2] if len(sub) > 2 else None
        row = {
            "variety": var,
            "nn_1_lm": top1["model"], "nn_1_ppl": top1["ppl"],
            "nn_2_lm": top2["model"] if top2 is not None else "",
            "nn_2_ppl": top2["ppl"] if top2 is not None else "",
            "nn_3_lm": top3["model"] if top3 is not None else "",
            "nn_3_ppl": top3["ppl"] if top3 is not None else "",
        }
        nearest_rows.append(row)
    nn_df = pd.DataFrame(nearest_rows)
    nn_csv = evaluation_subdir() / "nearest_lm_per_variety.csv"
    nn_df.to_csv(nn_csv, index=False)
    print(f"Saved: {nn_csv}")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
