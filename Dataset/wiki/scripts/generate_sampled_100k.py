"""
Generate 100k-sample Wiki CSVs from the *_complete.csv full corpora.

For every variety in the FINAL registry (17 codes), and for each text variant
(`normalized`, `not_normalized`):

  1. Rename `<group>/<code>.csv` -> `<group>/<code>_complete.csv` (idempotent —
     skipped if already done).
  2. Read `<code>_complete.csv`, sample 100k rows (seed=42), or take all if
     fewer than 100k available.
  3. Write the result back to `<group>/<code>.csv` (the name every method
     points to via `WIKI_VARIETY_DIR` / `WIKI_VARIETY_DIR_NATIVE`).

The `_complete.csv` files stay gitignored (heavy ~7 GB total). The new 100k
samples (~10 MB each, ~420 MB total across both variants) are committed to
git so the whole team uses the SAME training subset, deterministically.

Run from repo root:
    python Dataset/wiki/scripts/generate_sampled_100k.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis._shared.varieties import (
    VARIETY_CODES, WIKI_VARIETY_DIR, WIKI_VARIETY_DIR_NATIVE,
)


SAMPLE_SIZE = 100_000
RANDOM_STATE = 42


def _process(code: str, variant: str, dir_map: dict) -> dict:
    base = dir_map.get(code)
    if base is None:
        return {"code": code, "variant": variant, "status": "no_dir_map"}

    sampled = base / f"{code}.csv"
    complete = base / f"{code}_complete.csv"

    if not sampled.exists() and not complete.exists():
        return {"code": code, "variant": variant, "status": "missing_source"}

    # Step 1 — promote the existing full file to *_complete.csv (idempotent).
    if not complete.exists():
        shutil.move(str(sampled), str(complete))
        print(f"  [{variant:<14} {code:>4}] renamed -> {complete.name}")

    # Step 2 — read complete, sample (or take all).
    df = pd.read_csv(complete)
    n_avail = len(df)
    if n_avail <= SAMPLE_SIZE:
        sample = df
        action = f"all ({n_avail})"
    else:
        sample = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
        action = f"sampled {SAMPLE_SIZE} from {n_avail}"

    # Step 3 — write back to <code>.csv.
    sample.to_csv(sampled, index=False)
    size_mb = sampled.stat().st_size / 1024 / 1024
    print(f"  [{variant:<14} {code:>4}] {action:<28} -> {sampled.name}  ({size_mb:.1f} MB)")

    return {
        "code": code, "variant": variant,
        "n_complete": n_avail, "n_sampled": len(sample),
        "size_mb": round(size_mb, 2), "status": "ok",
    }


def main():
    print(f"Sampling Wiki to {SAMPLE_SIZE} sentences/variety (seed={RANDOM_STATE}).\n")

    rows = []
    for variant, dir_map in [
        ("normalized",     WIKI_VARIETY_DIR),
        ("not_normalized", WIKI_VARIETY_DIR_NATIVE),
    ]:
        print(f"=== {variant} ===")
        for code in VARIETY_CODES:
            rows.append(_process(code, variant, dir_map))
        print()

    summary = pd.DataFrame(rows)
    print("\nSummary:")
    print(summary.to_string(index=False))
    total_mb = summary["size_mb"].fillna(0).sum()
    print(f"\nTotal sampled-CSV size: {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
