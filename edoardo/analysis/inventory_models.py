"""
Inventory of trained models / embeddings / evaluation outputs in the repo.

Walks ``analysis/<method>/[experiments|old_experiments]/...`` and reports:
  - which (method, experiment, dataset, variant) have ``distances.csv``
  - which have ``variety_vectors.npz``  (needed for CKA / Procrustes)
  - which have ``run_meta.json``
  - which have a saved model checkpoint under ``method_outputs``

Run from the repo root:

    python -m edoardo.analysis.inventory_models                   # printed summary
    python -m edoardo.analysis.inventory_models --csv inv.csv      # also dump CSV
    python -m edoardo.analysis.inventory_models --include-old      # include old_experiments/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = REPO_ROOT / "analysis"

CHECKPOINT_FILES = {
    "pytorch_model.bin",
    "model.safetensors",
    "config.json",          # transformers checkpoint marker
    "model.npz",            # gensim/word2vec
    "model.bin",            # fasttext
}


def _iter_distances_csv(include_old: bool) -> Iterator[Path]:
    """Yield every ``distances.csv`` under analysis/."""
    for method_dir in sorted(p for p in ANALYSIS_DIR.iterdir() if p.is_dir()):
        if method_dir.name.startswith("_"):
            continue
        exp_roots = [method_dir / "experiments"]
        if include_old:
            exp_roots.append(method_dir / "old_experiments")
        for er in exp_roots:
            if not er.exists():
                continue
            yield from er.rglob("distances.csv")


def _parse_eval_path(p: Path) -> dict:
    """
    From a ``distances.csv`` path under analysis/, extract:
        method, root_kind ('experiments' / 'old_experiments'),
        experiment, variant_path (everything after evaluation_results/).
    """
    rel = p.relative_to(ANALYSIS_DIR)
    parts = rel.parts            # ('method', 'experiments', '<exp>', 'evaluation_results', ...)
    method = parts[0]
    root_kind = parts[1] if parts[1] in ("experiments", "old_experiments") else "?"
    if root_kind == "?":
        return {
            "method": method, "root_kind": "?", "experiment": "?",
            "variant_path": "/".join(parts[1:-1]),
        }
    experiment = parts[2]
    try:
        er_idx = parts.index("evaluation_results")
    except ValueError:
        er_idx = -1
    variant_path = "/".join(parts[er_idx + 1 : -1]) if er_idx >= 0 else ""
    return {
        "method": method,
        "root_kind": root_kind,
        "experiment": experiment,
        "variant_path": variant_path,
    }


def _matching_method_outputs(distances_csv: Path) -> Path:
    """Return the method_outputs/ directory matching this evaluation_results/ path."""
    parts = list(distances_csv.parts)
    try:
        er_idx = parts.index("evaluation_results")
    except ValueError:
        return distances_csv.parent
    parts[er_idx] = "method_outputs"
    return Path(*parts).parent


def _checkpoint_size_bytes(method_outputs_dir: Path) -> int:
    """Heuristic: size in bytes of any model checkpoint we recognise."""
    if not method_outputs_dir.exists():
        return 0
    total = 0
    for p in method_outputs_dir.rglob("*"):
        if p.is_file() and (p.name in CHECKPOINT_FILES or p.suffix == ".safetensors"):
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _read_run_meta(method_outputs_dir: Path) -> dict | None:
    rm = method_outputs_dir / "run_meta.json"
    if not rm.exists():
        # fallback: look one level up
        rm = method_outputs_dir.parent / "run_meta.json"
    if not rm.exists():
        return None
    try:
        with rm.open() as fh:
            return json.load(fh)
    except Exception:
        return None


def build_inventory(include_old: bool = False) -> pd.DataFrame:
    rows = []
    for dist_csv in _iter_distances_csv(include_old):
        info = _parse_eval_path(dist_csv)
        mo = _matching_method_outputs(dist_csv)
        vv = mo / "variety_vectors.npz"
        rm = _read_run_meta(mo)
        ckpt_bytes = _checkpoint_size_bytes(mo)
        try:
            n = pd.read_csv(dist_csv, index_col=0).shape[0]
        except Exception:
            n = -1
        rows.append({
            "method": info["method"],
            "root_kind": info["root_kind"],
            "experiment": info["experiment"],
            "variant_path": info["variant_path"],
            "n_varieties": n,
            "distances_csv": str(dist_csv.relative_to(REPO_ROOT)),
            "variety_vectors_npz": str(vv.relative_to(REPO_ROOT)) if vv.exists() else "",
            "method_outputs_dir": str(mo.relative_to(REPO_ROOT)) if mo.exists() else "",
            "checkpoint_size_mb": round(ckpt_bytes / (1024 * 1024), 1),
            "git_commit": (rm or {}).get("git_commit", "")[:8] if rm else "",
            "timestamp": (rm or {}).get("timestamp", "") if rm else "",
        })
    return pd.DataFrame(rows).sort_values(["method", "root_kind", "experiment", "variant_path"])


def _print_summary(df: pd.DataFrame) -> None:
    print(f"\nInventory: {len(df)} (method, experiment, variant) triples found\n")
    if df.empty:
        return

    # Per-method roll-up
    summary = (
        df.groupby(["method", "root_kind"], dropna=False)
          .agg(
              n_results=("variant_path", "size"),
              n_with_vectors=("variety_vectors_npz", lambda s: (s != "").sum()),
              n_with_checkpoint=("checkpoint_size_mb", lambda s: (s > 0).sum()),
              total_ckpt_mb=("checkpoint_size_mb", "sum"),
          )
          .reset_index()
    )
    print("Per-method summary:")
    print(summary.to_string(index=False))
    print()

    # Detailed listing
    cols = ["method", "root_kind", "experiment", "variant_path",
            "n_varieties", "checkpoint_size_mb"]
    has_vec = df["variety_vectors_npz"].astype(bool)
    print("All evaluation outputs (* = has variety_vectors.npz):")
    detail = df[cols].copy()
    detail.insert(0, "vec", has_vec.map({True: "*", False: " "}))
    print(detail.to_string(index=False))
    print()

    # Pair detection: same (method, experiment_family) with multiple experiments
    # → flags candidate (baseline, fine-tuned) groupings.
    print("Candidate baseline ↔ fine-tuned groupings (same method, ≥2 experiments):")
    grp = df.groupby("method")["experiment"].nunique()
    for method, n_exp in grp.items():
        if n_exp >= 2:
            exps = sorted(df.loc[df["method"] == method, "experiment"].unique())
            print(f"  {method:<22} : {exps}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=None,
                    help="If given, also dump the full inventory as CSV.")
    ap.add_argument("--include-old", action="store_true",
                    help="Also walk old_experiments/ folders.")
    args = ap.parse_args(argv)

    df = build_inventory(include_old=args.include_old)
    _print_summary(df)
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nFull inventory written to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
