"""
End-to-end orchestrator for the Subword / FastText approach on FLORES+.

Two sub-pipelines (variant subdirectories):
    fasttext   shared FastText + mean-pooled variety vectors
    bpe        SentencePiece BPE + TF-IDF on BPE pieces

Each variant produces its own evaluation artefacts under
``evaluation_results/<variant>/``; method-specific outputs (vectors,
top BPE pieces, models, run stats) live under ``method_outputs/``.

Launch (from repo root, with venv active):
    python analysis/fasttext/flores/src/run_approach.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
for p in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import (
    VARIETY_CODES, VARIETY_GROUP, VARIETY_NAMES,
    GROUP_NAMES, GROUP_COLORS,
    outputs_subdir, evaluation_subdir,
)
from data_loader import load_all_varieties
from embed_fasttext import run_fasttext_pipeline
from embed_bpe import run_bpe_pipeline, top_pieces_per_variety

from evaluation.evaluation import run_evaluation


ROMANCE_FAMILIES = {"italo_romance", "italian", "romance"}


def _save_top_features(top_dict: dict, variant: str) -> Path:
    rows = []
    for code, feats in top_dict.items():
        for rank, (f, w) in enumerate(feats, start=1):
            rows.append({"code": code, "rank": rank, "feature": f, "weight": w})
    out = outputs_subdir(variant) / "top_features.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def _evaluate(variant: str, X, codes) -> dict:
    return run_evaluation(
        variety_vectors=X.toarray() if hasattr(X, "toarray") else X,
        variety_codes=codes,
        out_dir=evaluation_subdir(variant),
        method_label=f"fasttext ({variant})",
        family_groups=VARIETY_GROUP,
        family_colors=GROUP_COLORS,
        family_display_names=GROUP_NAMES,
        display_names=VARIETY_NAMES,
        romance_families=ROMANCE_FAMILIES,
    )


def run_fasttext(data) -> dict:
    X, codes, _ = run_fasttext_pipeline(data)
    rep = _evaluate("fasttext", X, codes)
    return {
        "variant": "fasttext", "shape": tuple(X.shape),
        "silhouette_family": rep["silhouette_family"],
        "silhouette_romance_vs_rest": rep["silhouette_romance_vs_rest"],
    }


def run_bpe(data) -> dict:
    X, vec, codes, _sp = run_bpe_pipeline(data)
    _save_top_features(top_pieces_per_variety(X, vec, codes, k=30), "bpe")
    rep = _evaluate("bpe", X, codes)
    return {
        "variant": "bpe", "shape": tuple(X.shape),
        "silhouette_family": rep["silhouette_family"],
        "silhouette_romance_vs_rest": rep["silhouette_romance_vs_rest"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Subword / FastText approach orchestrator (FLORES+)"
    )
    parser.add_argument("--pipeline", choices=["fasttext", "bpe", "both"], default="both")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    args = parser.parse_args()

    from config import SAMPLE_SIZE, RANDOM_STATE
    sample_size = args.sample_size or SAMPLE_SIZE
    random_state = args.random_state if args.random_state is not None else RANDOM_STATE

    print("Subword / FastText approach on FLORES+")
    print("=" * 60)
    print(f"  sample_size   = {sample_size}")
    print(f"  random_state  = {random_state}")
    print(f"  pipeline      = {args.pipeline}")
    print(f"  varieties     = {VARIETY_CODES}")
    print()

    print("Loading + sampling...")
    data, stats = load_all_varieties(
        sample_size=sample_size,
        random_state=random_state,
    )
    stats["sample_size_param"] = sample_size
    stats["random_state"] = random_state
    stats.to_csv(outputs_subdir() / "run_stats.csv", index=False)

    reports = []
    if args.pipeline in ("fasttext", "both"):
        reports.append(run_fasttext(data))
    if args.pipeline in ("bpe", "both"):
        reports.append(run_bpe(data))

    print("\n" + "=" * 60)
    print("Done. Summary:")
    for r in reports:
        sf = r["silhouette_family"]
        sr = r["silhouette_romance_vs_rest"]
        print(f"  {r['variant']:>8}: shape={r['shape']}  "
              f"sil_family={sf:+.4f}  sil_romance={sr:+.4f}")
    print(f"\nMethod outputs:       {outputs_subdir()}")
    print(f"Evaluation artefacts: {evaluation_subdir()}")


if __name__ == "__main__":
    main()
