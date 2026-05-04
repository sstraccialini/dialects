"""
Task 3 orchestrator: XLM-R fine-tuning experiments on FLORES+.

Runs four conditions in sequence and writes per-condition evaluation artefacts
plus a cross-condition summary JSON.

    baseline      — xlm-roberta-base, no adaptation
    mlm_wiki      — continued MLM on Wikipedia dialect text
    tlm_oldi      — TLM on OLDI parallel pairs (Italian ↔ dialect)
    mlm_then_tlm  — MLM on wiki, then TLM on OLDI (sequential)

Launch (from repo root, with venv active):
    python analysis/xlmr_finetuned/flores/src/run_pipeline.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
for p in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import (
    BASE_MODEL,
    BATCH_SIZE,
    CONDITIONS,
    GROUP_COLORS,
    GROUP_NAMES,
    MAX_LENGTH,
    MAX_WIKI_SAMPLES,
    MLM_EPOCHS,
    TLM_EPOCHS,
    VARIETY_CODES,
    VARIETY_GROUP,
    VARIETY_NAMES,
    evaluation_subdir,
    model_dir,
    outputs_subdir,
)
from data_loader import (
    iter_labeled_sentences,
    load_all_flores,
    load_all_oldi_pairs,
    load_wiki_texts,
)
from embedder import MultilingualEmbedder
from trainer import run_mlm_training, run_tlm_training

from evaluation.evaluation import run_evaluation

ROMANCE_FAMILIES = {"italo_romance", "italian", "romance"}


def _l2_normalise(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(norm == 0, 1.0, norm)


def aggregate_variety_vectors(
    sent_vecs: np.ndarray,
    sent_codes: list,
    codes: list = VARIETY_CODES,
):
    arr_codes = np.asarray(sent_codes)
    codes_ordered, rows = [], []
    for slug in codes:
        mask = arr_codes == slug
        if mask.sum() == 0:
            continue
        rows.append(sent_vecs[mask].mean(axis=0))
        codes_ordered.append(slug)
    X = _l2_normalise(np.vstack(rows).astype(np.float32))
    return X, codes_ordered


def embed_and_evaluate(
    model_path: str,
    flores_data: dict,
    condition: str,
    device,
) -> tuple:
    t0 = time.time()
    embedder = MultilingualEmbedder(model_name=model_path, device=device, max_length=MAX_LENGTH)

    sents, sent_codes = iter_labeled_sentences(flores_data)
    sent_vecs = embedder.encode(sents, batch_size=BATCH_SIZE)
    X, codes = aggregate_variety_vectors(sent_vecs, sent_codes)
    del embedder  # free GPU memory before next condition

    out = outputs_subdir(condition)
    np.savez_compressed(out / "variety_vectors.npz", matrix=X, labels=np.asarray(codes))
    pd.DataFrame(X, index=codes).to_csv(out / "variety_vectors.csv", float_format="%.6f")

    report = run_evaluation(
        variety_vectors=X,
        variety_codes=codes,
        out_dir=evaluation_subdir(condition),
        method_label=f"XLM-R ({condition})",
        family_groups=VARIETY_GROUP,
        family_colors=GROUP_COLORS,
        family_display_names=GROUP_NAMES,
        display_names=VARIETY_NAMES,
        romance_families=ROMANCE_FAMILIES,
    )
    return report, time.time() - t0


def main():
    parser = argparse.ArgumentParser(description="Task 3: XLM-R fine-tuning experiments")
    parser.add_argument(
        "--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS,
        help="Conditions to run (default: all four)",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="If a fine-tuned model already exists in models/, skip retraining",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mlm-epochs", type=int, default=None)
    parser.add_argument("--tlm-epochs", type=int, default=None)
    parser.add_argument("--wiki-samples", type=int, default=None)
    args = parser.parse_args()

    mlm_epochs = args.mlm_epochs or MLM_EPOCHS
    tlm_epochs = args.tlm_epochs or TLM_EPOCHS
    wiki_samples = args.wiki_samples or MAX_WIKI_SAMPLES

    print("Task 3 — XLM-R Fine-tuning Experiments")
    print("=" * 60)
    print(f"  conditions   = {args.conditions}")
    print(f"  skip_train   = {args.skip_train}")
    print(f"  mlm_epochs   = {mlm_epochs}")
    print(f"  tlm_epochs   = {tlm_epochs}")
    print(f"  wiki_samples = {wiki_samples:,}/dialect")
    print()

    # Load FLORES+ evaluation data once — never used for training
    print("Loading FLORES+ evaluation data...")
    flores_data, flores_stats = load_all_flores()
    flores_stats.to_csv(outputs_subdir() / "flores_stats.csv", index=False)
    print()

    # Lazily load training data only when needed
    wiki_texts = None
    oldi_pairs = None
    mlm_model_path: str | None = None  # reused for mlm_then_tlm

    results: dict = {}

    for condition in args.conditions:
        print(f"\n{'='*60}")
        print(f"  Condition: {condition}")
        print(f"{'='*60}")

        if condition == "baseline":
            model_path = BASE_MODEL

        elif condition == "mlm_wiki":
            mdir = model_dir("mlm_wiki")
            trained = (mdir / "config.json").exists()
            if args.skip_train and trained:
                print(f"  Reloading existing model: {mdir}")
                model_path = str(mdir)
            else:
                if wiki_texts is None:
                    print("  Loading wiki texts...")
                    wiki_texts = load_wiki_texts(max_per_lang=wiki_samples)
                    print(f"  Total wiki texts: {len(wiki_texts):,}")
                run_mlm_training(BASE_MODEL, wiki_texts, mdir, epochs=mlm_epochs)
                model_path = str(mdir)
            mlm_model_path = model_path

        elif condition == "tlm_oldi":
            mdir = model_dir("tlm_oldi")
            trained = (mdir / "config.json").exists()
            if args.skip_train and trained:
                print(f"  Reloading existing model: {mdir}")
                model_path = str(mdir)
            else:
                if oldi_pairs is None:
                    print("  Loading OLDI pairs...")
                    oldi_pairs = load_all_oldi_pairs()
                    print(f"  Total pairs: {len(oldi_pairs):,}")
                run_tlm_training(BASE_MODEL, oldi_pairs, mdir, epochs=tlm_epochs)
                model_path = str(mdir)

        elif condition == "mlm_then_tlm":
            mdir = model_dir("mlm_then_tlm")
            trained = (mdir / "config.json").exists()
            if args.skip_train and trained:
                print(f"  Reloading existing model: {mdir}")
                model_path = str(mdir)
            else:
                # Ensure the MLM model exists (run it first if not yet done)
                if mlm_model_path is None:
                    mlm_dir = model_dir("mlm_wiki")
                    if wiki_texts is None:
                        print("  Loading wiki texts...")
                        wiki_texts = load_wiki_texts(max_per_lang=wiki_samples)
                    run_mlm_training(BASE_MODEL, wiki_texts, mlm_dir, epochs=mlm_epochs)
                    mlm_model_path = str(mlm_dir)
                if oldi_pairs is None:
                    print("  Loading OLDI pairs...")
                    oldi_pairs = load_all_oldi_pairs()
                run_tlm_training(mlm_model_path, oldi_pairs, mdir, epochs=tlm_epochs)
                model_path = str(mdir)

        report, elapsed = embed_and_evaluate(model_path, flores_data, condition, args.device)
        results[condition] = {
            "silhouette_family": report["silhouette_family"],
            "silhouette_romance_vs_rest": report["silhouette_romance_vs_rest"],
            "elapsed_s": round(elapsed),
        }
        print(
            f"  → sil_family={report['silhouette_family']:+.4f}  "
            f"sil_romance={report['silhouette_romance_vs_rest']:+.4f}  "
            f"time={elapsed:.0f}s"
        )

    # Cross-condition summary
    print("\n" + "=" * 60)
    print("Summary")
    print(f"  {'Condition':<18} {'sil_family':>12} {'sil_romance':>12} {'time(s)':>9}")
    print("  " + "-" * 55)
    for cond, r in results.items():
        print(
            f"  {cond:<18} {r['silhouette_family']:>+12.4f} "
            f"{r['silhouette_romance_vs_rest']:>+12.4f} "
            f"{r['elapsed_s']:>9d}"
        )

    summary_path = outputs_subdir() / "condition_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSummary → {summary_path}")
    print(f"Outputs → {outputs_subdir()}")
    print(f"Eval    → {evaluation_subdir()}")


if __name__ == "__main__":
    main()
