"""
CANINE — continued char-level MLM on Wiki+OLDI dialect-only (NATIVE), eval on FLORES.

Cell 12 of the FINAL 12-cell experimental matrix (EXPERIMENTAL_PLAN.md §3.3).

Setup:
  base model    : google/canine-c (character-level encoder)
  loss          : char-level MLM (custom head over a fixed Unicode vocab)
  text variant  : NATIVE
  training data : Wiki + OLDI dialect (cap rule §3.4) — DIALECTS ONLY (6 codes)
  hyperparams   : 3 epochs, batch 8 × grad_accum 8, lr 3e-5, max_length 512 chars
  eval          : centroid + parallel on FLORES — ALL 17 varieties

Launch:
    python analysis/canine/experiments/canine_finetuned_wikiOLDI_dialects_native/run.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis._shared.run_meta import write_run_meta
from analysis._shared.dataset_loaders import (
    load_wiki_plus_oldi_dialect, load_flores,
)
from analysis._shared.varieties import (
    VARIETY_CODES, DIALECT_CODES, SAMPLE_SIZE, RANDOM_STATE,
)
from analysis.canine.core.config import (
    DEFAULT_MODEL_NAME, MAX_LENGTH, BATCH_SIZE, experiment_dirs,
)
from analysis.canine.core.embedder import (
    CanineEmbedder, aggregate_variety_vectors,
)
from analysis.canine.core.evaluate import variety_eval, parallel_eval
from analysis.canine.core.trainer import run_mlm_training


METHOD = "canine"
EXPERIMENT = "canine_finetuned_wikiOLDI_dialects_native"
TEXT_VARIANT = "native"

DEFAULT_EPOCHS      = 3
DEFAULT_TRAIN_BATCH = 8
DEFAULT_GRAD_ACCUM  = 8
DEFAULT_LR          = 3e-5
DEFAULT_MAX_LENGTH  = 512


def _save_variety_vectors(X: np.ndarray, codes, out_dir: Path) -> None:
    np.savez_compressed(
        out_dir / "variety_vectors.npz",
        matrix=X.astype(np.float32), labels=np.asarray(codes),
    )
    pd.DataFrame(X, index=codes).to_csv(
        out_dir / "variety_vectors.csv", float_format="%.6f",
    )


def _flatten(data: dict, codes):
    sents, sent_codes = [], []
    for code in codes:
        if code not in data:
            continue
        for s in data[code]:
            sents.append(s)
            sent_codes.append(code)
    return sents, sent_codes


def evaluate_on(test_name: str, test_data, embedder, codes):
    print(f"\n--- Eval on {test_name.upper()} ---")
    mo = SCRIPT_DIR / "method_outputs" / test_name
    mo.mkdir(parents=True, exist_ok=True)

    sents, sent_codes = _flatten(test_data, codes)
    sent_vecs = embedder.encode(sents, batch_size=BATCH_SIZE)
    X, codes_out = aggregate_variety_vectors(sent_vecs, sent_codes, codes)
    _save_variety_vectors(X, codes_out, mo)
    _, er_centroid = experiment_dirs(SCRIPT_DIR, f"{test_name}/centroid")
    rep = variety_eval(X, codes_out, out_dir=er_centroid,
                       method_label=f"CANINE FT ({EXPERIMENT}, {test_name} centroid)")

    per_variety = embedder.encode_per_variety(test_data, codes, batch_size=BATCH_SIZE)
    _, er_parallel = experiment_dirs(SCRIPT_DIR, f"{test_name}/parallel")
    parallel_eval(per_variety, out_dir=er_parallel,
                  method_label=f"CANINE FT ({EXPERIMENT}, {test_name} parallel)")

    print(f"  centroid: sil_fam={rep['silhouette_family']:+.4f}  "
          f"sil_rom={rep['silhouette_romance_vs_rest']:+.4f}  "
          f"sil_rom_noDial={rep.get('silhouette_romance_no_dialects')}")


def main():
    parser = argparse.ArgumentParser(description="CANINE MLM finetune on Wiki+OLDI dialects → FLORES (native)")
    parser.add_argument("--sample-size",      type=int,   default=SAMPLE_SIZE)
    parser.add_argument("--random-state",     type=int,   default=RANDOM_STATE)
    parser.add_argument("--epochs",           type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--train-batch-size", type=int,   default=DEFAULT_TRAIN_BATCH)
    parser.add_argument("--grad-accumulation",type=int,   default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--lr",               type=float, default=DEFAULT_LR)
    parser.add_argument("--max-length",       type=int,   default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--device",           type=str,   default=None)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    print(f"{METHOD} — {EXPERIMENT}")
    print("=" * 60)
    print(f"  base_model    = {DEFAULT_MODEL_NAME}")
    print(f"  text variant  = {TEXT_VARIANT}")
    print(f"  train codes   = {DIALECT_CODES}  (only the 6 dialects)")
    print(f"  eval codes    = {VARIETY_CODES}  (all 17)")
    print(f"  sample_size   = {args.sample_size}")
    print(f"  epochs        = {args.epochs}  batch={args.train_batch_size}×accum {args.grad_accumulation}  lr={args.lr}")
    print(f"  max_length    = {args.max_length} chars")
    print()

    mo_root = SCRIPT_DIR / "method_outputs"
    mo_root.mkdir(parents=True, exist_ok=True)
    model_dir = mo_root / "models" / "canine_finetuned"

    print(f"Loading Wiki + OLDI (dialect cap rule, {TEXT_VARIANT}, dialects only) ...")
    train_data, train_stats = load_wiki_plus_oldi_dialect(
        text_variant=TEXT_VARIANT,
        codes=DIALECT_CODES,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )
    train_stats["sample_size_param"] = args.sample_size
    train_stats["random_state"]      = args.random_state
    train_stats.to_csv(mo_root / "run_stats.csv", index=False)

    sents, _ = _flatten(train_data, DIALECT_CODES)
    print(f"  total training sentences for MLM: {len(sents):,}")

    if args.skip_train and (model_dir / "config.json").exists():
        print(f"\n[skip-train] reusing checkpoint {model_dir}")
    else:
        run_mlm_training(
            base_model=DEFAULT_MODEL_NAME,
            texts=sents,
            output_dir=model_dir,
            epochs=args.epochs,
            batch_size=args.train_batch_size,
            grad_accumulation=args.grad_accumulation,
            lr=args.lr,
            max_length=args.max_length,
        )

    write_run_meta(
        out_dir=mo_root, method=METHOD, experiment=EXPERIMENT,
        params={
            "sample_size":       args.sample_size,
            "random_state":      args.random_state,
            "base_model":        DEFAULT_MODEL_NAME,
            "text_variant":      TEXT_VARIANT,
            "training_codes":    DIALECT_CODES,
            "eval_codes":        VARIETY_CODES,
            "epochs":            args.epochs,
            "train_batch_size":  args.train_batch_size,
            "grad_accumulation": args.grad_accumulation,
            "lr":                args.lr,
            "max_length":        args.max_length,
        },
    )

    print(f"\nLoading FLORES ({TEXT_VARIANT}) ...")
    flores_data, _ = load_flores(text_variant=TEXT_VARIANT, verbose=False)

    embedder = CanineEmbedder(
        model_name=str(model_dir), device=args.device, max_length=MAX_LENGTH,
    )
    evaluate_on("flores", flores_data, embedder, VARIETY_CODES)
    print("\nDone.")


if __name__ == "__main__":
    main()
