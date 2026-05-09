"""
ABLATION CELL — CANINE + NATIVE text + dialect-only training.

Purpose: complete the 2x2 ablation matrix (text variant × training regime)
for CANINE, alongside:
  - mlm_wiki_dialects_normalized/  (norm + 6 dial, +0.619 sil_rom)  done
  - mlm_wiki_to_flores_oldi/       (native + 13 var, +0.217)         done
  - mlm_wiki_rehearsal_normalized/ (norm + 13 var)                  todo
  - this run                       (native + 6 dial)                 ←

Setup:
  - text variant : NATIVE (with diacritics, case, punctuation)
  - training set : DIALECT_CODES only (fur, lij, lmo, sc, scn, vec)
  - eval         : FLORES + OLDI on all 13 varieties (centroid + parallel)

Uses the NATIVE path aliases from analysis.canine.core.config (which we
already aliased to the not_normalized FLORES/OLDI/Wiki). The ablation
question "is the regression caused by rehearsal or by native text?" is
answered by comparing this cell's sil_romance to the dialects_normalized
cell (+0.619).

Launch (HPC):
    sbatch slurm/jobs/canine__mlm_wiki_dialects_native.slurm
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis._shared.run_meta import write_run_meta
from analysis.canine.core.config import (
    DIALECT_CODES, VARIETY_CODES, SAMPLE_SIZE, RANDOM_STATE,
    DEFAULT_MODEL_NAME, MAX_LENGTH, BATCH_SIZE,
    experiment_dirs,
)
from analysis.canine.core.data_loader import (
    load_wiki_for_training, load_flores_parallel, load_oldi_parallel,
    iter_labeled_sentences,
)
from analysis.canine.core.embedder import (
    CanineEmbedder, aggregate_variety_vectors,
)
from analysis.canine.core.evaluate import variety_eval, parallel_eval
from analysis.canine.core.trainer import run_mlm_training


METHOD = "canine"
EXPERIMENT = "mlm_wiki_dialects_native"

DEFAULT_EPOCHS      = 3
DEFAULT_TRAIN_BATCH = 8
DEFAULT_GRAD_ACCUM  = 8
DEFAULT_LR          = 3e-5
DEFAULT_SAMPLE_SIZE = 100_000  # consistent with the +0.62 baseline


def _save_variety_vectors(X: np.ndarray, codes, out_dir: Path) -> None:
    np.savez_compressed(
        out_dir / "variety_vectors.npz",
        matrix=X.astype(np.float32), labels=np.asarray(codes),
    )
    pd.DataFrame(X, index=codes).to_csv(
        out_dir / "variety_vectors.csv", float_format="%.6f",
    )


def evaluate_on(test_name: str, test_data, embedder, codes):
    print(f"\n--- Eval on {test_name.upper()} ---")
    mo_root = SCRIPT_DIR / "method_outputs" / test_name
    mo_root.mkdir(parents=True, exist_ok=True)

    sents, sent_codes = iter_labeled_sentences(test_data, codes=codes)
    sent_vecs = embedder.encode(sents, batch_size=BATCH_SIZE)
    X, codes_out = aggregate_variety_vectors(sent_vecs, sent_codes, codes)
    _save_variety_vectors(X, codes_out, mo_root)
    _, er_centroid = experiment_dirs(SCRIPT_DIR, f"{test_name}/centroid")
    centroid_report = variety_eval(
        X, codes_out, out_dir=er_centroid,
        method_label=f"CANINE adapted ({test_name.upper()} centroid, dialects-native)",
    )

    per_variety = embedder.encode_per_variety(test_data, codes, batch_size=BATCH_SIZE)
    _, er_parallel = experiment_dirs(SCRIPT_DIR, f"{test_name}/parallel")
    parallel_eval(
        per_variety, out_dir=er_parallel,
        method_label=f"CANINE adapted ({test_name.upper()} parallel, dialects-native)",
    )

    print(f"  centroid: sil_family={centroid_report['silhouette_family']:+.4f}  "
          f"sil_romance={centroid_report['silhouette_romance_vs_rest']:+.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size",      type=int,   default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--random-state",     type=int,   default=RANDOM_STATE)
    parser.add_argument("--epochs",           type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--train-batch-size", type=int,   default=DEFAULT_TRAIN_BATCH)
    parser.add_argument("--grad-accumulation",type=int,   default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--lr",               type=float, default=DEFAULT_LR)
    parser.add_argument("--device",           type=str,   default=None)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    print(f"{METHOD} — {EXPERIMENT}")
    print("=" * 60)
    print(f"  base_model   = {DEFAULT_MODEL_NAME}")
    print(f"  text variant = NATIVE (cased, accented, punctuated)")
    print(f"  train codes  = {DIALECT_CODES}  (only the 6 dialects)")
    print(f"  eval codes   = {VARIETY_CODES}  (all 13 varieties)")
    print(f"  sample_size  = {args.sample_size}  (ablation cap)")
    print(f"  epochs       = {args.epochs}")
    print(f"  train batch  = {args.train_batch_size} × grad_acc {args.grad_accumulation}")
    print(f"  lr           = {args.lr}")
    print()

    mo_root = SCRIPT_DIR / "method_outputs"
    mo_root.mkdir(parents=True, exist_ok=True)
    model_dir = mo_root / "models" / "mlm_wiki_dialects_native"

    print(f"Loading Wiki native (training, {len(DIALECT_CODES)} dialects only) ...")
    wiki_data, wiki_stats = load_wiki_for_training(
        codes=DIALECT_CODES,
        sample_size=args.sample_size, random_state=args.random_state,
    )
    wiki_stats["sample_size_param"] = args.sample_size
    wiki_stats["random_state"]      = args.random_state
    wiki_stats.to_csv(mo_root / "run_stats.csv", index=False)

    sents, _ = iter_labeled_sentences(wiki_data, codes=DIALECT_CODES)
    print(f"  total Wiki sentences for MLM: {len(sents):,}")

    if args.skip_train and (model_dir / "config.json").exists():
        print(f"\n[skip-train] Reusing checkpoint {model_dir}")
    else:
        run_mlm_training(
            base_model=DEFAULT_MODEL_NAME,
            texts=sents,
            output_dir=model_dir,
            epochs=args.epochs,
            batch_size=args.train_batch_size,
            grad_accumulation=args.grad_accumulation,
            lr=args.lr,
            max_length=MAX_LENGTH,
        )

    write_run_meta(
        out_dir=mo_root, method=METHOD, experiment=EXPERIMENT,
        params={
            "sample_size":       args.sample_size,
            "random_state":      args.random_state,
            "base_model":        DEFAULT_MODEL_NAME,
            "text_variant":      "native",
            "training_codes":    DIALECT_CODES,
            "eval_codes":        VARIETY_CODES,
            "epochs":            args.epochs,
            "train_batch_size":  args.train_batch_size,
            "grad_accumulation": args.grad_accumulation,
            "lr":                args.lr,
            "max_length":        MAX_LENGTH,
        },
    )

    print("Loading FLORES native ...")
    flores_data, _ = load_flores_parallel(verbose=False)
    # Test ONLY on FLORES (the 7 new standards have no OLDI counterparts).

    embedder = CanineEmbedder(
        model_name=str(model_dir), device=args.device, max_length=MAX_LENGTH,
    )
    evaluate_on("flores", flores_data, embedder, VARIETY_CODES)

    print("\nDone.")


if __name__ == "__main__":
    main()
