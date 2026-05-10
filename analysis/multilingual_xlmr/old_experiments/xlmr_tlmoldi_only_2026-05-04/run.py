"""
Multilingual XLM-R: TLM (Translation LM) on OLDI pairs only, then eval on FLORES+.

The model is xlm-roberta-base, no Wiki step. Each (italian, dialect)
OLDI pair is fed as a sentence pair: `[CLS] ita [SEP] dial [SEP]`, with
standard MLM masking applied across the concatenation. Cross-lingual
masking encourages alignment between Italian and dialect representations.

Outputs:
    method_outputs/
    ├── models/                    adapted XLM-R checkpoint
    ├── flores/variety_vectors.{npz,csv}
    ├── run_stats.csv
    └── run_meta.json
    evaluation_results/
    └── flores/{centroid,parallel}/

Launch (HPC):
    sbatch slurm/jobs/multilingual_xlmr__tlm_oldi_to_flores.slurm
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
from analysis.multilingual_xlmr.core.config import (
    VARIETY_CODES, MODEL_NAME, MAX_LENGTH, BATCH_SIZE, experiment_dirs,
)
from analysis.multilingual_xlmr.core.data_loader import (
    load_flores_parallel, load_all_oldi_pairs, iter_labeled_sentences,
)
from analysis.multilingual_xlmr.core.embedder import (
    MultilingualEmbedder, aggregate_variety_vectors,
)
from analysis.multilingual_xlmr.core.evaluate import variety_eval, parallel_eval
from analysis.multilingual_xlmr.core.trainer import run_tlm_training


METHOD = "multilingual_xlmr"
EXPERIMENT = "tlm_oldi_to_flores"

DEFAULT_EPOCHS = 5
DEFAULT_TRAIN_BATCH = 16
DEFAULT_GRAD_ACCUM = 4
DEFAULT_LR = 3e-5
DEFAULT_MAX_LENGTH_TLM = 256


def _save_variety_vectors(X, codes, out_dir: Path) -> None:
    np.savez_compressed(out_dir / "variety_vectors.npz",
                        matrix=X.astype(np.float32), labels=np.asarray(codes))
    pd.DataFrame(X, index=codes).to_csv(out_dir / "variety_vectors.csv",
                                        float_format="%.6f")


def evaluate_flores(embedder, codes):
    print("\n--- Eval on FLORES ---")
    flores_data, _ = load_flores_parallel(verbose=False)
    mo_root = SCRIPT_DIR / "method_outputs" / "flores"
    mo_root.mkdir(parents=True, exist_ok=True)

    sents, sent_codes = iter_labeled_sentences(flores_data, codes=codes)
    sent_vecs = embedder.encode(sents, batch_size=BATCH_SIZE)
    X, codes_out = aggregate_variety_vectors(sent_vecs, sent_codes, codes)
    _save_variety_vectors(X, codes_out, mo_root)

    _, er_centroid = experiment_dirs(SCRIPT_DIR, "flores/centroid")
    centroid_report = variety_eval(
        X, codes_out, out_dir=er_centroid,
        method_label=f"XLM-R TLM-OLDI (FLORES centroid)",
    )

    per_variety = embedder.encode_per_variety(flores_data, codes, batch_size=BATCH_SIZE)
    _, er_parallel = experiment_dirs(SCRIPT_DIR, "flores/parallel")
    parallel_eval(per_variety, out_dir=er_parallel,
                  method_label=f"XLM-R TLM-OLDI (FLORES parallel)")

    print(f"  centroid: sil_family={centroid_report['silhouette_family']:+.4f}  "
          f"sil_romance={centroid_report['silhouette_romance_vs_rest']:+.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",          type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--train-batch-size",type=int,   default=DEFAULT_TRAIN_BATCH)
    parser.add_argument("--grad-accumulation",type=int,  default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--lr",              type=float, default=DEFAULT_LR)
    parser.add_argument("--max-length-tlm",  type=int,   default=DEFAULT_MAX_LENGTH_TLM)
    parser.add_argument("--device",          type=str,   default=None)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    print(f"{METHOD} — {EXPERIMENT}")
    print("=" * 60)
    print(f"  base_model    = {MODEL_NAME}")
    print(f"  epochs        = {args.epochs}")
    print(f"  train batch   = {args.train_batch_size} × grad_acc {args.grad_accumulation}")
    print(f"  lr            = {args.lr}")
    print(f"  max_len (TLM) = {args.max_length_tlm}")
    print()

    mo_root = SCRIPT_DIR / "method_outputs"
    mo_root.mkdir(parents=True, exist_ok=True)
    model_dir = mo_root / "models" / "tlm_oldi"

    # ------------------------------------------------------------------ #
    # Step 1: load OLDI Italian↔dialect pairs (6 dialects)
    # ------------------------------------------------------------------ #
    print("Loading OLDI pairs ...")
    pairs = load_all_oldi_pairs()
    print(f"  total OLDI pairs: {len(pairs):,}")

    pd.DataFrame({"n_pairs": [len(pairs)]}).to_csv(mo_root / "run_stats.csv", index=False)

    # ------------------------------------------------------------------ #
    # Step 2: TLM training
    # ------------------------------------------------------------------ #
    if args.skip_train and (model_dir / "config.json").exists():
        print(f"\n[skip-train] Reusing checkpoint {model_dir}")
    else:
        run_tlm_training(
            base_model=MODEL_NAME,
            pairs=pairs,
            output_dir=model_dir,
            epochs=args.epochs,
            batch_size=args.train_batch_size,
            grad_accumulation=args.grad_accumulation,
            lr=args.lr,
            max_length=args.max_length_tlm,
        )

    write_run_meta(
        out_dir=mo_root, method=METHOD, experiment=EXPERIMENT,
        params={
            "base_model":        MODEL_NAME,
            "n_pairs":           len(pairs),
            "epochs":            args.epochs,
            "train_batch_size":  args.train_batch_size,
            "grad_accumulation": args.grad_accumulation,
            "lr":                args.lr,
            "max_length_tlm":    args.max_length_tlm,
            "varieties":         VARIETY_CODES,
        },
    )

    # ------------------------------------------------------------------ #
    # Step 3: eval on FLORES+
    # ------------------------------------------------------------------ #
    embedder = MultilingualEmbedder(
        model_name=str(model_dir), device=args.device, max_length=MAX_LENGTH,
    )
    evaluate_flores(embedder, VARIETY_CODES)
    print("\nDone.")


if __name__ == "__main__":
    main()
