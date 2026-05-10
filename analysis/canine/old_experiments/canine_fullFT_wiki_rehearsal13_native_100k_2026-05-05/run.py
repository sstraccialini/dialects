"""
CANINE: continued MLM pretraining on Wiki (6 dialects only), then
evaluation on FLORES+ and OLDI (all 13 varieties).

CANINE is a tokenizer-free char-level encoder. We only fine-tune on the
6 Italo-Romance dialects (fur/lij/lmo/sc/scn/vec) since the model
already has decent character-level coverage of the 7 standard languages
from pre-training. Each dialect contributes its natural Wikipedia
footprint — no balanced down-sampling.

Note: CANINE has no native MaskedLM head in HuggingFace; the trainer
adds a custom char-level prediction head (see core/trainer.py).

Outputs mirror the multilingual_xlmr fine-tune experiment structure.

Launch (HPC, requires GPU):
    sbatch slurm/jobs/canine__mlm_wiki_to_flores_oldi.slurm
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
    VARIETY_CODES, DIALECT_CODES, SAMPLE_SIZE, RANDOM_STATE,
    DEFAULT_MODEL_NAME, MAX_LENGTH, BATCH_SIZE,
    experiment_dirs,
)
from analysis.canine.core.data_loader import (
    load_wiki_for_training, load_flores_parallel, load_oldi_parallel,
    iter_labeled_sentences,
)
from analysis.canine.core.embedder import (
    CanineEmbedder, aggregate_variety_vectors, aggregate_from_per_variety,
)
from analysis.canine.core.evaluate import variety_eval, parallel_eval
from analysis.canine.core.trainer import run_mlm_training


METHOD = "canine"
EXPERIMENT = "mlm_wiki_to_flores_oldi"

# Default training hyperparameters (CANINE has heavier sequences than XLM-R)
DEFAULT_EPOCHS = 3
DEFAULT_TRAIN_BATCH = 8
DEFAULT_GRAD_ACCUM = 8
DEFAULT_LR = 3e-5


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
        method_label=f"CANINE adapted ({test_name.upper()} centroid)",
    )

    per_variety = embedder.encode_per_variety(test_data, codes, batch_size=BATCH_SIZE)
    _, er_parallel = experiment_dirs(SCRIPT_DIR, f"{test_name}/parallel")
    parallel_eval(
        per_variety, out_dir=er_parallel,
        method_label=f"CANINE adapted ({test_name.upper()} parallel)",
    )

    print(f"  centroid: sil_family={centroid_report['silhouette_family']:+.4f}  "
          f"sil_romance={centroid_report['silhouette_romance_vs_rest']:+.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size",     type=int,   default=SAMPLE_SIZE)
    parser.add_argument("--random-state",    type=int,   default=RANDOM_STATE)
    parser.add_argument("--epochs",          type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--train-batch-size",type=int,   default=DEFAULT_TRAIN_BATCH)
    parser.add_argument("--grad-accumulation",type=int,  default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--lr",              type=float, default=DEFAULT_LR)
    parser.add_argument("--device",          type=str,   default=None)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    print(f"{METHOD} — {EXPERIMENT}")
    print("=" * 60)
    print(f"  base_model   = {DEFAULT_MODEL_NAME}")
    print(f"  train codes  = {VARIETY_CODES}  (all 13 varieties — rehearsal-style)")
    print(f"  eval codes   = {VARIETY_CODES}  (all 13 varieties)")
    print(f"  sample_size  = {args.sample_size}  (cap per variety, balanced sampling)")
    print(f"  epochs       = {args.epochs}")
    print(f"  train batch  = {args.train_batch_size} × grad_acc {args.grad_accumulation}")
    print(f"  lr           = {args.lr}")
    print()

    mo_root = SCRIPT_DIR / "method_outputs"
    mo_root.mkdir(parents=True, exist_ok=True)
    model_dir = mo_root / "models" / "mlm_wiki_dialects"

    # ------------------------------------------------------------------ #
    # Step 1: Wiki training corpus — ALL 13 varieties (rehearsal-style)
    # to prevent catastrophic forgetting of standards and to keep the
    # adaptation symmetric across dialect/standard regions of the space.
    # ------------------------------------------------------------------ #
    print(f"Loading Wiki (training, {len(VARIETY_CODES)} varieties — rehearsal) ...")
    wiki_data, wiki_stats = load_wiki_for_training(
        codes=VARIETY_CODES,
        sample_size=args.sample_size, random_state=args.random_state,
    )
    wiki_stats["sample_size_param"] = args.sample_size
    wiki_stats["random_state"]      = args.random_state
    wiki_stats.to_csv(mo_root / "run_stats.csv", index=False)

    sents, _ = iter_labeled_sentences(wiki_data, codes=VARIETY_CODES)
    print(f"  total Wiki sentences for MLM: {len(sents):,}")

    # ------------------------------------------------------------------ #
    # Step 2: continued MLM pretraining
    # ------------------------------------------------------------------ #
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
            "epochs":            args.epochs,
            "train_batch_size":  args.train_batch_size,
            "grad_accumulation": args.grad_accumulation,
            "lr":                args.lr,
            "max_length":        MAX_LENGTH,
            "training_codes":    VARIETY_CODES,
            "eval_codes":        VARIETY_CODES,
        },
    )

    # ------------------------------------------------------------------ #
    # Step 3: eval on FLORES + OLDI
    # ------------------------------------------------------------------ #
    print("\nLoading OLDI ...")
    oldi_data, _ = load_oldi_parallel(verbose=False)
    print("Loading FLORES ...")
    flores_data, _ = load_flores_parallel(verbose=False)

    embedder = CanineEmbedder(
        model_name=str(model_dir), device=args.device, max_length=MAX_LENGTH,
    )
    evaluate_on("flores", flores_data, embedder, VARIETY_CODES)
    evaluate_on("oldi",   oldi_data,   embedder, VARIETY_CODES)

    print("\nDone.")


if __name__ == "__main__":
    main()
