"""
Multilingual XLM-R: continued MLM pretraining on Wiki (13 varieties),
then evaluation on FLORES+ and OLDI.

Pipeline:
    1. Sample Wiki for all 13 varieties (6 dialects + 7 standard languages).
    2. Continued MLM pretraining of xlm-roberta-base on the concatenated
       Wiki corpus (the model sees the entire linguistic space).
    3. Save the adapted encoder under method_outputs/models/.
    4. Embed FLORES+ and OLDI sentences with the adapted encoder and
       compute centroid + parallel-alignment evaluations on each.

Outputs:
    method_outputs/
    ├── models/                    adapted XLM-R checkpoint (gitignored)
    ├── flores/variety_vectors.{npz,csv}
    ├── oldi/variety_vectors.{npz,csv}
    ├── run_stats.csv
    └── run_meta.json
    evaluation_results/
    ├── flores/{centroid,parallel}/
    └── oldi/{centroid,parallel}/

Launch (HPC, requires GPU):
    sbatch slurm/jobs/multilingual_xlmr__mlm_wiki_to_flores_oldi.slurm
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
    VARIETY_CODES, SAMPLE_SIZE, RANDOM_STATE,
    MODEL_NAME, MAX_LENGTH, BATCH_SIZE,
    experiment_dirs,
)
from analysis.multilingual_xlmr.core.data_loader import (
    load_wiki_for_training, load_flores_parallel, load_oldi_parallel,
    iter_labeled_sentences,
)
from analysis.multilingual_xlmr.core.embedder import (
    MultilingualEmbedder, aggregate_variety_vectors, aggregate_from_per_variety,
)
from analysis.multilingual_xlmr.core.evaluate import variety_eval, parallel_eval
from analysis.multilingual_xlmr.core.trainer import run_mlm_training


METHOD = "multilingual_xlmr"
EXPERIMENT = "mlm_wiki_to_flores_oldi"

# Default training hyperparameters (override via CLI on HPC if needed).
DEFAULT_EPOCHS = 3
DEFAULT_TRAIN_BATCH = 16
DEFAULT_GRAD_ACCUM = 4
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

    # Centroid eval — embed all sentences flat, aggregate per variety
    sents, sent_codes = iter_labeled_sentences(test_data, codes=codes)
    sent_vecs = embedder.encode(sents, batch_size=BATCH_SIZE)
    X, codes_out = aggregate_variety_vectors(sent_vecs, sent_codes, codes)
    _save_variety_vectors(X, codes_out, mo_root)
    _, er_centroid = experiment_dirs(SCRIPT_DIR, f"{test_name}/centroid")
    centroid_report = variety_eval(
        X, codes_out, out_dir=er_centroid,
        method_label=f"XLM-R adapted ({test_name.upper()} centroid)",
    )

    # Parallel eval — keep per-variety embeddings aligned by sentence
    per_variety = embedder.encode_per_variety(test_data, codes, batch_size=BATCH_SIZE)
    _, er_parallel = experiment_dirs(SCRIPT_DIR, f"{test_name}/parallel")
    parallel_eval(
        per_variety, out_dir=er_parallel,
        method_label=f"XLM-R adapted ({test_name.upper()} parallel)",
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
    parser.add_argument("--skip-train", action="store_true",
                        help="Reuse a previously trained checkpoint if present")
    args = parser.parse_args()

    print(f"{METHOD} — {EXPERIMENT}")
    print("=" * 60)
    print(f"  base_model   = {MODEL_NAME}")
    print(f"  varieties    = {VARIETY_CODES} (training on all 13)")
    print(f"  sample_size  = {args.sample_size}")
    print(f"  epochs       = {args.epochs}")
    print(f"  train batch  = {args.train_batch_size} × grad_acc {args.grad_accumulation}")
    print(f"  lr           = {args.lr}")
    print()

    mo_root = SCRIPT_DIR / "method_outputs"
    mo_root.mkdir(parents=True, exist_ok=True)
    model_dir = mo_root / "models" / "mlm_wiki_13"
    model_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Step 1: load training corpus (Wiki, 13 varieties)
    # ------------------------------------------------------------------ #
    print("Loading Wiki (training) ...")
    wiki_data, wiki_stats = load_wiki_for_training(
        sample_size=args.sample_size, random_state=args.random_state,
    )
    wiki_stats["sample_size_param"] = args.sample_size
    wiki_stats["random_state"]      = args.random_state
    wiki_stats.to_csv(mo_root / "run_stats.csv", index=False)

    sents, _ = iter_labeled_sentences(wiki_data)
    print(f"  total Wiki sentences for MLM: {len(sents):,}")

    # ------------------------------------------------------------------ #
    # Step 2: continued MLM pretraining
    # ------------------------------------------------------------------ #
    if args.skip_train and (model_dir / "config.json").exists():
        print(f"\n[skip-train] Reusing checkpoint {model_dir}")
    else:
        run_mlm_training(
            base_model=MODEL_NAME,
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
            "base_model":        MODEL_NAME,
            "epochs":            args.epochs,
            "train_batch_size":  args.train_batch_size,
            "grad_accumulation": args.grad_accumulation,
            "lr":                args.lr,
            "max_length":        MAX_LENGTH,
            "varieties":         VARIETY_CODES,
        },
    )

    # ------------------------------------------------------------------ #
    # Step 3: evaluate on FLORES + OLDI with the adapted encoder
    # ------------------------------------------------------------------ #
    print("\nLoading OLDI ...")
    oldi_data, _ = load_oldi_parallel(verbose=False)
    print("Loading FLORES ...")
    flores_data, _ = load_flores_parallel(verbose=False)

    embedder = MultilingualEmbedder(
        model_name=str(model_dir), device=args.device, max_length=MAX_LENGTH,
    )
    evaluate_on("flores", flores_data, embedder, VARIETY_CODES)
    evaluate_on("oldi",   oldi_data,   embedder, VARIETY_CODES)

    print("\nDone.")


if __name__ == "__main__":
    main()
