"""
sentence-MiniLM: MNRL (contrastive) on OLDI pairs only, then eval on FLORES+.

For each (italian, dialect) pair, MNRL pulls together the embedding of
the pair and pushes apart all other italians in the batch. Standard
contrastive objective for sentence-transformers.

Outputs mirror the multilingual_xlmr tlm_oldi_to_flores experiment.

Launch (HPC):
    sbatch slurm/jobs/sentence_minilm__mnrl_oldi_to_flores.slurm
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
from analysis.sentence_minilm.core.config import (
    VARIETY_CODES, SENTENCE_MODEL, MAX_LENGTH, BATCH_SIZE, experiment_dirs,
)
from analysis.sentence_minilm.core.data_loader import (
    load_flores_parallel, load_all_oldi_pairs,
)
from analysis.sentence_minilm.core.embedder import (
    embed_per_variety, aggregate_from_per_variety,
)
from analysis.sentence_minilm.core.evaluate import variety_eval, parallel_eval
from analysis.sentence_minilm.core.trainer import run_mnrl_training


METHOD = "sentence_minilm"
EXPERIMENT = "mnrl_oldi_to_flores"

DEFAULT_EPOCHS = 5
DEFAULT_TRAIN_BATCH = 64
DEFAULT_LR = 2e-5


def _save_variety_vectors(X, codes, out_dir: Path) -> None:
    np.savez_compressed(out_dir / "variety_vectors.npz",
                        matrix=X.astype(np.float32), labels=np.asarray(codes))
    pd.DataFrame(X, index=codes).to_csv(out_dir / "variety_vectors.csv",
                                        float_format="%.6f")


def evaluate_flores(model_path: str, codes, device):
    print("\n--- Eval on FLORES ---")
    flores_data, _ = load_flores_parallel(verbose=False)
    mo_root = SCRIPT_DIR / "method_outputs" / "flores"
    mo_root.mkdir(parents=True, exist_ok=True)

    per_variety = embed_per_variety(
        flores_data, codes, model_name_or_path=model_path,
        batch_size=BATCH_SIZE, device=device, max_length=MAX_LENGTH,
    )
    X, codes_out = aggregate_from_per_variety(per_variety, codes)
    _save_variety_vectors(X, codes_out, mo_root)

    _, er_centroid = experiment_dirs(SCRIPT_DIR, "flores/centroid")
    centroid_report = variety_eval(
        X, codes_out, out_dir=er_centroid,
        method_label=f"Sentence-MiniLM MNRL-OLDI (FLORES centroid)",
    )
    _, er_parallel = experiment_dirs(SCRIPT_DIR, "flores/parallel")
    parallel_eval(per_variety, out_dir=er_parallel,
                  method_label=f"Sentence-MiniLM MNRL-OLDI (FLORES parallel)")

    print(f"  centroid: sil_family={centroid_report['silhouette_family']:+.4f}  "
          f"sil_romance={centroid_report['silhouette_romance_vs_rest']:+.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",          type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--train-batch-size",type=int,   default=DEFAULT_TRAIN_BATCH)
    parser.add_argument("--lr",              type=float, default=DEFAULT_LR)
    parser.add_argument("--device",          type=str,   default=None)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    print(f"{METHOD} — {EXPERIMENT}")
    print("=" * 60)
    print(f"  base_model    = {SENTENCE_MODEL}")
    print(f"  epochs        = {args.epochs}")
    print(f"  train batch   = {args.train_batch_size}")
    print(f"  lr            = {args.lr}")
    print()

    mo_root = SCRIPT_DIR / "method_outputs"
    mo_root.mkdir(parents=True, exist_ok=True)
    model_dir = mo_root / "models" / "mnrl_oldi"

    print("Loading OLDI pairs ...")
    pairs = load_all_oldi_pairs()
    print(f"  total OLDI pairs: {len(pairs):,}")

    pd.DataFrame({"n_pairs": [len(pairs)]}).to_csv(mo_root / "run_stats.csv", index=False)

    if args.skip_train and (model_dir / "modules.json").exists():
        print(f"\n[skip-train] Reusing checkpoint {model_dir}")
    else:
        run_mnrl_training(
            base_model=SENTENCE_MODEL,
            pairs=pairs,
            output_dir=model_dir,
            epochs=args.epochs,
            batch_size=args.train_batch_size,
            lr=args.lr,
            max_length=MAX_LENGTH,
        )

    write_run_meta(
        out_dir=mo_root, method=METHOD, experiment=EXPERIMENT,
        params={
            "base_model":       SENTENCE_MODEL,
            "n_pairs":          len(pairs),
            "epochs":           args.epochs,
            "train_batch_size": args.train_batch_size,
            "lr":               args.lr,
            "max_length":       MAX_LENGTH,
            "varieties":        VARIETY_CODES,
        },
    )

    evaluate_flores(str(model_dir), VARIETY_CODES, args.device)
    print("\nDone.")


if __name__ == "__main__":
    main()
