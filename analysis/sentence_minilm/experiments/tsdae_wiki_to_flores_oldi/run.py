"""
sentence-MiniLM: continued TSDAE pretraining on Wiki (13 varieties),
then evaluation on FLORES+ and OLDI.

Pipeline:
    1. Sample Wiki for all 13 varieties.
    2. TSDAE unsupervised pretraining of paraphrase-multilingual-MiniLM
       on the concatenated Wiki corpus.
    3. Save adapted SentenceTransformer under method_outputs/models/.
    4. Embed FLORES+ and OLDI with the adapted model and compute
       centroid + parallel-alignment evaluations.

Outputs mirror the multilingual_xlmr fine-tune experiment structure.

Launch (HPC, requires GPU):
    sbatch slurm/jobs/sentence_minilm__tsdae_wiki_to_flores_oldi.slurm
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
    VARIETY_CODES, SAMPLE_SIZE, RANDOM_STATE,
    SENTENCE_MODEL, MAX_LENGTH, BATCH_SIZE,
    experiment_dirs,
)
from analysis.sentence_minilm.core.data_loader import (
    load_wiki_for_training, load_flores_parallel, load_oldi_parallel,
)
from analysis.sentence_minilm.core.embedder import (
    embed_per_variety, aggregate_from_per_variety,
)
from analysis.sentence_minilm.core.evaluate import variety_eval, parallel_eval
from analysis.sentence_minilm.core.trainer import run_tsdae_training


METHOD = "sentence_minilm"
EXPERIMENT = "tsdae_wiki_to_flores_oldi"

# Default training hyperparameters
DEFAULT_EPOCHS = 1
DEFAULT_TRAIN_BATCH = 16
DEFAULT_LR = 2e-5


def _save_variety_vectors(X: np.ndarray, codes, out_dir: Path) -> None:
    np.savez_compressed(
        out_dir / "variety_vectors.npz",
        matrix=X.astype(np.float32), labels=np.asarray(codes),
    )
    pd.DataFrame(X, index=codes).to_csv(
        out_dir / "variety_vectors.csv", float_format="%.6f",
    )


def evaluate_on(test_name: str, test_data, model_path: str, codes, device):
    print(f"\n--- Eval on {test_name.upper()} ---")
    mo_root = SCRIPT_DIR / "method_outputs" / test_name
    mo_root.mkdir(parents=True, exist_ok=True)

    per_variety = embed_per_variety(
        test_data, codes, model_name_or_path=model_path,
        batch_size=BATCH_SIZE, device=device, max_length=MAX_LENGTH,
    )
    X, codes_out = aggregate_from_per_variety(per_variety, codes)
    _save_variety_vectors(X, codes_out, mo_root)

    _, er_centroid = experiment_dirs(SCRIPT_DIR, f"{test_name}/centroid")
    centroid_report = variety_eval(
        X, codes_out, out_dir=er_centroid,
        method_label=f"Sentence-MiniLM TSDAE ({test_name.upper()} centroid)",
    )

    _, er_parallel = experiment_dirs(SCRIPT_DIR, f"{test_name}/parallel")
    parallel_eval(
        per_variety, out_dir=er_parallel,
        method_label=f"Sentence-MiniLM TSDAE ({test_name.upper()} parallel)",
    )

    print(f"  centroid: sil_family={centroid_report['silhouette_family']:+.4f}  "
          f"sil_romance={centroid_report['silhouette_romance_vs_rest']:+.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size",     type=int,   default=SAMPLE_SIZE)
    parser.add_argument("--random-state",    type=int,   default=RANDOM_STATE)
    parser.add_argument("--epochs",          type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--train-batch-size",type=int,   default=DEFAULT_TRAIN_BATCH)
    parser.add_argument("--lr",              type=float, default=DEFAULT_LR)
    parser.add_argument("--device",          type=str,   default=None)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    print(f"{METHOD} — {EXPERIMENT}")
    print("=" * 60)
    print(f"  base_model   = {SENTENCE_MODEL}")
    print(f"  varieties    = {VARIETY_CODES} (training on all 13)")
    print(f"  sample_size  = {args.sample_size}")
    print(f"  epochs       = {args.epochs}")
    print(f"  train batch  = {args.train_batch_size}")
    print(f"  lr           = {args.lr}")
    print()

    mo_root = SCRIPT_DIR / "method_outputs"
    mo_root.mkdir(parents=True, exist_ok=True)
    model_dir = mo_root / "models" / "tsdae_wiki_13"

    # ------------------------------------------------------------------ #
    # Step 1: load Wiki training corpus
    # ------------------------------------------------------------------ #
    print("Loading Wiki (training) ...")
    wiki_data, wiki_stats = load_wiki_for_training(
        sample_size=args.sample_size, random_state=args.random_state,
    )
    wiki_stats["sample_size_param"] = args.sample_size
    wiki_stats["random_state"]      = args.random_state
    wiki_stats.to_csv(mo_root / "run_stats.csv", index=False)

    # Flatten all sentences (TSDAE doesn't need labels)
    all_sents: list[str] = []
    for code in VARIETY_CODES:
        if code in wiki_data:
            all_sents.extend(wiki_data[code])
    print(f"  total Wiki sentences for TSDAE: {len(all_sents):,}")

    # ------------------------------------------------------------------ #
    # Step 2: TSDAE continued pretraining
    # ------------------------------------------------------------------ #
    if args.skip_train and (model_dir / "modules.json").exists():
        print(f"\n[skip-train] Reusing checkpoint {model_dir}")
    else:
        run_tsdae_training(
            base_model=SENTENCE_MODEL,
            texts=all_sents,
            output_dir=model_dir,
            epochs=args.epochs,
            batch_size=args.train_batch_size,
            lr=args.lr,
            max_length=MAX_LENGTH,
        )

    write_run_meta(
        out_dir=mo_root, method=METHOD, experiment=EXPERIMENT,
        params={
            "sample_size":      args.sample_size,
            "random_state":     args.random_state,
            "base_model":       SENTENCE_MODEL,
            "epochs":           args.epochs,
            "train_batch_size": args.train_batch_size,
            "lr":               args.lr,
            "max_length":       MAX_LENGTH,
            "varieties":        VARIETY_CODES,
        },
    )

    # ------------------------------------------------------------------ #
    # Step 3: eval on FLORES + OLDI
    # ------------------------------------------------------------------ #
    print("\nLoading OLDI ...")
    oldi_data, _ = load_oldi_parallel(verbose=False)
    print("Loading FLORES ...")
    flores_data, _ = load_flores_parallel(verbose=False)

    evaluate_on("flores", flores_data, str(model_dir), VARIETY_CODES, args.device)
    evaluate_on("oldi",   oldi_data,   str(model_dir), VARIETY_CODES, args.device)

    print("\nDone.")


if __name__ == "__main__":
    main()
