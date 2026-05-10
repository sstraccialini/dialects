"""
CANINE: sequential MLM-on-Wiki then TLM-on-OLDI, eval on FLORES+.

Stage 1 — MLM continued pretraining on Wiki for the 6 dialects ONLY
(fur/lij/lmo/sc/scn/vec). Each dialect contributes its natural Wikipedia
footprint. Uses a custom char-level MLM head (see core/trainer.py).
Stage 2 — Sentence-pair MLM on OLDI (italian, dialect) FROM stage1.
Stage 3 — Eval on FLORES+ (centroid + parallel) over all 13 varieties.

Launch (HPC):
    sbatch slurm/jobs/canine__mlm_wiki_then_tlm_oldi_to_flores.slurm
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
    DEFAULT_MODEL_NAME, MAX_LENGTH, BATCH_SIZE, experiment_dirs,
)
from analysis.canine.core.data_loader import (
    load_wiki_for_training, load_flores_parallel, load_all_oldi_pairs,
    iter_labeled_sentences,
)
from analysis.canine.core.embedder import (
    CanineEmbedder, aggregate_variety_vectors,
)
from analysis.canine.core.evaluate import variety_eval, parallel_eval
from analysis.canine.core.trainer import run_mlm_training, run_tlm_training


METHOD = "canine"
EXPERIMENT = "mlm_wiki_then_tlm_oldi_to_flores"

DEFAULT_MLM_EPOCHS = 3
DEFAULT_TLM_EPOCHS = 5
DEFAULT_MLM_BATCH = 8
DEFAULT_MLM_GRAD = 8
DEFAULT_TLM_BATCH = 4
DEFAULT_TLM_GRAD = 16
DEFAULT_LR = 3e-5
DEFAULT_MAX_LENGTH_TLM = 1024


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
        method_label="CANINE MLM-Wiki→TLM-OLDI (FLORES centroid)",
    )
    per_variety = embedder.encode_per_variety(flores_data, codes, batch_size=BATCH_SIZE)
    _, er_parallel = experiment_dirs(SCRIPT_DIR, "flores/parallel")
    parallel_eval(per_variety, out_dir=er_parallel,
                  method_label="CANINE MLM-Wiki→TLM-OLDI (FLORES parallel)")

    print(f"  centroid: sil_family={centroid_report['silhouette_family']:+.4f}  "
          f"sil_romance={centroid_report['silhouette_romance_vs_rest']:+.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size",     type=int,   default=SAMPLE_SIZE)
    parser.add_argument("--random-state",    type=int,   default=RANDOM_STATE)
    parser.add_argument("--mlm-epochs",      type=int,   default=DEFAULT_MLM_EPOCHS)
    parser.add_argument("--tlm-epochs",      type=int,   default=DEFAULT_TLM_EPOCHS)
    parser.add_argument("--mlm-batch",       type=int,   default=DEFAULT_MLM_BATCH)
    parser.add_argument("--mlm-grad-accum",  type=int,   default=DEFAULT_MLM_GRAD)
    parser.add_argument("--tlm-batch",       type=int,   default=DEFAULT_TLM_BATCH)
    parser.add_argument("--tlm-grad-accum",  type=int,   default=DEFAULT_TLM_GRAD)
    parser.add_argument("--lr",              type=float, default=DEFAULT_LR)
    parser.add_argument("--max-length-tlm",  type=int,   default=DEFAULT_MAX_LENGTH_TLM)
    parser.add_argument("--device",          type=str,   default=None)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    print(f"{METHOD} — {EXPERIMENT}")
    print("=" * 60)

    mo_root = SCRIPT_DIR / "method_outputs"
    mo_root.mkdir(parents=True, exist_ok=True)
    stage1_dir = mo_root / "models" / "stage1_mlm_wiki"
    stage2_dir = mo_root / "models" / "stage2_tlm_oldi"

    # ------------------------------------------------------------------ #
    # STAGE 1: MLM on Wiki — 6 dialects ONLY
    # ------------------------------------------------------------------ #
    print(f"\n[STAGE 1/3] MLM on Wiki ({len(DIALECT_CODES)} dialects only)")
    if args.skip_train and (stage1_dir / "config.json").exists():
        print(f"  [skip-train] Reusing stage1 checkpoint {stage1_dir}")
    else:
        wiki_data, wiki_stats = load_wiki_for_training(
            codes=DIALECT_CODES,
            sample_size=args.sample_size, random_state=args.random_state,
        )
        wiki_stats["sample_size_param"] = args.sample_size
        wiki_stats["random_state"]      = args.random_state
        wiki_stats.to_csv(mo_root / "run_stats.csv", index=False)
        sents, _ = iter_labeled_sentences(wiki_data, codes=DIALECT_CODES)
        print(f"  total Wiki sentences: {len(sents):,}")
        run_mlm_training(
            base_model=DEFAULT_MODEL_NAME, texts=sents, output_dir=stage1_dir,
            epochs=args.mlm_epochs,
            batch_size=args.mlm_batch,
            grad_accumulation=args.mlm_grad_accum,
            lr=args.lr, max_length=MAX_LENGTH,
        )

    # ------------------------------------------------------------------ #
    # STAGE 2: TLM on OLDI starting from stage1
    # ------------------------------------------------------------------ #
    print("\n[STAGE 2/3] TLM on OLDI (starting from stage1)")
    if args.skip_train and (stage2_dir / "config.json").exists():
        print(f"  [skip-train] Reusing stage2 checkpoint {stage2_dir}")
    else:
        pairs = load_all_oldi_pairs()
        print(f"  total OLDI pairs: {len(pairs):,}")
        run_tlm_training(
            base_model=str(stage1_dir),
            pairs=pairs, output_dir=stage2_dir,
            epochs=args.tlm_epochs,
            batch_size=args.tlm_batch,
            grad_accumulation=args.tlm_grad_accum,
            lr=args.lr,
            max_length=args.max_length_tlm,
        )

    write_run_meta(
        out_dir=mo_root, method=METHOD, experiment=EXPERIMENT,
        params={
            "base_model":     DEFAULT_MODEL_NAME,
            "sample_size":    args.sample_size,
            "mlm_epochs":     args.mlm_epochs,
            "tlm_epochs":     args.tlm_epochs,
            "mlm_batch":      args.mlm_batch,
            "mlm_grad_accum": args.mlm_grad_accum,
            "tlm_batch":      args.tlm_batch,
            "tlm_grad_accum": args.tlm_grad_accum,
            "lr":             args.lr,
            "max_length":     MAX_LENGTH,
            "max_length_tlm": args.max_length_tlm,
            "stage1_codes":   DIALECT_CODES,
            "eval_codes":     VARIETY_CODES,
        },
    )

    # ------------------------------------------------------------------ #
    # STAGE 3: eval on FLORES+
    # ------------------------------------------------------------------ #
    print("\n[STAGE 3/3] Eval on FLORES")
    embedder = CanineEmbedder(
        model_name=str(stage2_dir), device=args.device, max_length=MAX_LENGTH,
    )
    evaluate_flores(embedder, VARIETY_CODES)
    print("\nDone.")


if __name__ == "__main__":
    main()
