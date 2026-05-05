"""
sentence-MiniLM: sequential TSDAE-on-Wiki then MNRL-on-OLDI, eval on FLORES+.

Stage 1 — TSDAE unsupervised on Wiki for the 6 dialects ONLY
(fur/lij/lmo/sc/scn/vec). The base model already covers the 7 standard
languages; each dialect contributes its natural Wikipedia footprint.
Stage 2 — MNRL contrastive on OLDI (italian, dialect) pairs FROM stage 1.
Stage 3 — Eval on FLORES+ (centroid + parallel) over all 13 varieties.

Launch (HPC):
    sbatch slurm/jobs/sentence_minilm__tsdae_wiki_then_mnrl_oldi_to_flores.slurm
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
    VARIETY_CODES, DIALECT_CODES, SAMPLE_SIZE, RANDOM_STATE,
    SENTENCE_MODEL, MAX_LENGTH, BATCH_SIZE, experiment_dirs,
)
from analysis.sentence_minilm.core.data_loader import (
    load_wiki_for_training, load_flores_parallel, load_all_oldi_pairs,
)
from analysis.sentence_minilm.core.embedder import (
    embed_per_variety, aggregate_from_per_variety,
)
from analysis.sentence_minilm.core.evaluate import variety_eval, parallel_eval
from analysis.sentence_minilm.core.trainer import (
    run_tsdae_training, run_mnrl_training,
)


METHOD = "sentence_minilm"
EXPERIMENT = "tsdae_wiki_then_mnrl_oldi_to_flores"

DEFAULT_TSDAE_EPOCHS = 1
DEFAULT_MNRL_EPOCHS = 5
DEFAULT_TSDAE_BATCH = 16
DEFAULT_MNRL_BATCH = 64
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
        method_label="MiniLM TSDAE-Wiki→MNRL-OLDI (FLORES centroid)",
    )
    _, er_parallel = experiment_dirs(SCRIPT_DIR, "flores/parallel")
    parallel_eval(per_variety, out_dir=er_parallel,
                  method_label="MiniLM TSDAE-Wiki→MNRL-OLDI (FLORES parallel)")

    print(f"  centroid: sil_family={centroid_report['silhouette_family']:+.4f}  "
          f"sil_romance={centroid_report['silhouette_romance_vs_rest']:+.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size",   type=int,   default=SAMPLE_SIZE)
    parser.add_argument("--random-state",  type=int,   default=RANDOM_STATE)
    parser.add_argument("--tsdae-epochs",  type=int,   default=DEFAULT_TSDAE_EPOCHS)
    parser.add_argument("--mnrl-epochs",   type=int,   default=DEFAULT_MNRL_EPOCHS)
    parser.add_argument("--tsdae-batch",   type=int,   default=DEFAULT_TSDAE_BATCH)
    parser.add_argument("--mnrl-batch",    type=int,   default=DEFAULT_MNRL_BATCH)
    parser.add_argument("--lr",            type=float, default=DEFAULT_LR)
    parser.add_argument("--device",        type=str,   default=None)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    print(f"{METHOD} — {EXPERIMENT}")
    print("=" * 60)

    mo_root = SCRIPT_DIR / "method_outputs"
    mo_root.mkdir(parents=True, exist_ok=True)
    stage1_dir = mo_root / "models" / "stage1_tsdae_wiki"
    stage2_dir = mo_root / "models" / "stage2_mnrl_oldi"

    # ------------------------------------------------------------------ #
    # STAGE 1: TSDAE on Wiki — 6 dialects ONLY
    # ------------------------------------------------------------------ #
    print(f"\n[STAGE 1/3] TSDAE on Wiki ({len(DIALECT_CODES)} dialects only)")
    if args.skip_train and (stage1_dir / "modules.json").exists():
        print(f"  [skip-train] Reusing stage1 checkpoint {stage1_dir}")
    else:
        wiki_data, wiki_stats = load_wiki_for_training(
            codes=DIALECT_CODES,
            sample_size=args.sample_size, random_state=args.random_state,
        )
        wiki_stats["sample_size_param"] = args.sample_size
        wiki_stats["random_state"]      = args.random_state
        wiki_stats.to_csv(mo_root / "run_stats.csv", index=False)
        all_sents = []
        for code in DIALECT_CODES:
            if code in wiki_data:
                all_sents.extend(wiki_data[code])
        print(f"  total Wiki sentences: {len(all_sents):,}")
        run_tsdae_training(
            base_model=SENTENCE_MODEL,
            texts=all_sents, output_dir=stage1_dir,
            epochs=args.tsdae_epochs, batch_size=args.tsdae_batch,
            lr=args.lr, max_length=MAX_LENGTH,
        )

    # ------------------------------------------------------------------ #
    # STAGE 2: MNRL on OLDI starting from stage1
    # ------------------------------------------------------------------ #
    print("\n[STAGE 2/3] MNRL on OLDI (starting from stage1)")
    if args.skip_train and (stage2_dir / "modules.json").exists():
        print(f"  [skip-train] Reusing stage2 checkpoint {stage2_dir}")
    else:
        pairs = load_all_oldi_pairs()
        print(f"  total OLDI pairs: {len(pairs):,}")
        run_mnrl_training(
            base_model=str(stage1_dir),
            pairs=pairs, output_dir=stage2_dir,
            epochs=args.mnrl_epochs, batch_size=args.mnrl_batch,
            lr=args.lr, max_length=MAX_LENGTH,
        )

    write_run_meta(
        out_dir=mo_root, method=METHOD, experiment=EXPERIMENT,
        params={
            "base_model":     SENTENCE_MODEL,
            "sample_size":    args.sample_size,
            "tsdae_epochs":   args.tsdae_epochs,
            "mnrl_epochs":    args.mnrl_epochs,
            "tsdae_batch":    args.tsdae_batch,
            "mnrl_batch":     args.mnrl_batch,
            "lr":             args.lr,
            "max_length":     MAX_LENGTH,
            "stage1_codes":   DIALECT_CODES,
            "eval_codes":     VARIETY_CODES,
        },
    )

    # ------------------------------------------------------------------ #
    # STAGE 3: eval on FLORES+
    # ------------------------------------------------------------------ #
    print("\n[STAGE 3/3] Eval on FLORES")
    evaluate_flores(str(stage2_dir), VARIETY_CODES, args.device)
    print("\nDone.")


if __name__ == "__main__":
    main()
