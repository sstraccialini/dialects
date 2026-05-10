"""
LaBSE — MNRL fine-tuning on OLDI parallel pairs (ita↔dialect), eval on FLORES.

Cell 10 of the FINAL 12-cell experimental matrix (EXPERIMENTAL_PLAN.md §3.3, §3.5).

This is a methodological COUNTER-EXAMPLE: LaBSE was originally trained with
translation-ranking objectives (AMS / contrastive). When we fine-tune it with
MNRL on (Italian, dialect) parallel pairs, the model learns to COLLAPSE
translations into the same embedding — destroying the language-identity
signal we need to measure typological similarity. Expected: silhouette_romance
drops dramatically vs zero-shot baseline.

Setup:
  base model   : sentence-transformers/LaBSE
  loss         : MultipleNegativesRankingLoss (MNRL)
  pairs        : OLDI parallel ita↔dialect, 6 dialects × ~5167 ≈ 31k pairs
  text variant : NATIVE
  hyperparams  : 5 epochs, batch 64, lr 2e-5, max_length 128
  eval         : centroid + parallel on FLORES (1827 × 17)

Launch:
    python analysis/labse/experiments/labse_finetuned_oldi_dialects_native/run.py
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
from analysis._shared.dataset_loaders import load_flores, load_oldi_pairs
from analysis._shared.varieties import (
    VARIETY_CODES, VARIETY_GROUP, GROUP_NAMES, GROUP_COLORS, VARIETY_NAMES,
    ROMANCE_FAMILIES, DIALECT_FAMILIES, experiment_dirs,
)
from evaluation import run_evaluation, run_parallel_alignment


METHOD = "labse"
EXPERIMENT = "labse_finetuned_oldi_dialects_native"
TEXT_VARIANT = "native"

LABSE_MODEL = "sentence-transformers/LaBSE"
MAX_LENGTH = 128
BATCH_SIZE = 64

DEFAULT_EPOCHS      = 5
DEFAULT_TRAIN_BATCH = 64
DEFAULT_LR          = 2e-5


def _save_variety_vectors(X: np.ndarray, codes, out_dir: Path) -> None:
    np.savez_compressed(
        out_dir / "variety_vectors.npz",
        matrix=X.astype(np.float32), labels=np.asarray(codes),
    )
    pd.DataFrame(X, index=codes).to_csv(
        out_dir / "variety_vectors.csv", float_format="%.6f",
    )


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / np.where(norms == 0, 1.0, norms)


def _variety_eval(X, codes, out_dir, *, method_label: str):
    return run_evaluation(
        variety_vectors=np.asarray(X), variety_codes=list(codes),
        out_dir=out_dir, method_label=method_label,
        family_groups=VARIETY_GROUP, family_colors=GROUP_COLORS,
        family_display_names=GROUP_NAMES, display_names=VARIETY_NAMES,
        romance_families=ROMANCE_FAMILIES, dialect_families=DIALECT_FAMILIES,
        isotropy=False, isotropy_top_k_pc=1,
    )


def _parallel_eval(per_variety, out_dir, *, method_label: str):
    return run_parallel_alignment(
        sentence_vectors=per_variety, out_dir=out_dir, method_label=method_label,
        family_groups=VARIETY_GROUP, family_colors=GROUP_COLORS,
        family_display_names=GROUP_NAMES, display_names=VARIETY_NAMES,
    )


def _train_mnrl(base_model: str, pairs_df: pd.DataFrame, output_dir: Path,
                epochs: int, batch_size: int, lr: float, max_length: int) -> Path:
    """Fine-tune a SentenceTransformer with MultipleNegativesRankingLoss on (anchor, positive) pairs."""
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    print(f"\n[MNRL] base='{base_model}' pairs={len(pairs_df):,} epochs={epochs} lr={lr}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = [InputExample(texts=[r.ita, r.dial]) for r in pairs_df.itertuples(index=False)]
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    model = SentenceTransformer(base_model)
    model.max_seq_length = max_length
    loss = losses.MultipleNegativesRankingLoss(model=model)

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=int(0.1 * len(loader) * epochs),
        optimizer_params={"lr": lr},
        output_path=str(output_dir),
        show_progress_bar=True,
    )
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="LaBSE MNRL on OLDI dialects → FLORES (native)")
    parser.add_argument("--epochs",          type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--train-batch-size",type=int,   default=DEFAULT_TRAIN_BATCH)
    parser.add_argument("--lr",              type=float, default=DEFAULT_LR)
    parser.add_argument("--device",          type=str,   default=None)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    print(f"{METHOD} — {EXPERIMENT}")
    print("=" * 60)
    print(f"  base_model    = {LABSE_MODEL}")
    print(f"  text variant  = {TEXT_VARIANT}")
    print(f"  loss          = MNRL (contrastive)")
    print(f"  epochs        = {args.epochs}  batch={args.train_batch_size}  lr={args.lr}")
    print()

    mo_root = SCRIPT_DIR / "method_outputs"
    mo_root.mkdir(parents=True, exist_ok=True)
    model_dir = mo_root / "models" / "labse_mnrl"

    print(f"Loading OLDI pairs (ita↔dialect, {TEXT_VARIANT}) ...")
    pairs_df = load_oldi_pairs(text_variant=TEXT_VARIANT, verbose=True)
    print(f"  total pairs: {len(pairs_df):,}")

    pd.DataFrame({"n_pairs": [len(pairs_df)]}).to_csv(mo_root / "run_stats.csv", index=False)

    if args.skip_train and (model_dir / "modules.json").exists():
        print(f"\n[skip-train] reusing checkpoint {model_dir}")
    else:
        _train_mnrl(
            base_model=LABSE_MODEL,
            pairs_df=pairs_df, output_dir=model_dir,
            epochs=args.epochs, batch_size=args.train_batch_size,
            lr=args.lr, max_length=MAX_LENGTH,
        )

    write_run_meta(
        out_dir=mo_root, method=METHOD, experiment=EXPERIMENT,
        params={
            "base_model":       LABSE_MODEL,
            "text_variant":     TEXT_VARIANT,
            "loss":             "MNRL",
            "n_pairs":          len(pairs_df),
            "epochs":           args.epochs,
            "train_batch_size": args.train_batch_size,
            "lr":               args.lr,
            "max_length":       MAX_LENGTH,
        },
    )

    print(f"\nLoading FLORES ({TEXT_VARIANT}) ...")
    flores_data, _ = load_flores(text_variant=TEXT_VARIANT, verbose=False)

    print(f"\nLoading fine-tuned model from {model_dir} ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(model_dir), device=args.device)

    print("\n--- Encoding FLORES ---")
    per_variety = {}
    for code in VARIETY_CODES:
        if code not in flores_data:
            continue
        emb = model.encode(
            flores_data[code], batch_size=BATCH_SIZE,
            show_progress_bar=True, convert_to_numpy=True,
            normalize_embeddings=True,
        )
        per_variety[code] = emb.astype(np.float32)

    codes_out = [c for c in VARIETY_CODES if c in per_variety]
    X = np.vstack([per_variety[c].mean(axis=0) for c in codes_out])
    X = _l2_normalize(X).astype(np.float32)

    mo_flores = mo_root / "flores"
    mo_flores.mkdir(parents=True, exist_ok=True)
    _save_variety_vectors(X, codes_out, mo_flores)

    _, er_centroid = experiment_dirs(SCRIPT_DIR, "flores/centroid")
    rep = _variety_eval(X, codes_out, out_dir=er_centroid,
                        method_label="LaBSE MNRL-OLDI (FLORES centroid)")
    _, er_parallel = experiment_dirs(SCRIPT_DIR, "flores/parallel")
    _parallel_eval(per_variety, out_dir=er_parallel,
                   method_label="LaBSE MNRL-OLDI (FLORES parallel)")

    print(f"\n  centroid: sil_fam={rep['silhouette_family']:+.4f}  "
          f"sil_rom={rep['silhouette_romance_vs_rest']:+.4f}  "
          f"sil_rom_noDial={rep.get('silhouette_romance_no_dialects')}")
    print("\nDone.")


if __name__ == "__main__":
    main()
