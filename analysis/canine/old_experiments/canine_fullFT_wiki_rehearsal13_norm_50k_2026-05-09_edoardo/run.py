"""
ABLATION CELL — CANINE + NORMALIZED text + 13-variety rehearsal training.

Purpose: complete the 2x2 ablation matrix (text variant × training regime)
for CANINE, alongside:
  - mlm_wiki_dialects_normalized/  (norm + 6 dial, +0.619 sil_rom)  done
  - mlm_wiki_to_flores_oldi/       (native + 13 var, +0.217)         done
  - mlm_wiki_dialects_native/      (native + 6 dial)                 todo
  - this run                       (norm + 13 var)                   ←

Setup:
  - text variant : NORMALIZED (lowercase ASCII)
  - training set : VARIETY_CODES, all 13 (rehearsal-style)
  - eval         : FLORES + OLDI on all 13 varieties

Bypasses canine.core.config aliases (which point to NATIVE) and uses the
NORMALIZED paths from analysis._shared.varieties directly.

Launch (HPC):
    sbatch slurm/jobs/canine__mlm_wiki_rehearsal_normalized.slurm
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis._shared.varieties import (
    WIKI_NORMALIZED_DIR, FLORES_DIR as FLORES_NORM_DIR, OLDI_DIR as OLDI_NORM_DIR,
    DIALECT_CODES, VARIETY_CODES,
    FLORES_SLUG, OLDI_PARQUET,
    SAMPLE_SIZE, RANDOM_STATE,
    experiment_dirs,
)
from analysis._shared.run_meta import write_run_meta
from analysis.canine.core.config import (
    DEFAULT_MODEL_NAME, MAX_LENGTH, BATCH_SIZE,
)
from analysis.canine.core.embedder import (
    CanineEmbedder, aggregate_variety_vectors,
)
from analysis.canine.core.evaluate import variety_eval, parallel_eval
from analysis.canine.core.trainer import run_mlm_training


METHOD = "canine"
EXPERIMENT = "mlm_wiki_rehearsal_normalized"

DEFAULT_EPOCHS      = 3
DEFAULT_TRAIN_BATCH = 8
DEFAULT_GRAD_ACCUM  = 8
DEFAULT_LR          = 3e-5
DEFAULT_SAMPLE_SIZE = 50_000

WIKI_GROUP_A_NORM = WIKI_NORMALIZED_DIR / "dialects_in_both_OLDI_and_Flores"
WIKI_LANG_NORM    = WIKI_NORMALIZED_DIR / "languages"


def _wiki_path_normalized(code: str) -> Path:
    if code in {"fur", "lij", "lmo", "sc", "scn", "vec"}:
        return WIKI_GROUP_A_NORM / f"{code}.csv"
    return WIKI_LANG_NORM / f"{code}.csv"


def load_wiki_normalized(codes: List[str], sample_size: int, random_state: int,
                         verbose: bool = True) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    data: Dict[str, List[str]] = {}
    rows = []
    for code in codes:
        path = _wiki_path_normalized(code)
        df = pd.read_csv(path, usecols=["text"]).dropna(subset=["text"])
        n_avail = len(df)
        if n_avail <= sample_size:
            sampled = df["text"].tolist()
        else:
            sampled = df.sample(n=sample_size, random_state=random_state)["text"].tolist()
        data[code] = sampled
        rows.append({"code": code, "n_available": n_avail, "n_used": len(sampled)})
        if verbose:
            print(f"  [wiki   {code:>4}] available={n_avail:>7d}  used={len(sampled):>7d}")
    return data, pd.DataFrame(rows)


def load_flores_normalized(codes: List[str]) -> Dict[str, List[str]]:
    out = {}
    for code in codes:
        if code not in FLORES_SLUG:
            continue
        path = FLORES_NORM_DIR / f"{FLORES_SLUG[code]}.txt"
        with path.open(encoding="utf-8") as fh:
            out[code] = [line.strip() for line in fh if line.strip()]
    return out


def load_oldi_normalized(codes: List[str]) -> Dict[str, List[str]]:
    out = {}
    for code in codes:
        if code not in OLDI_PARQUET:
            continue
        path = OLDI_NORM_DIR / f"{OLDI_PARQUET[code]}.parquet"
        df = pd.read_parquet(path, columns=["text"])
        out[code] = df["text"].tolist()
    return out


def iter_labeled_sentences(data: Dict[str, List[str]], codes: List[str]):
    sents, sent_codes = [], []
    for code in codes:
        if code not in data:
            continue
        for s in data[code]:
            sents.append(s)
            sent_codes.append(code)
    return sents, sent_codes


def _save_variety_vectors(X, codes, out_dir: Path) -> None:
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
        method_label=f"CANINE adapted ({test_name.upper()} centroid, rehearsal-norm)",
    )

    per_variety = embedder.encode_per_variety(test_data, codes, batch_size=BATCH_SIZE)
    _, er_parallel = experiment_dirs(SCRIPT_DIR, f"{test_name}/parallel")
    parallel_eval(
        per_variety, out_dir=er_parallel,
        method_label=f"CANINE adapted ({test_name.upper()} parallel, rehearsal-norm)",
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
    print(f"  text variant = NORMALIZED (lowercase ASCII)")
    print(f"  train codes  = {VARIETY_CODES}  (all 13 varieties — rehearsal)")
    print(f"  eval codes   = {VARIETY_CODES}  (all 13 varieties)")
    print(f"  sample_size  = {args.sample_size}  (ablation cap)")
    print(f"  epochs       = {args.epochs}")
    print(f"  train batch  = {args.train_batch_size} × grad_acc {args.grad_accumulation}")
    print(f"  lr           = {args.lr}")
    print()

    mo_root = SCRIPT_DIR / "method_outputs"
    mo_root.mkdir(parents=True, exist_ok=True)
    model_dir = mo_root / "models" / "mlm_wiki_rehearsal_norm"

    print(f"Loading Wiki normalized (training, {len(VARIETY_CODES)} varieties — rehearsal) ...")
    wiki_data, wiki_stats = load_wiki_normalized(
        VARIETY_CODES, sample_size=args.sample_size, random_state=args.random_state,
    )
    wiki_stats["sample_size_param"] = args.sample_size
    wiki_stats["random_state"]      = args.random_state
    wiki_stats.to_csv(mo_root / "run_stats.csv", index=False)

    sents, _ = iter_labeled_sentences(wiki_data, codes=VARIETY_CODES)
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
            "text_variant":      "normalized",
            "training_codes":    VARIETY_CODES,
            "eval_codes":        VARIETY_CODES,
            "epochs":            args.epochs,
            "train_batch_size":  args.train_batch_size,
            "grad_accumulation": args.grad_accumulation,
            "lr":                args.lr,
            "max_length":        MAX_LENGTH,
        },
    )

    print("\nLoading OLDI normalized ...")
    oldi_data = load_oldi_normalized(VARIETY_CODES)
    print("Loading FLORES normalized ...")
    flores_data = load_flores_normalized(VARIETY_CODES)

    embedder = CanineEmbedder(
        model_name=str(model_dir), device=args.device, max_length=MAX_LENGTH,
    )
    evaluate_on("flores", flores_data, embedder, VARIETY_CODES)
    evaluate_on("oldi",   oldi_data,   embedder, VARIETY_CODES)

    print("\nDone.")


if __name__ == "__main__":
    main()
