"""
End-to-end orchestrator for the multilingual XLM-R embeddings approach on
FLORES+.

Launch (from repo root, with venv active):
    python analysis/multilingual_xlmr/flores/src/run_pipeline.py

Method outputs land under ``method_outputs/`` (sentence/variety vectors,
run stats); evaluation artefacts (distances, dendrogram, projections, ...)
are produced by the central ``evaluation`` module under ``evaluation_results/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the project root importable so ``from evaluation.evaluation import ...``
# works when running this script directly.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
for p in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import (
    VARIETY_CODES, VARIETY_GROUP, VARIETY_NAMES,
    GROUP_NAMES, GROUP_COLORS,
    DEFAULT_MODEL_NAME, MAX_LENGTH, BATCH_SIZE,
    outputs_subdir, evaluation_subdir,
)
from data_loader import load_all_varieties, iter_labeled_sentences
from embedder import MultilingualEmbedder

from evaluation.evaluation import run_evaluation


ROMANCE_FAMILIES = {"italo_romance", "italian", "romance"}


def _l2_normalise(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return v / norm


def aggregate_variety_vectors(
    sentence_vectors: np.ndarray,
    sentence_codes,
    codes=VARIETY_CODES,
):
    """Per-variety vector = L2-normalised mean of L2-normalised sentence vectors."""
    codes_ordered = []
    rows = []
    arr_codes = np.asarray(sentence_codes)
    for slug in codes:
        mask = (arr_codes == slug)
        if mask.sum() == 0:
            continue
        rows.append(sentence_vectors[mask].mean(axis=0))
        codes_ordered.append(slug)
    X = np.vstack(rows).astype(np.float32)
    X = _l2_normalise(X)
    return X, codes_ordered


def save_sentence_vectors(sentence_vectors, sentence_codes, model_name) -> str:
    out = outputs_subdir() / "sentence_vectors.npz"
    np.savez_compressed(
        out,
        vectors=sentence_vectors.astype(np.float32),
        codes=np.asarray(sentence_codes),
        model_name=np.asarray(model_name),
    )
    return str(out)


def save_variety_vectors(X, codes, model_name) -> dict:
    out_csv = outputs_subdir() / "variety_vectors.csv"
    pd.DataFrame(X, index=codes).to_csv(out_csv, float_format="%.6f")
    out_npz = outputs_subdir() / "variety_vectors.npz"
    np.savez_compressed(
        out_npz,
        matrix=X.astype(np.float32),
        labels=np.asarray(codes),
        model_name=np.asarray(model_name),
    )
    return {"csv": str(out_csv), "npz": str(out_npz)}


def main():
    parser = argparse.ArgumentParser(
        description="Multilingual XLM-R embeddings orchestrator (FLORES+)"
    )
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    from config import SAMPLE_SIZE, RANDOM_STATE
    sample_size = args.sample_size or SAMPLE_SIZE
    random_state = args.random_state if args.random_state is not None else RANDOM_STATE

    print("Multilingual embeddings on FLORES+")
    print("=" * 60)
    print(f"  model_name    = {args.model_name}")
    print(f"  sample_size   = {sample_size}")
    print(f"  random_state  = {random_state}")
    print(f"  batch_size    = {args.batch_size}")
    print(f"  max_length    = {MAX_LENGTH}")
    print(f"  varieties     = {VARIETY_CODES}")
    print()

    print("Loading + sampling...")
    data, stats = load_all_varieties(
        sample_size=sample_size,
        random_state=random_state,
    )
    stats["sample_size_param"] = sample_size
    stats["random_state"] = random_state
    stats["model_name"] = args.model_name
    stats.to_csv(outputs_subdir() / "run_stats.csv", index=False)

    sents, sent_codes = iter_labeled_sentences(data)
    print(f"  {len(sents)} sentences to embed")

    print("\n--- Loading model ---")
    embedder = MultilingualEmbedder(
        model_name=args.model_name,
        device=args.device,
        max_length=MAX_LENGTH,
    )

    print("\n--- Computing sentence embeddings ---")
    sent_vecs = embedder.encode(sents, batch_size=args.batch_size)
    print(f"  sentence_vectors: shape={sent_vecs.shape}")

    print("\n--- Aggregating per variety ---")
    X, codes = aggregate_variety_vectors(sent_vecs, sent_codes)
    print(f"  variety_vectors : shape={X.shape}, codes={codes}")

    sv_path = save_sentence_vectors(sent_vecs, sent_codes, args.model_name)
    vv_paths = save_variety_vectors(X, codes, args.model_name)
    print(f"  Saved: {sv_path}")
    print(f"  Saved: {vv_paths['csv']}")
    print(f"  Saved: {vv_paths['npz']}")

    print("\n--- Central evaluation ---")
    report = run_evaluation(
        variety_vectors=X,
        variety_codes=codes,
        out_dir=evaluation_subdir(),
        method_label=f"XLM-R ({args.model_name})",
        family_groups=VARIETY_GROUP,
        family_colors=GROUP_COLORS,
        family_display_names=GROUP_NAMES,
        display_names=VARIETY_NAMES,
        romance_families=ROMANCE_FAMILIES,
    )

    print("\n" + "=" * 60)
    print("Done. Summary:")
    sf = report["silhouette_family"]
    sr = report["silhouette_romance_vs_rest"]
    print(f"  multilingual_xlmr ({args.model_name}): X.shape={X.shape}  "
          f"sil_family={sf:+.4f}  sil_romance={sr:+.4f}")
    print(f"\nMethod outputs:       {outputs_subdir()}")
    print(f"Evaluation artefacts: {evaluation_subdir()}")


if __name__ == "__main__":
    main()
