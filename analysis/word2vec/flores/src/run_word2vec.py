"""
End-to-end orchestrator for the Word2Vec approach on FLORES+.

Launch (from repo root, with venv active):
    python analysis/word2vec/flores/src/run_word2vec.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
for p in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import (
    VARIETY_CODES, VARIETY_GROUP, VARIETY_NAMES,
    GROUP_NAMES, GROUP_COLORS,
    outputs_subdir, evaluation_subdir,
)
from data_loader import load_all_varieties
from train import build_tokenised_corpus, train_word2vec, save_word2vec
from build_vectors import (
    embed_corpus, aggregate_variety_vectors,
    save_sentence_vectors, save_variety_vectors,
)

from evaluation.evaluation import run_evaluation


ROMANCE_FAMILIES = {"italo_romance", "italian", "romance"}


def main():
    parser = argparse.ArgumentParser(description="Word2Vec approach orchestrator (FLORES+)")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    args = parser.parse_args()

    from config import SAMPLE_SIZE, RANDOM_STATE
    sample_size = args.sample_size or SAMPLE_SIZE
    random_state = args.random_state if args.random_state is not None else RANDOM_STATE

    print("Word2Vec approach on FLORES+")
    print("=" * 60)
    print(f"  sample_size   = {sample_size}")
    print(f"  random_state  = {random_state}")
    print(f"  varieties     = {VARIETY_CODES}")
    print()

    print("Loading + sampling...")
    data, stats = load_all_varieties(
        sample_size=sample_size,
        random_state=random_state,
    )
    stats["sample_size_param"] = sample_size
    stats["random_state"] = random_state
    stats.to_csv(outputs_subdir() / "run_stats.csv", index=False)

    print("\n--- Building tokenised corpus ---")
    tokenised, sentence_codes = build_tokenised_corpus(data)
    total_tokens = sum(len(s) for s in tokenised)
    print(f"  {len(tokenised)} sentences, {total_tokens:,} tokens")

    print("\n--- Training Word2Vec (shared, skip-gram) ---")
    model = train_word2vec(tokenised)
    model_path = save_word2vec(model)
    print(f"  Saved: {model_path}")

    print("\n--- Embedding sentences + aggregating per variety ---")
    sent_vecs, sent_codes_out = embed_corpus(model, tokenised, sentence_codes)
    print(f"  sentence_vectors: shape={sent_vecs.shape}")

    X, codes = aggregate_variety_vectors(sent_vecs, sent_codes_out)
    print(f"  variety_vectors : shape={X.shape}, codes={codes}")

    sv_path = save_sentence_vectors(sent_vecs, sent_codes_out)
    vv_paths = save_variety_vectors(X, codes)
    print(f"  Saved: {sv_path}")
    print(f"  Saved: {vv_paths['csv']}")
    print(f"  Saved: {vv_paths['npz']}")

    print("\n--- Central evaluation ---")
    report = run_evaluation(
        variety_vectors=X,
        variety_codes=codes,
        out_dir=evaluation_subdir(),
        method_label="Word2Vec",
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
    print(f"  word2vec: X.shape={X.shape}  "
          f"sil_family={sf:+.4f}  sil_romance={sr:+.4f}")
    print(f"\nMethod outputs:       {outputs_subdir()}")
    print(f"Evaluation artefacts: {evaluation_subdir()}")


if __name__ == "__main__":
    main()
