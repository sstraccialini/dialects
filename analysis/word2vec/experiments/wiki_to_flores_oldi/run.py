"""
Word2Vec: train on Wikipedia, evaluate on FLORES+ and OLDI separately.

A single shared skip-gram Word2Vec is fitted on the tokenised Wiki
corpus (all 13 varieties combined). The resulting word vectors are
then applied to FLORES+ and OLDI to embed each parallel sentence; per
test corpus we compute both:

  - centroid eval     (run_evaluation): 13×13 cosine matrix between
                      L2-normalised mean-pooled variety centroids.
  - parallel eval     (run_parallel_alignment): mean cosine over
                      sentence-aligned pairs across varieties.

Output layout:

  method_outputs/
  ├── models/word2vec.model          gensim Word2Vec
  ├── flores/variety_vectors.{npz,csv}
  ├── oldi/variety_vectors.{npz,csv}
  ├── run_stats.csv
  └── run_meta.json
  evaluation_results/
  ├── flores/centroid/...            run_evaluation outputs
  ├── flores/parallel/...            run_parallel_alignment outputs
  ├── oldi/centroid/...
  └── oldi/parallel/...

Launch:
    python analysis/word2vec/experiments/wiki_to_flores_oldi/run.py
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
from analysis.word2vec.core.config import (
    VARIETY_CODES, SAMPLE_SIZE, RANDOM_STATE,
    W2V_VECTOR_SIZE, W2V_WINDOW, W2V_MIN_COUNT, W2V_SG, W2V_EPOCHS, W2V_SEED,
    experiment_dirs,
)
from analysis.word2vec.core.data_loader import (
    load_wiki_for_training, load_flores_parallel, load_oldi_parallel,
)
from analysis.word2vec.core.preprocess import build_tokenised_corpus
from analysis.word2vec.core.train import train_word2vec, save_word2vec
from analysis.word2vec.core.embed import (
    embed_corpus, aggregate_variety_vectors, embed_data_per_sentence,
)
from analysis.word2vec.core.evaluate import variety_eval, parallel_eval


METHOD = "word2vec"
EXPERIMENT = "wiki_to_flores_oldi"


def _save_variety_vectors(X: np.ndarray, codes, out_dir: Path) -> None:
    np.savez_compressed(
        out_dir / "variety_vectors.npz",
        matrix=X.astype(np.float32), labels=np.asarray(codes),
    )
    pd.DataFrame(X, index=codes).to_csv(
        out_dir / "variety_vectors.csv", float_format="%.6f",
    )


def evaluate_on(test_name: str, test_data, model, codes):
    """Run centroid + parallel eval on one parallel test corpus."""
    print(f"\n--- Eval on {test_name.upper()} ---")
    mo_root = SCRIPT_DIR / "method_outputs" / test_name
    mo_root.mkdir(parents=True, exist_ok=True)

    # Centroid eval: tokenise + embed sentence-by-sentence + mean-pool per variety.
    test_tokenised, test_sent_codes = build_tokenised_corpus(test_data, codes=codes)
    print(f"  tokenised sentences: {len(test_tokenised):,}")
    sent_vecs, sent_codes_out = embed_corpus(model, test_tokenised, test_sent_codes)
    X, codes_out = aggregate_variety_vectors(sent_vecs, sent_codes_out, codes=codes)

    _save_variety_vectors(X, codes_out, mo_root)
    _, er_centroid = experiment_dirs(SCRIPT_DIR, f"{test_name}/centroid")
    centroid_report = variety_eval(
        X, codes_out, out_dir=er_centroid,
        method_label=f"Word2Vec ({test_name.upper()} centroid)",
    )

    # Parallel eval: per-sentence vectors for every variety, kept aligned.
    per_sent = embed_data_per_sentence(model, test_data, codes=codes)
    _, er_parallel = experiment_dirs(SCRIPT_DIR, f"{test_name}/parallel")
    parallel_report = parallel_eval(
        per_sent, out_dir=er_parallel,
        method_label=f"Word2Vec ({test_name.upper()} parallel)",
    )

    print(f"  centroid: sil_family={centroid_report['silhouette_family']:+.4f}  "
          f"sil_romance={centroid_report['silhouette_romance_vs_rest']:+.4f}")
    return centroid_report, parallel_report


def main():
    parser = argparse.ArgumentParser(description="Word2Vec: train Wiki, eval FLORES + OLDI")
    parser.add_argument("--sample-size",  type=int, default=SAMPLE_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    print(f"{METHOD} — {EXPERIMENT}")
    print("=" * 60)
    print(f"  sample_size  = {args.sample_size}")
    print(f"  random_state = {args.random_state}")
    print(f"  varieties    = {VARIETY_CODES}")
    print()

    print("Loading Wiki (training) ...")
    wiki_data, wiki_stats = load_wiki_for_training(
        sample_size=args.sample_size, random_state=args.random_state,
    )
    print("\nLoading FLORES (parallel test) ...")
    flores_data, _ = load_flores_parallel(verbose=False)
    print(f"  {len(flores_data)} varieties × {len(next(iter(flores_data.values()))):,} sentences")
    # Test ONLY on FLORES (the 7 new standards have no OLDI counterparts).

    mo_root = SCRIPT_DIR / "method_outputs"
    mo_root.mkdir(parents=True, exist_ok=True)
    wiki_stats["sample_size_param"] = args.sample_size
    wiki_stats["random_state"]      = args.random_state
    wiki_stats.to_csv(mo_root / "run_stats.csv", index=False)

    print("\n--- Tokenising Wiki + training Word2Vec ---")
    wiki_tokenised, wiki_sentence_codes = build_tokenised_corpus(wiki_data)
    print(f"  {len(wiki_tokenised):,} tokenised sentences")
    model = train_word2vec(wiki_tokenised)
    save_word2vec(model, mo_root / "models" / "word2vec.model")

    write_run_meta(
        out_dir=mo_root,
        method=METHOD,
        experiment=EXPERIMENT,
        params={
            "sample_size":   args.sample_size,
            "random_state":  args.random_state,
            "vector_size":   W2V_VECTOR_SIZE,
            "window":        W2V_WINDOW,
            "min_count":     W2V_MIN_COUNT,
            "sg":            W2V_SG,
            "epochs":        W2V_EPOCHS,
            "seed":          W2V_SEED,
        },
    )

    evaluate_on("flores", flores_data, model, VARIETY_CODES)

    print("\nDone.")


if __name__ == "__main__":
    main()
