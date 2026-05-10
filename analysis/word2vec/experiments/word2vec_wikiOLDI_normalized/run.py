"""
Word2Vec — train Wiki+OLDI dialect, evaluate on FLORES (NORMALIZED).

Cell 5 of the FINAL 12-cell experimental matrix.

Launch:
    python analysis/word2vec/experiments/word2vec_wikiOLDI_normalized/run.py
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
from analysis._shared.dataset_loaders import (
    load_wiki_plus_oldi_dialect, load_flores,
)
from analysis._shared.varieties import (
    VARIETY_CODES, SAMPLE_SIZE, RANDOM_STATE,
)
from analysis.word2vec.core.config import (
    W2V_VECTOR_SIZE, W2V_WINDOW, W2V_MIN_COUNT, W2V_SG, W2V_EPOCHS, W2V_SEED,
    experiment_dirs,
)
from analysis.word2vec.core.preprocess import build_tokenised_corpus
from analysis.word2vec.core.train import train_word2vec, save_word2vec
from analysis.word2vec.core.embed import (
    embed_corpus, aggregate_variety_vectors, embed_data_per_sentence,
)
from analysis.word2vec.core.evaluate import variety_eval, parallel_eval


METHOD = "word2vec"
EXPERIMENT = "word2vec_wikiOLDI_normalized"
TEXT_VARIANT = "normalized"


def _save_variety_vectors(X: np.ndarray, codes, out_dir: Path) -> None:
    np.savez_compressed(
        out_dir / "variety_vectors.npz",
        matrix=X.astype(np.float32), labels=np.asarray(codes),
    )
    pd.DataFrame(X, index=codes).to_csv(
        out_dir / "variety_vectors.csv", float_format="%.6f",
    )


def evaluate_on(test_name: str, test_data, model, codes):
    print(f"\n--- Eval on {test_name.upper()} ---")
    mo = SCRIPT_DIR / "method_outputs" / test_name
    mo.mkdir(parents=True, exist_ok=True)

    test_tok, test_sent_codes = build_tokenised_corpus(test_data, codes=codes)
    sent_vecs, sent_codes_out = embed_corpus(model, test_tok, test_sent_codes)
    X, codes_out = aggregate_variety_vectors(sent_vecs, sent_codes_out, codes=codes)
    _save_variety_vectors(X, codes_out, mo)
    _, er_centroid = experiment_dirs(SCRIPT_DIR, f"{test_name}/centroid")
    rep = variety_eval(X, codes_out, out_dir=er_centroid,
                       method_label=f"Word2Vec ({EXPERIMENT}, {test_name} centroid)")

    per_sent = embed_data_per_sentence(model, test_data, codes=codes)
    _, er_parallel = experiment_dirs(SCRIPT_DIR, f"{test_name}/parallel")
    parallel_eval(per_sent, out_dir=er_parallel,
                  method_label=f"Word2Vec ({EXPERIMENT}, {test_name} parallel)")

    print(f"  centroid: sil_fam={rep['silhouette_family']:+.4f}  "
          f"sil_rom={rep['silhouette_romance_vs_rest']:+.4f}  "
          f"sil_rom_noDial={rep.get('silhouette_romance_no_dialects')}")


def main():
    parser = argparse.ArgumentParser(description="Word2Vec wikiOLDI → FLORES (normalized)")
    parser.add_argument("--sample-size",  type=int, default=SAMPLE_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    print(f"{METHOD} — {EXPERIMENT}")
    print("=" * 60)
    print(f"  text variant = {TEXT_VARIANT}")
    print(f"  sample_size  = {args.sample_size}")
    print(f"  random_state = {args.random_state}")
    print(f"  varieties    = {VARIETY_CODES}")
    print()

    print(f"Loading Wiki + OLDI (dialect cap rule, {TEXT_VARIANT}) ...")
    train_data, train_stats = load_wiki_plus_oldi_dialect(
        text_variant=TEXT_VARIANT,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )
    mo_root = SCRIPT_DIR / "method_outputs"
    mo_root.mkdir(parents=True, exist_ok=True)
    train_stats.to_csv(mo_root / "run_stats.csv", index=False)

    print(f"\nLoading FLORES ({TEXT_VARIANT}) ...")
    flores_data, _ = load_flores(text_variant=TEXT_VARIANT, verbose=False)

    print("\n--- Tokenising train + training Word2Vec ---")
    train_tok, _ = build_tokenised_corpus(train_data)
    print(f"  {len(train_tok):,} tokenised sentences")
    model = train_word2vec(train_tok)
    save_word2vec(model, mo_root / "models" / "word2vec.model")

    write_run_meta(
        out_dir=mo_root, method=METHOD, experiment=EXPERIMENT,
        params={
            "sample_size":   args.sample_size,
            "random_state":  args.random_state,
            "text_variant":  TEXT_VARIANT,
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
