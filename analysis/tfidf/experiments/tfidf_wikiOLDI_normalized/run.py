"""
TF-IDF (sentence-level) — train Wiki + OLDI dialect, evaluate on FLORES (NORMALIZED).

Cell 1 of the FINAL 12-cell experimental matrix (EXPERIMENTAL_PLAN.md §3.1).

Approach (b):
  fit       : sentence-level TF-IDF on Wiki + OLDI dialect sentences
              (cap rule §3.4: for each dialect take ALL OLDI first, then Wiki up to 100k)
  transform : 1827 parallel FLORES sentences per variety
  centroid  : mean of FLORES sentence vectors per variety (17 × V matrix)
  eval      : silhouette / dendrogram / projections via run_evaluation
              (isotropy=False, dialect_families injected — see core/evaluate.py)

Two sub-pipelines: word and char_wb (3-5).

Launch:
    python analysis/tfidf/experiments/tfidf_wikiOLDI_normalized/run.py
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
from analysis.tfidf.core.config import (
    WORD_NGRAM_RANGE, WORD_MIN_DF, WORD_MAX_DF, WORD_MAX_FEATURES,
    CHAR_NGRAM_RANGE, CHAR_ANALYZER, CHAR_MIN_DF, CHAR_MAX_DF, CHAR_MAX_FEATURES,
    SUBLINEAR_TF, NORM,
    experiment_dirs,
)
from analysis.tfidf.core.preprocess import preprocess_for_word, preprocess_for_char
from analysis.tfidf.core.evaluate import variety_eval

from sklearn.feature_extraction.text import TfidfVectorizer


METHOD = "tfidf"
EXPERIMENT = "tfidf_wikiOLDI_normalized"
TEXT_VARIANT = "normalized"


def _build_word_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=WORD_NGRAM_RANGE,
        min_df=WORD_MIN_DF, max_df=WORD_MAX_DF, max_features=WORD_MAX_FEATURES,
        sublinear_tf=SUBLINEAR_TF, norm=NORM,
        lowercase=False, stop_words=None, strip_accents=None,
    )


def _build_char_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=CHAR_ANALYZER,
        ngram_range=CHAR_NGRAM_RANGE,
        min_df=CHAR_MIN_DF, max_df=CHAR_MAX_DF, max_features=CHAR_MAX_FEATURES,
        sublinear_tf=SUBLINEAR_TF, norm=NORM,
        lowercase=False, stop_words=None, strip_accents=None,
    )


def _flatten_sentences(data: dict, preprocess_fn, codes) -> list:
    """Concatenate all sentences across varieties (no super-doc, just the flat list).

    TF-IDF is fitted at sentence level so the IDF statistics are computed
    over hundreds of thousands of independent documents instead of 17.
    """
    out = []
    for code in codes:
        if code not in data:
            continue
        for sent in data[code]:
            s = preprocess_fn(sent)
            if s:
                out.append(s)
    return out


def _centroid_per_variety(
    vectorizer: TfidfVectorizer,
    flores_data: dict,
    preprocess_fn,
    codes,
):
    """Transform FLORES sentences per variety and average → variety centroid."""
    vectors, kept = [], []
    for code in codes:
        if code not in flores_data:
            continue
        sents = [preprocess_fn(s) for s in flores_data[code]]
        sents = [s for s in sents if s]
        if not sents:
            continue
        Xs = vectorizer.transform(sents)             # (n_sents, V), sparse
        centroid = np.asarray(Xs.mean(axis=0)).ravel()
        vectors.append(centroid)
        kept.append(code)
    return np.vstack(vectors).astype(np.float32), kept


def _save_variety_vectors(X: np.ndarray, codes, out_dir: Path) -> None:
    np.savez_compressed(
        out_dir / "variety_vectors.npz",
        matrix=X.astype(np.float32), labels=np.asarray(codes),
    )
    pd.DataFrame(X, index=codes).to_csv(
        out_dir / "variety_vectors.csv", float_format="%.6f",
    )


def run_pipeline(variant: str, vec_factory, preprocess_fn, train_data, flores_data):
    print(f"\n=== Pipeline: {variant} ===")
    mo, _  = experiment_dirs(SCRIPT_DIR, variant)
    _, er = experiment_dirs(SCRIPT_DIR, f"flores/{variant}")

    train_sents = _flatten_sentences(train_data, preprocess_fn, VARIETY_CODES)
    print(f"  train sentences: {len(train_sents):,}")

    vec = vec_factory()
    vec.fit(train_sents)
    print(f"  vocab size:      {len(vec.vocabulary_):,}")

    X, codes = _centroid_per_variety(vec, flores_data, preprocess_fn, VARIETY_CODES)
    print(f"  centroid matrix: {X.shape}")
    _save_variety_vectors(X, codes, mo)

    report = variety_eval(
        X, codes, out_dir=er,
        method_label=f"TF-IDF {variant} ({EXPERIMENT})",
    )
    return {
        "variant": variant,
        "tfidf_shape": tuple(X.shape),
        "silhouette_family":          report["silhouette_family"],
        "silhouette_romance_vs_rest": report["silhouette_romance_vs_rest"],
        "silhouette_romance_no_dialects": report.get("silhouette_romance_no_dialects"),
    }


def main():
    parser = argparse.ArgumentParser(description="TF-IDF wikiOLDI → FLORES (normalized)")
    parser.add_argument("--sample-size",  type=int, default=SAMPLE_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--pipeline",     choices=["word", "char", "both"], default="both")
    args = parser.parse_args()

    print(f"{METHOD} — {EXPERIMENT}")
    print("=" * 60)
    print(f"  text variant   = {TEXT_VARIANT}")
    print(f"  pipeline       = {args.pipeline}")
    print(f"  sample_size    = {args.sample_size}  (cap per variety)")
    print(f"  random_state   = {args.random_state}")
    print(f"  varieties      = {VARIETY_CODES}")
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

    write_run_meta(
        out_dir=mo_root,
        method=METHOD,
        experiment=EXPERIMENT,
        params={
            "sample_size":       args.sample_size,
            "random_state":      args.random_state,
            "text_variant":      TEXT_VARIANT,
            "pipeline":          args.pipeline,
            "word_ngram_range":  list(WORD_NGRAM_RANGE),
            "char_ngram_range":  list(CHAR_NGRAM_RANGE),
            "char_analyzer":     CHAR_ANALYZER,
            "sublinear_tf":      SUBLINEAR_TF,
            "norm":              NORM,
        },
    )

    reports = []
    if args.pipeline in ("word", "both"):
        reports.append(run_pipeline("word", _build_word_vectorizer, preprocess_for_word, train_data, flores_data))
    if args.pipeline in ("char", "both"):
        reports.append(run_pipeline("char", _build_char_vectorizer, preprocess_for_char, train_data, flores_data))

    print("\n" + "=" * 60)
    for r in reports:
        sf = r["silhouette_family"]
        sr = r["silhouette_romance_vs_rest"]
        sn = r["silhouette_romance_no_dialects"]
        print(f"  {r['variant']:>5}: shape={r['tfidf_shape']}  "
              f"sil_fam={sf:+.4f}  sil_rom={sr:+.4f}  sil_rom_noDial={sn if sn is None else f'{sn:+.4f}'}")


if __name__ == "__main__":
    main()
