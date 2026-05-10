"""
TF-IDF (sentence-level) — train Wiki + OLDI dialect, evaluate on FLORES (NATIVE).

Cell 2 of the FINAL 12-cell experimental matrix. Same setup as the normalized
counterpart but with native (cased, accented, punctuated) text — keeps
diacritics/case/punct as distinguishing features.

Launch:
    python analysis/tfidf/experiments/tfidf_wikiOLDI_native/run.py
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
from analysis.tfidf.core.evaluate import variety_eval

from sklearn.feature_extraction.text import TfidfVectorizer


METHOD = "tfidf"
EXPERIMENT = "tfidf_wikiOLDI_native"
TEXT_VARIANT = "native"


# Native pipeline: keep case, diacritics, punctuation as distinguishing features.
# Use raw text minus collapse-whitespace (do NOT lowercase, do NOT strip punct).
import re
_WS_RE = re.compile(r"\s+")
def _normalize_native(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return _WS_RE.sub(" ", s).strip()


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


def _flatten_sentences(data: dict, codes) -> list:
    out = []
    for code in codes:
        if code not in data:
            continue
        for sent in data[code]:
            s = _normalize_native(sent)
            if s:
                out.append(s)
    return out


def _centroid_per_variety(vectorizer, flores_data, codes):
    vectors, kept = [], []
    for code in codes:
        if code not in flores_data:
            continue
        sents = [_normalize_native(s) for s in flores_data[code]]
        sents = [s for s in sents if s]
        if not sents:
            continue
        Xs = vectorizer.transform(sents)
        centroid = np.asarray(Xs.mean(axis=0)).ravel()
        vectors.append(centroid)
        kept.append(code)
    return np.vstack(vectors).astype(np.float32), kept


def _save_variety_vectors(X, codes, out_dir):
    np.savez_compressed(
        out_dir / "variety_vectors.npz",
        matrix=X.astype(np.float32), labels=np.asarray(codes),
    )
    pd.DataFrame(X, index=codes).to_csv(
        out_dir / "variety_vectors.csv", float_format="%.6f",
    )


def run_pipeline(variant: str, vec_factory, train_data, flores_data):
    print(f"\n=== Pipeline: {variant} ===")
    mo, _  = experiment_dirs(SCRIPT_DIR, variant)
    _, er = experiment_dirs(SCRIPT_DIR, f"flores/{variant}")

    train_sents = _flatten_sentences(train_data, VARIETY_CODES)
    print(f"  train sentences: {len(train_sents):,}")

    vec = vec_factory()
    vec.fit(train_sents)
    print(f"  vocab size:      {len(vec.vocabulary_):,}")

    X, codes = _centroid_per_variety(vec, flores_data, VARIETY_CODES)
    print(f"  centroid matrix: {X.shape}")
    _save_variety_vectors(X, codes, mo)

    report = variety_eval(X, codes, out_dir=er,
                          method_label=f"TF-IDF {variant} ({EXPERIMENT})")
    return {
        "variant": variant,
        "tfidf_shape": tuple(X.shape),
        "silhouette_family":          report["silhouette_family"],
        "silhouette_romance_vs_rest": report["silhouette_romance_vs_rest"],
        "silhouette_romance_no_dialects": report.get("silhouette_romance_no_dialects"),
    }


def main():
    parser = argparse.ArgumentParser(description="TF-IDF wikiOLDI → FLORES (native)")
    parser.add_argument("--sample-size",  type=int, default=SAMPLE_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--pipeline",     choices=["word", "char", "both"], default="both")
    args = parser.parse_args()

    print(f"{METHOD} — {EXPERIMENT}")
    print("=" * 60)
    print(f"  text variant   = {TEXT_VARIANT}")
    print(f"  pipeline       = {args.pipeline}")
    print(f"  sample_size    = {args.sample_size}")
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
        out_dir=mo_root, method=METHOD, experiment=EXPERIMENT,
        params={
            "sample_size":     args.sample_size,
            "random_state":    args.random_state,
            "text_variant":    TEXT_VARIANT,
            "pipeline":        args.pipeline,
            "word_ngram_range": list(WORD_NGRAM_RANGE),
            "char_ngram_range": list(CHAR_NGRAM_RANGE),
            "char_analyzer":   CHAR_ANALYZER,
            "sublinear_tf":    SUBLINEAR_TF,
            "norm":            NORM,
        },
    )

    reports = []
    if args.pipeline in ("word", "both"):
        reports.append(run_pipeline("word", _build_word_vectorizer, train_data, flores_data))
    if args.pipeline in ("char", "both"):
        reports.append(run_pipeline("char", _build_char_vectorizer, train_data, flores_data))

    print("\n" + "=" * 60)
    for r in reports:
        sf = r["silhouette_family"]
        sr = r["silhouette_romance_vs_rest"]
        sn = r["silhouette_romance_no_dialects"]
        print(f"  {r['variant']:>5}: shape={r['tfidf_shape']}  "
              f"sil_fam={sf:+.4f}  sil_rom={sr:+.4f}  sil_rom_noDial={sn if sn is None else f'{sn:+.4f}'}")


if __name__ == "__main__":
    main()
