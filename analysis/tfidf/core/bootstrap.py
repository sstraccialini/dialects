"""
Bootstrap CI on the Spearman gold correlations for a TF-IDF experiment.

TF-IDF differs from the other families: the trained ``TfidfVectorizer`` is
NOT persisted to disk (sklearn fit is deterministic — same training data +
hyperparams ↦ bit-for-bit identical vectorizer).  So we *re-fit* the
vectorizer from Wiki+OLDI inside this script.  That is NOT stochastic
retraining; it is a deterministic recomputation that produces the exact
same vectorizer the original run.py used.

Sparse-aware bootstrap: vocab can exceed 10^5, saving per-sentence
embeddings dense would take ≈12 GB per experiment.  We keep the
per-variety sentence vectors as scipy.sparse, resample row indices, take
the sparse mean per variety, densify only the 17×D centroid matrix
(≈ a few MB), then L2-normalise + cosine + Spearman as elsewhere.

One run produces both the char and the word pipeline by default; pass
``--pipeline word`` or ``--pipeline char`` to limit.  Each variant writes
to its own evaluation_results path:
    analysis/tfidf/experiments/<exp>/evaluation_results/flores/{char,word}/bootstrap_results.csv

CLI:
    python -m analysis.tfidf.core.bootstrap \\
        --experiment tfidf_wikiOLDI_native --n-boot 1000
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis._shared.dataset_loaders import (
    load_wiki_plus_oldi_dialect, load_flores,
)
from analysis._shared.varieties import VARIETY_CODES, SAMPLE_SIZE, RANDOM_STATE
from analysis.tfidf.core.config import (
    WORD_NGRAM_RANGE, WORD_MIN_DF, WORD_MAX_DF, WORD_MAX_FEATURES,
    CHAR_NGRAM_RANGE, CHAR_ANALYZER, CHAR_MIN_DF, CHAR_MAX_DF, CHAR_MAX_FEATURES,
    SUBLINEAR_TF, NORM,
)
from evaluation._bootstrap_core import (
    _cosine_distance_matrix, _l2_normalise,
    _spearman_pair_from_dist, default_gold_paths,
)
from evaluation._gold_correlation import default_roles, load_gold


EXPERIMENTS = {
    "tfidf_wikiOLDI_native":     "native",
    "tfidf_wikiOLDI_normalized": "normalized",
}


# --------------------------------------------------------------------------- #
# Vectorizer factories — must match analysis/tfidf/experiments/<exp>/run.py
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Pipeline helpers
# --------------------------------------------------------------------------- #

def _experiment_dir(experiment: str) -> Path:
    return REPO_ROOT / "analysis" / "tfidf" / "experiments" / experiment


def _output_csv(experiment: str, variant: str) -> Path:
    return (_experiment_dir(experiment)
            / "evaluation_results" / "flores" / variant
            / "bootstrap_results.csv")


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


def _per_variety_sentences(data: dict, codes) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for code in codes:
        if code not in data:
            continue
        cleaned = [_normalize_native(s) for s in data[code]]
        cleaned = [s for s in cleaned if s]
        if cleaned:
            out[code] = cleaned
    return out


def _bootstrap_sparse(
    per_variety: Dict[str, sp.csr_matrix],
    variety_codes: List[str],
    gold_paths,
    *,
    n_boot: int, alpha: float, rng: np.random.Generator,
    dialect_codes, external_codes,
) -> pd.DataFrame:
    """Sparse-aware variant of bootstrap_from_sentence_vectors."""
    codes_present = [c for c in variety_codes if c in per_variety]
    golds = [(p.stem, *load_gold(p)[:2]) for p in gold_paths]

    # Observed centroids (no resample).
    obs_rows = []
    for c in codes_present:
        m = per_variety[c].mean(axis=0)               # 1×D dense matrix
        obs_rows.append(np.asarray(m).ravel())
    obs_cent = _l2_normalise(np.vstack(obs_rows).astype(np.float32))
    obs_dist = _cosine_distance_matrix(obs_cent)

    observed = {}
    for name, mat, labels in golds:
        observed[name] = _spearman_pair_from_dist(
            obs_dist, codes_present, mat, labels,
            dialect_codes, external_codes,
        )

    samples = {n: [] for n, _, _ in golds}
    for _ in range(n_boot):
        rows = []
        for c in codes_present:
            X_c = per_variety[c]
            n_var = X_c.shape[0]
            idx = rng.integers(0, n_var, size=n_var)
            m = X_c[idx].mean(axis=0)
            rows.append(np.asarray(m).ravel())
        cent = _l2_normalise(np.vstack(rows).astype(np.float32))
        dist = _cosine_distance_matrix(cent)
        for name, mat, labels in golds:
            samples[name].append(
                _spearman_pair_from_dist(dist, codes_present, mat, labels,
                                         dialect_codes, external_codes)
            )

    lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0
    rows_out = []
    for name in samples:
        arr = np.asarray(samples[name], dtype=np.float64)
        for j, block in enumerate(("full", "dialect_external")):
            col = arr[:, j]
            col = col[np.isfinite(col)]
            rho_obs = observed[name][j]
            if col.size:
                rows_out.append({
                    "gold": name, "block": block,
                    "rho_observed": rho_obs,
                    "rho_mean": float(col.mean()),
                    "rho_lo":   float(np.quantile(col, lo_q)),
                    "rho_hi":   float(np.quantile(col, hi_q)),
                    "n_boot":   int(col.size),
                })
            else:
                rows_out.append({
                    "gold": name, "block": block,
                    "rho_observed": rho_obs,
                    "rho_mean": float("nan"),
                    "rho_lo":   float("nan"), "rho_hi": float("nan"),
                    "n_boot":   0,
                })
    return pd.DataFrame(rows_out)


def _run_one_pipeline(
    variant: str, factory: Callable[[], TfidfVectorizer],
    train_sents: List[str], flores_per_variety: Dict[str, List[str]],
    experiment: str, n_boot: int, seed: int,
) -> Path:
    print(f"\n--- pipeline: {variant} ---")
    vec = factory()
    vec.fit(train_sents)
    print(f"  vocab size: {len(vec.vocabulary_):,}")

    per_variety_sparse: Dict[str, sp.csr_matrix] = {}
    for code, sents in flores_per_variety.items():
        Xs = vec.transform(sents)
        per_variety_sparse[code] = Xs.tocsr()
    n_sent = sum(m.shape[0] for m in per_variety_sparse.values())
    D = next(iter(per_variety_sparse.values())).shape[1]
    print(f"  transformed: {n_sent:,} sentences (D={D})")

    rng = np.random.default_rng(seed)
    dialect_codes, external_codes = default_roles()
    df = _bootstrap_sparse(
        per_variety_sparse, VARIETY_CODES,
        default_gold_paths(REPO_ROOT),
        n_boot=n_boot, alpha=0.05, rng=rng,
        dialect_codes=dialect_codes, external_codes=external_codes,
    )
    out = _output_csv(experiment, variant)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.4f")
    print(df.to_string(index=False))
    print(f"  → {out.relative_to(REPO_ROOT)}")
    return out


def run(experiment: str, pipeline: str, n_boot: int, seed: int,
        sample_size: int, random_state: int) -> List[Path]:
    if experiment not in EXPERIMENTS:
        raise SystemExit(f"Unknown experiment {experiment!r}. "
                         f"Choose from {list(EXPERIMENTS)}.")
    text_variant = EXPERIMENTS[experiment]
    print(f"=== {experiment}  (text_variant={text_variant}) ===")

    print(f"  loading Wiki + OLDI ({text_variant}) ...")
    train_data, _ = load_wiki_plus_oldi_dialect(
        text_variant=text_variant,
        sample_size=sample_size, random_state=random_state,
    )
    train_sents = _flatten_sentences(train_data, VARIETY_CODES)
    print(f"  training sentences: {len(train_sents):,}")

    print(f"  loading FLORES ({text_variant}) ...")
    flores_data, _ = load_flores(text_variant=text_variant, verbose=False)
    flores_pv = _per_variety_sentences(flores_data, VARIETY_CODES)

    outs: List[Path] = []
    if pipeline in ("word", "both"):
        outs.append(_run_one_pipeline(
            "word", _build_word_vectorizer, train_sents, flores_pv,
            experiment, n_boot, seed,
        ))
    if pipeline in ("char", "both"):
        outs.append(_run_one_pipeline(
            "char", _build_char_vectorizer, train_sents, flores_pv,
            experiment, n_boot, seed,
        ))
    return outs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    ap.add_argument("--pipeline",   choices=["word", "char", "both"], default="both")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--sample-size",  type=int, default=SAMPLE_SIZE)
    ap.add_argument("--random-state", type=int, default=RANDOM_STATE)
    args = ap.parse_args(argv)
    run(args.experiment, args.pipeline, args.n_boot, args.seed,
        args.sample_size, args.random_state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
