"""
Bootstrap CI on the Spearman gold correlations for a word2vec experiment.

Loads the already-trained word2vec model from
``method_outputs/models/word2vec.model`` (errors out if missing — does NOT
retrain), embeds every FLORES sentence, and runs B resamples via the
shared math in ``evaluation._bootstrap_core``.

Output:
    analysis/word2vec/experiments/<exp>/evaluation_results/flores/centroid/bootstrap_results.csv

CLI:
    python -m analysis.word2vec.core.bootstrap \\
        --experiment word2vec_wikiOLDI_native --n-boot 1000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis._shared.dataset_loaders import load_flores
from analysis._shared.varieties import VARIETY_CODES
from analysis.word2vec.core.preprocess import build_tokenised_corpus
from analysis.word2vec.core.train import load_word2vec
from analysis.word2vec.core.embed import embed_corpus
from evaluation._bootstrap_core import (
    bootstrap_from_sentence_vectors, default_gold_paths,
)


EXPERIMENTS = {
    "word2vec_wikiOLDI_native":     "native",
    "word2vec_wikiOLDI_normalized": "normalized",
}


def _experiment_dir(experiment: str) -> Path:
    return REPO_ROOT / "analysis" / "word2vec" / "experiments" / experiment


def _model_path(experiment: str) -> Path:
    return _experiment_dir(experiment) / "method_outputs" / "models" / "word2vec.model"


def _output_csv(experiment: str) -> Path:
    return (_experiment_dir(experiment)
            / "evaluation_results" / "flores" / "centroid"
            / "bootstrap_results.csv")


def run(experiment: str, n_boot: int, seed: int) -> Path:
    if experiment not in EXPERIMENTS:
        raise SystemExit(f"Unknown experiment {experiment!r}. "
                         f"Choose from {list(EXPERIMENTS)}.")
    text_variant = EXPERIMENTS[experiment]
    mp = _model_path(experiment)
    if not mp.exists():
        raise SystemExit(
            f"Trained model not found at {mp.relative_to(REPO_ROOT)}.\n"
            f"Run analysis/word2vec/experiments/{experiment}/run.py first "
            f"to produce it — this bootstrap script never retrains."
        )

    print(f"=== {experiment}  (text_variant={text_variant}) ===")
    print(f"  load model: {mp.relative_to(REPO_ROOT)}")
    model = load_word2vec(mp)

    print(f"  loading FLORES ({text_variant}) ...")
    flores_data, _ = load_flores(text_variant=text_variant, verbose=False)

    test_tok, test_sent_codes = build_tokenised_corpus(
        flores_data, codes=VARIETY_CODES,
    )
    sent_vecs, sent_codes = embed_corpus(model, test_tok, test_sent_codes)
    print(f"  embedded {sent_vecs.shape[0]:,} sentences (dim={sent_vecs.shape[1]})")

    df = bootstrap_from_sentence_vectors(
        sent_vecs, sent_codes, VARIETY_CODES,
        default_gold_paths(REPO_ROOT),
        n_boot=n_boot, seed=seed,
    )
    out = _output_csv(experiment)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.4f")
    print(df.to_string(index=False))
    print(f"\n  → {out.relative_to(REPO_ROOT)}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed",   type=int, default=42)
    args = ap.parse_args(argv)
    run(args.experiment, args.n_boot, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
