"""
Bootstrap CI on the Spearman gold correlations for a fastText experiment.

Loads the already-trained fastText model from
``method_outputs/models/fasttext_model.bin`` (errors out if missing — does
NOT retrain), embeds every FLORES sentence, and runs B resamples via the
shared math in ``evaluation._bootstrap_core``.

Output:
    analysis/fasttext/experiments/<exp>/evaluation_results/flores/centroid/bootstrap_results.csv

CLI:
    python -m analysis.fasttext.core.bootstrap \\
        --experiment fasttext_wikiOLDI_native --n-boot 1000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gensim.models import FastText

from analysis._shared.dataset_loaders import load_flores
from analysis._shared.varieties import VARIETY_CODES
from analysis.fasttext.core.embed import per_sentence_fasttext
from evaluation._bootstrap_core import (
    bootstrap_from_sentence_vectors, default_gold_paths, flatten_per_variety,
)


EXPERIMENTS = {
    "fasttext_wikiOLDI_native":     "native",
    "fasttext_wikiOLDI_normalized": "normalized",
}


def _experiment_dir(experiment: str) -> Path:
    return REPO_ROOT / "analysis" / "fasttext" / "experiments" / experiment


def _model_path(experiment: str) -> Path:
    return (_experiment_dir(experiment) / "method_outputs"
            / "models" / "fasttext_model.bin")


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
            f"Run analysis/fasttext/experiments/{experiment}/run.py first "
            f"to produce it — this bootstrap script never retrains."
        )

    print(f"=== {experiment}  (text_variant={text_variant}) ===")
    print(f"  load model: {mp.relative_to(REPO_ROOT)}")
    model = FastText.load(str(mp))

    print(f"  loading FLORES ({text_variant}) ...")
    flores_data, _ = load_flores(text_variant=text_variant, verbose=False)

    per_var = per_sentence_fasttext(model, flores_data, VARIETY_CODES)
    sent_vecs, sent_codes = flatten_per_variety(per_var, VARIETY_CODES)
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
