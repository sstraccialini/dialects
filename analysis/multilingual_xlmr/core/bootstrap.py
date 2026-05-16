"""
Bootstrap CI on the Spearman gold correlations for an XLM-R experiment.

zero-shot   : loads the pretrained ``xlm-roberta-base`` from HF cache.
fine-tuned  : loads the fine-tuned checkpoint at
              ``method_outputs/models/xlmr_finetuned``.
              Errors out if missing — does NOT retrain.

Output:
    analysis/multilingual_xlmr/experiments/<exp>/evaluation_results/flores/centroid/bootstrap_results.csv

CLI:
    python -m analysis.multilingual_xlmr.core.bootstrap \\
        --experiment xlmr_zeroshot_native --n-boot 1000
    python -m analysis.multilingual_xlmr.core.bootstrap \\
        --experiment xlmr_finetuned_wikiOLDI_dialects_native --n-boot 1000
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
from analysis.multilingual_xlmr.core.config import MODEL_NAME, MAX_LENGTH, BATCH_SIZE
from analysis.multilingual_xlmr.core.embedder import MultilingualEmbedder
from evaluation._bootstrap_core import (
    bootstrap_from_sentence_vectors, default_gold_paths,
)


EXPERIMENTS = {
    "xlmr_zeroshot_native":                    ("native", None),
    "xlmr_finetuned_wikiOLDI_dialects_native": ("native", "xlmr_finetuned"),
}


def _experiment_dir(experiment: str) -> Path:
    return REPO_ROOT / "analysis" / "multilingual_xlmr" / "experiments" / experiment


def _model_source(experiment: str) -> str:
    _, ckpt_name = EXPERIMENTS[experiment]
    if ckpt_name is None:
        return MODEL_NAME
    p = _experiment_dir(experiment) / "method_outputs" / "models" / ckpt_name
    if not (p / "config.json").exists():
        raise SystemExit(
            f"Fine-tuned checkpoint not found at {p.relative_to(REPO_ROOT)}.\n"
            f"Run analysis/multilingual_xlmr/experiments/{experiment}/run.py "
            f"first — this bootstrap script never retrains."
        )
    return str(p)


def _output_csv(experiment: str) -> Path:
    return (_experiment_dir(experiment)
            / "evaluation_results" / "flores" / "centroid"
            / "bootstrap_results.csv")


def _flatten(data: dict, codes):
    sents, sent_codes = [], []
    for code in codes:
        if code not in data:
            continue
        for s in data[code]:
            sents.append(s)
            sent_codes.append(code)
    return sents, sent_codes


def run(experiment: str, n_boot: int, seed: int, device: str | None) -> Path:
    if experiment not in EXPERIMENTS:
        raise SystemExit(f"Unknown experiment {experiment!r}. "
                         f"Choose from {list(EXPERIMENTS)}.")
    text_variant, _ = EXPERIMENTS[experiment]
    source = _model_source(experiment)

    print(f"=== {experiment}  (text_variant={text_variant}) ===")
    print(f"  load model: {source}")
    embedder = MultilingualEmbedder(
        model_name=source, device=device, max_length=MAX_LENGTH,
    )

    print(f"  loading FLORES ({text_variant}) ...")
    flores_data, _ = load_flores(text_variant=text_variant, verbose=False)
    sents, sent_codes = _flatten(flores_data, VARIETY_CODES)

    print(f"  encoding {len(sents):,} sentences ...")
    sent_vecs = embedder.encode(sents, batch_size=BATCH_SIZE)
    print(f"  done (dim={sent_vecs.shape[1]})")

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
    ap.add_argument("--device", type=str, default=None,
                    help='"cuda", "cpu", "mps", or None for auto.')
    args = ap.parse_args(argv)
    run(args.experiment, args.n_boot, args.seed, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
