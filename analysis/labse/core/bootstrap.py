"""
Bootstrap CI on the Spearman gold correlations for a LaBSE experiment.

zero-shot   : loads pretrained ``sentence-transformers/LaBSE`` from HF cache.
fine-tuned  : loads the MNRL checkpoint at
              ``method_outputs/models/labse_mnrl``.
              Errors out if missing — does NOT retrain.

Output:
    analysis/labse/experiments/<exp>/evaluation_results/flores/centroid/bootstrap_results.csv

CLI:
    python -m analysis.labse.core.bootstrap \\
        --experiment labse_zeroshot_native --n-boot 1000
    python -m analysis.labse.core.bootstrap \\
        --experiment labse_finetuned_oldi_dialects_native --n-boot 1000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis._shared.dataset_loaders import load_flores
from analysis._shared.varieties import VARIETY_CODES
from evaluation._bootstrap_core import (
    bootstrap_from_sentence_vectors, default_gold_paths, flatten_per_variety,
)


LABSE_MODEL = "sentence-transformers/LaBSE"
BATCH_SIZE  = 64

EXPERIMENTS = {
    "labse_zeroshot_native":               ("native", None),
    "labse_finetuned_oldi_dialects_native": ("native", "labse_mnrl"),
}


def _experiment_dir(experiment: str) -> Path:
    return REPO_ROOT / "analysis" / "labse" / "experiments" / experiment


def _model_source(experiment: str) -> str:
    _, ckpt_name = EXPERIMENTS[experiment]
    if ckpt_name is None:
        return LABSE_MODEL
    p = _experiment_dir(experiment) / "method_outputs" / "models" / ckpt_name
    if not p.exists():
        raise SystemExit(
            f"Fine-tuned checkpoint not found at {p.relative_to(REPO_ROOT)}.\n"
            f"Run analysis/labse/experiments/{experiment}/run.py first — "
            f"this bootstrap script never retrains."
        )
    return str(p)


def _output_csv(experiment: str) -> Path:
    return (_experiment_dir(experiment)
            / "evaluation_results" / "flores" / "centroid"
            / "bootstrap_results.csv")


def run(experiment: str, n_boot: int, seed: int, device: str | None) -> Path:
    if experiment not in EXPERIMENTS:
        raise SystemExit(f"Unknown experiment {experiment!r}. "
                         f"Choose from {list(EXPERIMENTS)}.")
    text_variant, _ = EXPERIMENTS[experiment]
    source = _model_source(experiment)

    print(f"=== {experiment}  (text_variant={text_variant}) ===")
    print(f"  load model: {source}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(source, device=device)

    print(f"  loading FLORES ({text_variant}) ...")
    flores_data, _ = load_flores(text_variant=text_variant, verbose=False)

    print(f"  encoding ...")
    per_variety: dict = {}
    for code in VARIETY_CODES:
        if code not in flores_data:
            continue
        emb = model.encode(
            flores_data[code], batch_size=BATCH_SIZE,
            convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
        per_variety[code] = emb.astype(np.float32)
    sent_vecs, sent_codes = flatten_per_variety(per_variety, VARIETY_CODES)
    print(f"  done — {sent_vecs.shape[0]:,} sentences (dim={sent_vecs.shape[1]})")

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
