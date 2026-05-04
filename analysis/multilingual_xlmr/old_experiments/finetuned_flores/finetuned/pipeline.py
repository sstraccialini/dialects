"""
Shared embedding+evaluation routine used by every condition's run.py.
Eliminates repetition across baseline / mlm_wiki / tlm_oldi / mlm_then_tlm.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..core.config import VARIETY_CODES, BATCH_SIZE, MAX_LENGTH
from ..core.data_loader import (
    iter_labeled_sentences,
    load_flores_parallel, load_oldi_parallel,
)
from ..core.embedder import (
    MultilingualEmbedder, aggregate_variety_vectors, aggregate_from_per_variety,
)
from ..core.evaluate import variety_eval, parallel_eval


def embed_and_evaluate_flores(
    model_path: str,
    experiment_dir: Path,
    *,
    condition: str,
    device: Optional[str] = None,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
):
    """Embed FLORES+ centroid + parallel; write outputs under experiment_dir.

    Returns the report dict from `variety_eval`.
    """
    flores_data, _ = load_flores_parallel(verbose=False)

    embedder = MultilingualEmbedder(model_name=model_path, device=device, max_length=max_length)
    sents, sent_codes = iter_labeled_sentences(flores_data)
    sent_vecs = embedder.encode(sents, batch_size=batch_size)
    X, codes = aggregate_variety_vectors(sent_vecs, sent_codes, VARIETY_CODES)

    mo = experiment_dir / "method_outputs" / "flores"
    mo.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X, index=codes).to_csv(mo / "variety_vectors.csv", float_format="%.6f")
    np.savez_compressed(mo / "variety_vectors.npz",
                        matrix=X.astype(np.float32), labels=np.asarray(codes),
                        condition=np.asarray(condition))

    er_centroid = experiment_dir / "evaluation_results" / "flores" / "centroid"
    er_centroid.mkdir(parents=True, exist_ok=True)
    report = variety_eval(X, codes, out_dir=er_centroid,
                          method_label=f"XLM-R FLORES centroid ({condition})")

    # Per-variety per-sentence for parallel-alignment eval
    per_variety = embedder.encode_per_variety(flores_data, VARIETY_CODES, batch_size=batch_size)
    er_par = experiment_dir / "evaluation_results" / "flores" / "parallel"
    er_par.mkdir(parents=True, exist_ok=True)
    parallel_eval(per_variety, out_dir=er_par,
                  method_label=f"XLM-R FLORES parallel ({condition})")

    del embedder
    return report


def embed_and_evaluate_oldi(
    model_path: str,
    experiment_dir: Path,
    *,
    condition: str,
    device: Optional[str] = None,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
):
    """Same as embed_and_evaluate_flores but on OLDI parallel data."""
    oldi_data, _ = load_oldi_parallel(verbose=False)
    if not oldi_data:
        print("  [oldi] no data — skipping OLDI evaluation")
        return None

    embedder = MultilingualEmbedder(model_name=model_path, device=device, max_length=max_length)
    per_variety = embedder.encode_per_variety(oldi_data, VARIETY_CODES, batch_size=batch_size)
    X, codes = aggregate_from_per_variety(per_variety, VARIETY_CODES)

    mo = experiment_dir / "method_outputs" / "oldi"
    mo.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X, index=codes).to_csv(mo / "variety_vectors.csv", float_format="%.6f")
    np.savez_compressed(mo / "variety_vectors.npz",
                        matrix=X.astype(np.float32), labels=np.asarray(codes),
                        condition=np.asarray(condition))

    er_centroid = experiment_dir / "evaluation_results" / "oldi" / "centroid"
    er_centroid.mkdir(parents=True, exist_ok=True)
    report = variety_eval(X, codes, out_dir=er_centroid,
                          method_label=f"XLM-R OLDI centroid ({condition})")

    er_par = experiment_dir / "evaluation_results" / "oldi" / "parallel"
    er_par.mkdir(parents=True, exist_ok=True)
    parallel_eval(per_variety, out_dir=er_par,
                  method_label=f"XLM-R OLDI parallel ({condition})")

    del embedder
    return report
