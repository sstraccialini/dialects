"""Shared embedding+evaluation routine for the 4 conditions."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..core.config import VARIETY_CODES, BATCH_SIZE, MAX_LENGTH
from ..core.data_loader import load_flores_parallel, load_oldi_parallel
from ..core.embedder import embed_per_variety, aggregate_from_per_variety
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
    flores_data, _ = load_flores_parallel(verbose=False)
    per_variety = embed_per_variety(flores_data, VARIETY_CODES,
                                    model_name_or_path=model_path,
                                    batch_size=batch_size, device=device,
                                    max_length=max_length)
    X, codes = aggregate_from_per_variety(per_variety, VARIETY_CODES)

    mo = experiment_dir / "method_outputs" / "flores"
    mo.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X, index=codes).to_csv(mo / "variety_vectors.csv", float_format="%.6f")
    np.savez_compressed(mo / "variety_vectors.npz",
                        matrix=X.astype(np.float32), labels=np.asarray(codes),
                        condition=np.asarray(condition))

    er_centroid = experiment_dir / "evaluation_results" / "flores" / "centroid"
    er_centroid.mkdir(parents=True, exist_ok=True)
    report = variety_eval(X, codes, out_dir=er_centroid,
                          method_label=f"SentenceTransformer FLORES centroid ({condition})")

    er_par = experiment_dir / "evaluation_results" / "flores" / "parallel"
    er_par.mkdir(parents=True, exist_ok=True)
    parallel_eval(per_variety, out_dir=er_par,
                  method_label=f"SentenceTransformer FLORES parallel ({condition})")
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
    oldi_data, _ = load_oldi_parallel(verbose=False)
    if not oldi_data:
        print("  [oldi] no data — skipping OLDI evaluation")
        return None

    per_variety = embed_per_variety(oldi_data, VARIETY_CODES,
                                    model_name_or_path=model_path,
                                    batch_size=batch_size, device=device,
                                    max_length=max_length)
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
                          method_label=f"SentenceTransformer OLDI centroid ({condition})")

    er_par = experiment_dir / "evaluation_results" / "oldi" / "parallel"
    er_par.mkdir(parents=True, exist_ok=True)
    parallel_eval(per_variety, out_dir=er_par,
                  method_label=f"SentenceTransformer OLDI parallel ({condition})")
    return report
