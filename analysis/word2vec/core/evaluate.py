"""
Thin wrappers around the central `evaluation/` package, pre-applying
the Word2Vec method's variety taxonomy.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Union

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation import run_evaluation, run_parallel_alignment

from analysis._shared.varieties import DIALECT_FAMILIES

from .config import (
    VARIETY_GROUP, GROUP_NAMES, GROUP_COLORS, VARIETY_NAMES, ROMANCE_FAMILIES,
)


def variety_eval(
    X,
    codes,
    out_dir: Union[str, Path],
    *,
    method_label: str = "Word2Vec",
):
    if hasattr(X, "toarray"):
        X = X.toarray()
    return run_evaluation(
        variety_vectors=np.asarray(X),
        variety_codes=list(codes),
        out_dir=out_dir,
        method_label=method_label,
        family_groups=VARIETY_GROUP,
        family_colors=GROUP_COLORS,
        family_display_names=GROUP_NAMES,
        display_names=VARIETY_NAMES,
        romance_families=ROMANCE_FAMILIES,
        dialect_families=DIALECT_FAMILIES,
        isotropy=False,
        isotropy_top_k_pc=1,
    )


def parallel_eval(
    sentence_vectors: Dict[str, np.ndarray],
    out_dir: Union[str, Path],
    *,
    method_label: str = "Word2Vec",
):
    return run_parallel_alignment(
        sentence_vectors=sentence_vectors,
        out_dir=out_dir,
        method_label=method_label,
        family_groups=VARIETY_GROUP,
        family_colors=GROUP_COLORS,
        family_display_names=GROUP_NAMES,
        display_names=VARIETY_NAMES,
    )
