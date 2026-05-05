"""Thin wrapper around the central evaluation package, with the
xphonebert-specific 15-variety registry."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Union

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation import run_evaluation  # noqa: E402

from .config import (
    VARIETY_GROUP, GROUP_NAMES, GROUP_COLORS, VARIETY_NAMES, ROMANCE_FAMILIES,
)


def variety_eval(
    X, codes, out_dir: Union[str, Path], *, method_label: str = "XPhoneBERT",
):
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
    )
