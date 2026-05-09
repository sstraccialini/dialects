"""
Shared helpers for correlating any model's distance matrix against
the team-curated gold reference matrices under ``gold/<family>/matrices/``.

Used by:
  * ``evaluation/correlate_against_gold.py`` (batch, runs across all
    models discovered under ``analysis/``)
  * ``evaluation/evaluation.py::run_evaluation`` (per-experiment, called
    by every method's ``run.py``)

Two correlation metrics, computed for every (model, gold) pair:

    Spearman ρ (full matrix)
        Spearman rank correlation on the FULL upper triangle of the
        shared-variety distance matrix.

    Spearman ρ (dialect ↔ external)
        Spearman restricted to the cross-block of (dialect × external)
        pairs, where ``external`` excludes both the other dialects and
        standard Italian.  Captures genealogy-crossing relationships
        only.

Both metrics are in [-1, +1].  See gold/_correlations/README.md for
interpretation per gold type.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# Public column names — used wherever this is rendered as a table.
COL_RHO_FULL   = "Spearman ρ (full matrix)"
COL_RHO_DIAEXT = "Spearman ρ (dialect ↔ external)"


# --------------------------------------------------------------------------- #
# Default gold-matrix discovery
# --------------------------------------------------------------------------- #

def repo_root() -> Path:
    """Return the project root (.../Language-Technology-Project)."""
    return Path(__file__).resolve().parents[1]


def default_gold_roots() -> List[Path]:
    """Standard places where gold matrices live.  Extend here when adding
    new gold families (e.g. typological)."""
    root = repo_root() / "gold"
    return [
        root / "lexicostatistical" / "matrices",
        root / "geographic"        / "matrices",
    ]


def find_gold_matrices(roots: Iterable[Path]) -> List[Path]:
    """Return every ``*.npz`` found in the given directories."""
    out: List[Path] = []
    for r in roots:
        r = Path(r)
        if r.is_dir():
            out.extend(sorted(r.glob("*.npz")))
    return out


def load_gold(npz_path: Path) -> Tuple[np.ndarray, List[str], dict]:
    """Load a gold .npz with keys ``matrix``, ``labels``, ``meta``."""
    data = np.load(npz_path, allow_pickle=True)
    mat = np.asarray(data["matrix"], dtype=np.float64)
    labels = [str(x) for x in data["labels"]]
    try:
        meta = json.loads(str(data["meta"][0]))
    except Exception:
        meta = {}
    return mat, labels, meta


# --------------------------------------------------------------------------- #
# Default role assignment (which codes are dialects vs external)
# --------------------------------------------------------------------------- #

def default_roles() -> Tuple[List[str], List[str]]:
    """Read ``DIALECT_CODES`` and ``EXTERNAL_CODES`` from the
    lexicostatistical varieties module.  Falls back to the current 6 + 6
    set if the import fails for any reason.
    """
    try:
        from gold.lexicostatistical.varieties import (
            DIALECT_CODES,
            EXTERNAL_CODES,
        )
        return list(DIALECT_CODES), list(EXTERNAL_CODES)
    except Exception:
        return (
            ["fur", "lij", "lmo", "sc", "scn", "vec"],
            ["fra", "spa", "cat", "deu", "slv", "eng"],
        )


# --------------------------------------------------------------------------- #
# Math
# --------------------------------------------------------------------------- #

def _restrict_to(matrix: np.ndarray, labels: Sequence[str],
                 target: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    keep = [c for c in target if c in labels]
    idx = [list(labels).index(c) for c in keep]
    return matrix[np.ix_(idx, idx)], keep


def _spearman_safe(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or b.size < 3:
        return float("nan")
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    r, _ = spearmanr(a[mask], b[mask])
    return float(r) if r is not None else float("nan")


def correlate_against_gold(
    model_dist: np.ndarray, model_labels: Sequence[str],
    gold_dist: np.ndarray,  gold_labels:  Sequence[str],
    dialect_codes: Sequence[str], external_codes: Sequence[str],
) -> Tuple[float, float, int]:
    """Return ``(rho_full, rho_dialect_external, n_shared)``."""
    shared = [c for c in gold_labels if c in model_labels]
    n_shared = len(shared)
    if n_shared < 4:
        return float("nan"), float("nan"), n_shared

    md, _ = _restrict_to(model_dist, model_labels, shared)
    gd, _ = _restrict_to(gold_dist,  gold_labels,  shared)

    iu = np.triu_indices(md.shape[0], k=1)
    rho_full = _spearman_safe(md[iu], gd[iu])

    d_set = [c for c in dialect_codes if c in shared]
    e_set = [c for c in external_codes if c in shared]
    if d_set and e_set:
        d_idx = [shared.index(c) for c in d_set]
        e_idx = [shared.index(c) for c in e_set]
        a = md[np.ix_(d_idx, e_idx)].flatten()
        b = gd[np.ix_(d_idx, e_idx)].flatten()
        rho_dia = _spearman_safe(a, b)
    else:
        rho_dia = float("nan")

    return rho_full, rho_dia, n_shared


def correlate_one_model_to_all_golds(
    model_dist: np.ndarray, model_labels: Sequence[str],
    gold_paths: Sequence[Path],
    dialect_codes: Optional[Sequence[str]] = None,
    external_codes: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """For a single model and a list of gold .npz files, return a DataFrame
    with one row per gold and columns
        gold, Spearman ρ (full matrix), Spearman ρ (dialect ↔ external),
        n_shared_with_gold.
    """
    if dialect_codes is None or external_codes is None:
        d_def, e_def = default_roles()
        if dialect_codes is None:
            dialect_codes = d_def
        if external_codes is None:
            external_codes = e_def

    rows = []
    for gp in gold_paths:
        try:
            gold_mat, gold_labels, _ = load_gold(gp)
        except Exception as exc:
            rows.append({
                "gold": gp.stem,
                COL_RHO_FULL:   float("nan"),
                COL_RHO_DIAEXT: float("nan"),
                "n_shared_with_gold": 0,
                "_load_error": str(exc),
            })
            continue
        rho_full, rho_dia, n_sh = correlate_against_gold(
            model_dist, model_labels, gold_mat, gold_labels,
            dialect_codes, external_codes,
        )
        rows.append({
            "gold": gp.stem,
            COL_RHO_FULL:   rho_full,
            COL_RHO_DIAEXT: rho_dia,
            "n_shared_with_gold": n_sh,
        })
    return pd.DataFrame(rows)


def write_per_experiment_gold_csv(
    out_path: Path,
    model_dist: np.ndarray, model_labels: Sequence[str],
    gold_paths: Optional[Sequence[Path]] = None,
    dialect_codes: Optional[Sequence[str]] = None,
    external_codes: Optional[Sequence[str]] = None,
) -> Optional[pd.DataFrame]:
    """Convenience wrapper for ``run_evaluation``.  Discovers golds, writes
    a small CSV with one row per gold to ``out_path``.  Returns the
    DataFrame (or None if no gold found)."""
    if gold_paths is None:
        gold_paths = find_gold_matrices(default_gold_roots())
    if not gold_paths:
        return None

    df = correlate_one_model_to_all_golds(
        model_dist, model_labels, gold_paths,
        dialect_codes=dialect_codes, external_codes=external_codes,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.4f")
    return df
