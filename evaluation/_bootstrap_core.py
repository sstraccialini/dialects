"""
Shared bootstrap math used by every method-specific bootstrap script under
``analysis/<method>/core/bootstrap.py``.

Given per-sentence FLORES embeddings (one row per sentence, with the
sentence's variety code), produce a tidy DataFrame with the Spearman ρ
observed value and a bootstrap CI against every gold reference matrix.

Stratified resampling: for B iterations, within each variety we draw
N_var indices with replacement and average → centroid → L2-normalise →
17×17 cosine distance → Spearman vs each gold (full triangle +
dialect↔external block).

The CI is the [α/2, 1-α/2] empirical quantile of the resampled ρ.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from evaluation._gold_correlation import (
    _restrict_to,
    _spearman_safe,
    default_roles,
    load_gold,
)


# --------------------------------------------------------------------------- #
# Centroid + distance helpers (mirror what the production pipeline does)
# --------------------------------------------------------------------------- #

def _l2_normalise(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=-1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return X / n


def _cosine_distance_matrix(X: np.ndarray) -> np.ndarray:
    sim = cosine_similarity(X)
    sim = (sim + sim.T) / 2.0
    d = 1.0 - sim
    np.fill_diagonal(d, 0.0)
    np.clip(d, 0.0, None, out=d)
    return d


def _spearman_pair_from_dist(
    dist: np.ndarray, codes: Sequence[str],
    gold_mat: np.ndarray, gold_labels: Sequence[str],
    dialect_codes: Sequence[str], external_codes: Sequence[str],
) -> Tuple[float, float]:
    """Like ``_spearman_pair`` but takes a pre-computed 17×17 distance
    matrix.  Hoisting the cosine out of the per-gold loop gives a 3×
    speedup when there are 3 gold matrices (LDND, LEV, Geo)."""
    shared = [c for c in gold_labels if c in codes]
    if len(shared) < 4:
        return float("nan"), float("nan")
    md, _ = _restrict_to(dist,     codes,       shared)
    gd, _ = _restrict_to(gold_mat, gold_labels, shared)
    iu = np.triu_indices(len(shared), k=1)
    rho_full = _spearman_safe(md[iu], gd[iu])

    d = [c for c in dialect_codes if c in shared]
    e = [c for c in external_codes if c in shared]
    if d and e:
        d_idx = [shared.index(c) for c in d]
        e_idx = [shared.index(c) for c in e]
        rho_dia = _spearman_safe(
            md[np.ix_(d_idx, e_idx)].flatten(),
            gd[np.ix_(d_idx, e_idx)].flatten(),
        )
    else:
        rho_dia = float("nan")
    return rho_full, rho_dia


def _spearman_pair(
    cent: np.ndarray, codes: Sequence[str],
    gold_mat: np.ndarray, gold_labels: Sequence[str],
    dialect_codes: Sequence[str], external_codes: Sequence[str],
) -> Tuple[float, float]:
    return _spearman_pair_from_dist(
        _cosine_distance_matrix(cent), codes,
        gold_mat, gold_labels, dialect_codes, external_codes,
    )


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #

def bootstrap_from_sentence_vectors(
    sent_vecs:    np.ndarray,
    sent_codes:   Sequence[str],
    variety_codes: Sequence[str],
    gold_paths:   Sequence[Path],
    *,
    n_boot:        int = 1000,
    alpha:         float = 0.05,
    seed:          int = 42,
    dialect_codes:  Sequence[str] | None = None,
    external_codes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return tidy DataFrame:
        gold, block, rho_observed, rho_mean, rho_lo, rho_hi, n_boot
    where block ∈ {"full", "dialect_external"}.
    """
    sent_codes_arr = np.asarray(sent_codes)
    if dialect_codes is None or external_codes is None:
        d_def, e_def = default_roles()
        dialect_codes  = dialect_codes  or d_def
        external_codes = external_codes or e_def

    golds = [(p.stem, *load_gold(p)[:2]) for p in gold_paths]
    rng = np.random.default_rng(seed)

    per_code_indices: Dict[str, np.ndarray] = {
        c: np.where(sent_codes_arr == c)[0] for c in variety_codes
    }
    per_code_indices = {c: idx for c, idx in per_code_indices.items() if len(idx) > 0}
    codes_present = [c for c in variety_codes if c in per_code_indices]

    # Observed centroids.
    obs_rows = [sent_vecs[per_code_indices[c]].mean(axis=0) for c in codes_present]
    obs_cent = _l2_normalise(np.vstack(obs_rows).astype(np.float32))

    obs_dist = _cosine_distance_matrix(obs_cent)
    observed: Dict[str, Tuple[float, float]] = {}
    for name, mat, labels in golds:
        observed[name] = _spearman_pair_from_dist(
            obs_dist, codes_present, mat, labels,
            dialect_codes, external_codes,
        )

    samples: Dict[str, List[Tuple[float, float]]] = {n: [] for n, _, _ in golds}
    for _ in range(n_boot):
        rows = []
        for c in codes_present:
            idx = per_code_indices[c]
            sample = rng.choice(idx, size=len(idx), replace=True)
            rows.append(sent_vecs[sample].mean(axis=0))
        cent = _l2_normalise(np.vstack(rows).astype(np.float32))
        dist = _cosine_distance_matrix(cent)
        for name, mat, labels in golds:
            samples[name].append(
                _spearman_pair_from_dist(dist, codes_present, mat, labels,
                                         dialect_codes, external_codes)
            )

    lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0
    out_rows = []
    for name in samples:
        arr = np.asarray(samples[name], dtype=np.float64)  # (n_boot, 2)
        for j, block in enumerate(("full", "dialect_external")):
            col = arr[:, j]
            col = col[np.isfinite(col)]
            rho_obs = observed[name][j]
            if col.size:
                out_rows.append({
                    "gold":         name,
                    "block":        block,
                    "rho_observed": rho_obs,
                    "rho_mean":     float(col.mean()),
                    "rho_lo":       float(np.quantile(col, lo_q)),
                    "rho_hi":       float(np.quantile(col, hi_q)),
                    "n_boot":       int(col.size),
                })
            else:
                out_rows.append({
                    "gold": name, "block": block,
                    "rho_observed": rho_obs,
                    "rho_mean": float("nan"),
                    "rho_lo":   float("nan"),
                    "rho_hi":   float("nan"),
                    "n_boot":   0,
                })
    return pd.DataFrame(out_rows)


def flatten_per_variety(
    per_variety: Dict[str, np.ndarray],
    codes: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Helper: turn ``{code: (N_c, D)}`` into ``(sent_vecs, sent_codes)``."""
    rows, sc = [], []
    for c in codes:
        mat = per_variety.get(c)
        if mat is None or mat.shape[0] == 0:
            continue
        rows.append(mat.astype(np.float32))
        sc.extend([c] * mat.shape[0])
    return np.vstack(rows), np.asarray(sc)


def default_gold_paths(repo_root: Path) -> List[Path]:
    """Standard locations for the 2 matrix golds (lex + geo).  Historical
    influence gold is set-based, not matrix, so excluded here."""
    out: List[Path] = []
    for d in (
        repo_root / "gold" / "lexicostatistical" / "matrices",
        repo_root / "gold" / "geographic"        / "matrices",
    ):
        if d.is_dir():
            out.extend(sorted(d.glob("*.npz")))
    return out
