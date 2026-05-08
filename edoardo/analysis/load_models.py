"""
Discover and load every model output in the repo.

A "model" here is one (method, root_kind, experiment, variant) tuple, e.g.

    method='multilingual_xlmr', root_kind='experiments',
    experiment='mlm_wiki_to_flores_oldi', variant='flores/centroid'

For each model we expose:

    distance_matrix(codes)  → (N, N) cosine distance restricted to ``codes``
    variety_vectors(codes)  → (N, D) embedding rows (NaN-padded if missing)

Used by:
    - correlate_with_gold.py       (needs distance matrices)
    - cka_baseline_vs_finetuned.py (needs variety_vectors)
    - cluster_agreement.py         (needs distance matrices)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from edoardo.varieties_extra import VARIETY_CODES


REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = REPO_ROOT / "analysis"


@dataclass(frozen=True)
class ModelOutput:
    method: str
    root_kind: str
    experiment: str
    variant_path: str
    distances_csv: Path
    variety_vectors_npz: Optional[Path]
    run_meta: Dict = field(default_factory=dict)

    @property
    def model_id(self) -> str:
        parts = [self.method, self.experiment]
        if self.variant_path:
            parts.append(self.variant_path.replace("/", "__"))
        return "/".join([self.method, self.experiment] +
                        ([self.variant_path] if self.variant_path else []))

    @property
    def short_id(self) -> str:
        """Stable filename-safe identifier."""
        v = self.variant_path.replace("/", "_") if self.variant_path else "default"
        return f"{self.method}__{self.experiment}__{v}"

    def load_distances(self) -> Tuple[np.ndarray, List[str]]:
        df = pd.read_csv(self.distances_csv, index_col=0)
        return df.values.astype(np.float64), list(df.index)

    def load_vectors(self) -> Tuple[np.ndarray, List[str]]:
        if self.variety_vectors_npz is None or not self.variety_vectors_npz.exists():
            raise FileNotFoundError(
                f"variety_vectors.npz not found for {self.model_id}")
        data = np.load(self.variety_vectors_npz, allow_pickle=True)
        return data["matrix"].astype(np.float64), [str(x) for x in data["labels"]]


def _parse_eval_path(p: Path) -> Tuple[str, str, str, str]:
    rel = p.relative_to(ANALYSIS_DIR).parts
    method = rel[0]
    root_kind = rel[1] if rel[1] in ("experiments", "old_experiments") else "?"
    experiment = rel[2] if root_kind != "?" else ""
    try:
        er_idx = rel.index("evaluation_results")
        variant = "/".join(rel[er_idx + 1 : -1])
    except ValueError:
        variant = ""
    return method, root_kind, experiment, variant


def _matching_method_outputs(distances_csv: Path) -> Path:
    parts = list(distances_csv.parts)
    try:
        er_idx = parts.index("evaluation_results")
    except ValueError:
        return distances_csv.parent
    parts[er_idx] = "method_outputs"
    return Path(*parts).parent


def _load_run_meta(method_outputs_dir: Path) -> Dict:
    for cand in (method_outputs_dir / "run_meta.json",
                 method_outputs_dir.parent / "run_meta.json"):
        if cand.exists():
            try:
                with cand.open() as fh:
                    return json.load(fh)
            except Exception:
                pass
    return {}


def discover_models(include_old: bool = False) -> List[ModelOutput]:
    out: List[ModelOutput] = []
    for method_dir in sorted(p for p in ANALYSIS_DIR.iterdir() if p.is_dir()):
        if method_dir.name.startswith("_"):
            continue
        roots = [method_dir / "experiments"]
        if include_old:
            roots.append(method_dir / "old_experiments")
        for er in roots:
            if not er.exists():
                continue
            for dist_csv in er.rglob("distances.csv"):
                method, root_kind, experiment, variant = _parse_eval_path(dist_csv)
                if not experiment:
                    continue
                mo = _matching_method_outputs(dist_csv)
                vv = mo / "variety_vectors.npz"
                out.append(ModelOutput(
                    method=method,
                    root_kind=root_kind,
                    experiment=experiment,
                    variant_path=variant,
                    distances_csv=dist_csv,
                    variety_vectors_npz=vv if vv.exists() else None,
                    run_meta=_load_run_meta(mo),
                ))
    return out


def restrict_to_codes(
    matrix: np.ndarray, labels: List[str], target_codes: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """Restrict a distance/vector matrix to ``target_codes`` (in order).

    Returns the restricted (square or N*D) matrix and the actual list of
    codes kept (a subset of ``target_codes``).  Codes not present in
    ``labels`` are silently dropped — caller is responsible for noting
    coverage in the report.
    """
    keep = [c for c in target_codes if c in labels]
    idx = [labels.index(c) for c in keep]
    if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1] == len(labels):
        return matrix[np.ix_(idx, idx)], keep
    if matrix.ndim == 2 and matrix.shape[0] == len(labels):
        return matrix[idx, :], keep
    raise ValueError(f"Unexpected matrix shape {matrix.shape} vs {len(labels)} labels")


def canonical_codes() -> List[str]:
    """The 13 canonical varieties (project-wide order)."""
    return list(VARIETY_CODES)
