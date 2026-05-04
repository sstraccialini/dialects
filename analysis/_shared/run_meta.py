"""
Reproducibility helper: write a `run_meta.json` next to each experiment's
`run_stats.csv` so we can always reconstruct what produced a given output.

Usage from a `run.py`:

    from analysis._shared.run_meta import write_run_meta
    write_run_meta(
        out_dir=mo,
        method="fasttext",
        experiment="wiki_to_flores",
        params={"vector_size": 200, "epochs": 10, ...},
    )
"""
from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def write_run_meta(
    out_dir: Path,
    *,
    method: str,
    experiment: str,
    params: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a small JSON describing how a run was produced.

    Output: `<out_dir>/run_meta.json`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta: Dict[str, Any] = {
        "method":     method,
        "experiment": experiment,
        "timestamp":  datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "git_commit": _git_commit(),
        "python":     platform.python_version(),
        "platform":   platform.platform(),
        "params":     params or {},
    }
    if extra:
        meta.update(extra)

    path = out_dir / "run_meta.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    return path
