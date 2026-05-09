"""
Common helpers for downloading + parsing CLDF datasets (Grambank, PHOIBLE,
Lexibank).  All three follow the Cross-Linguistic Data Format convention:
   <root>/cldf/{languages,parameters,values,codes}.csv

We pin to specific release versions (Zenodo or GitHub tags) so that runs
are reproducible.  Each builder calls ``ensure_dataset(name, url, cache)``
once and then parses the cached unpack.
"""
from __future__ import annotations

import csv
import hashlib
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional


CACHE_DEFAULT = Path(os.environ.get("EDOARDO_GOLD_CACHE",
                                    Path.home() / ".cache" / "edoardo_gold"))


# Pinned releases — change these only if you intend to upgrade the dataset.
DATASETS: Dict[str, Dict[str, str]] = {
    "grambank": {
        "url":   "https://zenodo.org/records/7844558/files/grambank/grambank-v1.0.3.zip?download=1",
        "name":  "grambank-v1.0.3",
    },
    "phoible": {
        "url":   "https://github.com/cldf-datasets/phoible/archive/refs/tags/v2.0.1.zip",
        "name":  "phoible-2.0.1",
    },
    "lexibank": {
        "url":   "https://github.com/lexibank/lexibank-analysed/archive/refs/tags/v2.1.zip",
        "name":  "lexibank-analysed-2.1",
    },
    "glottolog_coords": {
        # Compact CSV of (Glottocode, Latitude, Longitude) extracted from
        # Glottolog 5.x.  We mirror it on the user's HPC if absent — see
        # ``download_glottolog_languoids`` below.
        "url":   "https://glottolog.org/static/download/5.0/glottolog_languoid.cldf.zip",
        "name":  "glottolog-5.0",
    },
}


def ensure_dataset(key: str, cache: Path = CACHE_DEFAULT) -> Path:
    """Download (once) + unpack a dataset; return the unpacked root directory.

    If a previous extraction failed and left debris, clean it before
    retrying.  We treat an extract_dir as "valid" only if a ``languages.csv``
    file is reachable beneath it (the universal CLDF marker).
    """
    cfg = DATASETS[key]
    cache.mkdir(parents=True, exist_ok=True)
    url = cfg["url"]
    extract_dir = cache / cfg["name"]

    if extract_dir.exists():
        if any(extract_dir.rglob("languages.csv")):
            return _find_inner_root(extract_dir)
        # partial / broken extract — wipe and retry
        print(f"[cldf] cleaning partial extract at {extract_dir}")
        shutil.rmtree(extract_dir)

    zip_path = cache / f"{cfg['name']}.zip"
    if not zip_path.exists():
        print(f"[cldf] downloading {url} -> {zip_path}")
        urllib.request.urlretrieve(url, zip_path)
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"[cldf] unpacking {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)
    return _find_inner_root(extract_dir)


def _find_inner_root(extract_dir: Path) -> Path:
    """GitHub zips wrap content in a single inner folder; descend if needed."""
    children = [p for p in extract_dir.iterdir() if p.is_dir()]
    if len(children) == 1 and not (extract_dir / "cldf").exists():
        return children[0]
    return extract_dir


def find_cldf_dir(root: Path) -> Path:
    """Locate the ``cldf/`` subdirectory inside an unpacked release."""
    for cand in [root / "cldf",
                 root / "cldf-datasets" / "cldf",
                 *root.rglob("languages.csv")]:
        if cand.is_dir() and (cand / "languages.csv").exists():
            return cand
        if cand.is_file() and cand.name == "languages.csv":
            return cand.parent
    raise RuntimeError(f"No cldf/ subtree with languages.csv found under {root}")


def read_cldf_table(path: Path) -> List[Dict[str, str]]:
    """Parse a CLDF CSV file into list-of-dicts.  Tolerates large files."""
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def index_languages(rows: Iterable[Dict[str, str]],
                    glottocodes: List[str]) -> Dict[str, List[Dict[str, str]]]:
    """Group language rows by Glottocode.  Returns dict mapping each
    target Glottocode to the (possibly multiple) matching rows.

    Some CLDF datasets (Lexibank, PHOIBLE) have several entries per
    Glottocode (different inventories or wordlists).  Caller decides
    how to aggregate.
    """
    targets = set(glottocodes)
    out: Dict[str, List[Dict[str, str]]] = {gc: [] for gc in glottocodes}
    for row in rows:
        gc = (row.get("Glottocode") or "").strip()
        if gc in targets:
            out[gc].append(row)
    return out
