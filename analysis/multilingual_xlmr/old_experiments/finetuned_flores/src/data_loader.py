"""
Data loading for Task 3 fine-tuning experiments.

Three data sources with distinct roles:
  FLORES+  — evaluation only (never touches training)
  Wiki     — MLM continued pretraining (monolingual dialect text)
  OLDI     — TLM alignment training (Italian ↔ dialect parallel pairs)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from config import (
    FLORES_DIR,
    WIKI_DIR,
    WIKI_VARIETY_DIR,
    OLDI_DIR,
    VARIETY_CODES,
    SAMPLE_SIZE,
    RANDOM_STATE,
    WIKI_CODES,
    OLDI_VARIETIES,
    MAX_WIKI_SAMPLES,
)


# ── FLORES+ (evaluation) ────────────────────────────────────────────────────

def _read_txt(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_all_flores(
    codes: List[str] = VARIETY_CODES,
    sample_size: int = SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    data: Dict[str, List[str]] = {}
    rows = []
    for slug in codes:
        sents = _read_txt(FLORES_DIR / f"{slug}.txt")
        n_available = len(sents)
        if n_available <= sample_size:
            sampled = sents
        else:
            sampled = pd.Series(sents).sample(n=sample_size, random_state=random_state).tolist()
        data[slug] = sampled
        rows.append({"code": slug, "n_available": n_available, "n_used": len(sampled)})
    return data, pd.DataFrame(rows)


def iter_labeled_sentences(
    data: Dict[str, List[str]],
    codes: List[str] = VARIETY_CODES,
) -> Tuple[List[str], List[str]]:
    sents, slugs = [], []
    for slug in codes:
        if slug not in data:
            continue
        for s in data[slug]:
            sents.append(s)
            slugs.append(slug)
    return sents, slugs


# ── Wikipedia (MLM training) ────────────────────────────────────────────────

def load_wiki_texts(
    varieties: List[str] = None,
    max_per_lang: int = MAX_WIKI_SAMPLES,
    random_state: int = RANDOM_STATE,
    min_length: int = 20,
) -> List[str]:
    """Return a flat list of sentences from Wikipedia CSVs for the given varieties."""
    if varieties is None:
        varieties = list(WIKI_CODES.keys())

    all_texts: List[str] = []
    for variety in varieties:
        code = WIKI_CODES.get(variety)
        if code is None:
            continue
        # Look up the correct subfolder via WIKI_VARIETY_DIR mapping.
        subdir = WIKI_VARIETY_DIR.get(code)
        if subdir is None:
            print(f"  [wiki] WARNING: no subfolder mapping for code '{code}' — skipping {variety}")
            continue
        csv_path = subdir / f"{code}.csv"
        if not csv_path.exists():
            print(f"  [wiki] WARNING: {csv_path} not found — skipping {variety}")
            continue

        df = pd.read_csv(csv_path, usecols=["text"], dtype=str, on_bad_lines="skip")
        texts = [t.strip() for t in df["text"].dropna() if len(str(t).strip()) >= min_length]

        if len(texts) > max_per_lang:
            texts = (
                pd.Series(texts)
                .sample(n=max_per_lang, random_state=random_state)
                .tolist()
            )

        print(f"  [wiki] {variety:12s} ({code}): {len(texts):>6d} sentences")
        all_texts.extend(texts)

    return all_texts


# ── OLDI parallel pairs (TLM training) ──────────────────────────────────────

def load_oldi_pairs(variety: str) -> List[Tuple[str, str]]:
    """Load (italian, dialect) sentence pairs for one variety."""
    tsv_path = OLDI_DIR / f"pairs_ita_{variety}.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(f"OLDI pairs file not found: {tsv_path}")

    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    pairs = []
    for _, row in df.iterrows():
        ita = str(row.get("italiano", "")).strip()
        dial = str(row.get(variety, "")).strip()
        if ita and dial and ita != "nan" and dial != "nan":
            pairs.append((ita, dial))
    return pairs


def load_all_oldi_pairs(varieties: List[str] = None) -> List[Tuple[str, str]]:
    if varieties is None:
        varieties = OLDI_VARIETIES
    all_pairs: List[Tuple[str, str]] = []
    for variety in varieties:
        pairs = load_oldi_pairs(variety)
        print(f"  [oldi] {variety:12s}: {len(pairs):>6d} pairs")
        all_pairs.extend(pairs)
    return all_pairs
