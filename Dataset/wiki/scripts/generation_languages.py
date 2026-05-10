"""
Wikipedia preprocessing pipeline for the 14 STANDARD comparison languages.

Mirrors generation.py architecturally (10 stages, same article-level cleaning,
same aggressive_normalize, same dedup) but tuned for high-resource languages:

  - MIN_LEN_PRE_LANG = 60 chars (vs 30 for dialects). Standards have
    abundant data, the 30-60 range is mostly fragments/captions.
  - No per-variety substring filters in Stage 7. Boilerplate templates
    (Italian "comune italiano", German "Gemeinde im", French "commune
    française", ...) on big wikis are a tiny minority and are caught by
    Stages 8-9 (sentence dedup + prefix-dedup) data-driven.
  - Hard cap of N_ARTICLES_CAP = 100,000 articles BEFORE Stage 2
    (random-sample with fixed seed). Without this cap, processing 4M
    Italian articles or 1M+ Spanish articles takes hours and uses 16+ GB
    RAM. With it: ~10-20 min and ~1.5 GB per language.

Output: Dataset/wiki/normalized/languages/<iso3>.csv

Invocation: same as generation.py — runs from `Dataset/wiki/_cache/` (where
wikiextractor's `*_texts/` folders sit). Picks up only folders matching
LANG_FOLDERS below.
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Reuse generic helpers from generation.py — DO NOT duplicate.  Import works
# because Dataset/wiki/scripts/ is the same directory.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from generation import (
    clean_article,
    aggressive_normalize,
    NLP,
    HAS_LOWER_ASCII, HAS_WORD_LOWER, SENTENCE_TERMINATORS,
)


# --------------------------------------------------------------------------- #
# Language registry — 14 standard comparison languages.
#
# Folder name (cwd of wikiextractor output) → (label, output csv stem).
# Labels 9-15 reuse the existing slots in generation.py; 17-23 are new.
# (16 is taken by eml in generation.py — skipped here.)
# --------------------------------------------------------------------------- #
LANG_FOLDERS = {
    "ita_texts":  (9,  "ita"),  # Italian
    "spa_texts":  (10, "spa"),  # Spanish
    "fra_texts":  (11, "fra"),  # French
    "eng_texts":  (12, "eng"),  # English
    "deu_texts":  (13, "deu"),  # German
    "cat_texts":  (14, "cat"),  # Catalan
    "slv_texts":  (15, "slv"),  # Slovenian
    "por_texts":  (17, "por"),  # Portuguese
    "ron_texts":  (18, "ron"),  # Romanian
    "oci_texts":  (19, "oci"),  # Occitan
    "glg_texts":  (20, "glg"),  # Galician
    "hrv_texts":  (21, "hrv"),  # Croatian
    "sqi_texts":  (22, "sqi"),  # Albanian (Tosk standard)
    "hun_texts":  (23, "hun"),  # Hungarian
}

OUT_DIR = SCRIPT_DIR.parent / "normalized" / "languages"


# --------------------------------------------------------------------------- #
# Tuning knobs — different from dialects.
# --------------------------------------------------------------------------- #
MIN_LEN_PRE_LANG = 60          # standards have abundant data → be selective
MAX_LEN_POST     = 500         # same as dialects (mega-list killer)
N_ARTICLES_CAP   = 100_000     # bound RAM regardless of dump size; matches
                               # the order of magnitude of dialect article
                               # counts (lmo 79k, vec 68k post-extraction)
RANDOM_SEED      = 42


# --------------------------------------------------------------------------- #
# Stage 5 — pre-normalize filter, NATIVE text, with stricter MIN_LEN.
# --------------------------------------------------------------------------- #
def pre_norm_filter_lang(text: str) -> str | None:
    if len(text) < MIN_LEN_PRE_LANG:
        return None
    if not HAS_LOWER_ASCII.search(text):
        return None
    if not HAS_WORD_LOWER.search(text):
        return None
    if not text.rstrip().endswith(SENTENCE_TERMINATORS):
        return None
    return text


# --------------------------------------------------------------------------- #
# Stage 7 — post-normalize length filter (no per-variety substrings).
# --------------------------------------------------------------------------- #
def post_norm_filter_lang(text: str) -> str | None:
    if len(text) > MAX_LEN_POST:
        return None
    return text


# --------------------------------------------------------------------------- #
# Per-language pipeline.
# --------------------------------------------------------------------------- #
def process_language(folder: str) -> dict | None:
    label, code = LANG_FOLDERS[folder]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_main  = OUT_DIR / f"{code}.csv"
    out_meta  = OUT_DIR / f"{code}_meta.csv"
    out_stats = OUT_DIR / f"{code}_stats.json"

    if out_main.exists() and out_meta.exists() and out_stats.exists():
        print(f"[skip] {folder}: outputs already present")
        return None

    print(f"\n=== {folder} (label={label}, output={code}.csv) ===")

    folder_path = Path(folder)
    if not folder_path.is_dir():
        print(f"  [warn] {folder_path} not found, skip")
        return None

    # Wikiextractor output is sharded into AA/, AB/, AC/, ..., BF/, ...
    # Each subdir holds up to 100 files of ~1MB.  Big wikis (it/de/en/...)
    # span many subdirs.  Iterate ALL of them, not just AA.
    subdirs = sorted(d for d in folder_path.iterdir() if d.is_dir())
    if not subdirs:
        print(f"  [warn] no subdirs in {folder_path}, skip")
        return None

    # ----- Stage 1: load all articles from every subdir ----- #
    articles: list[dict] = []
    for sd in subdirs:
        for fname in sorted(os.listdir(sd)):
            with (sd / fname).open("r", encoding="utf-8") as f:
                for jline in f:
                    obj = json.loads(jline)
                    t = obj.get("text") or ""
                    if not t.strip():
                        continue
                    articles.append({
                        "article_id": int(obj["id"]),
                        "url": obj.get("url", ""),
                        "title": obj.get("title", ""),
                        "text": t,
                    })
    raw_n = len(articles)
    print(f"  raw articles loaded:        {raw_n:>10,}")
    if raw_n == 0:
        return None

    # ----- Random cap (the standards-specific knob) ----- #
    if raw_n > N_ARTICLES_CAP:
        rng = random.Random(RANDOM_SEED)
        articles = rng.sample(articles, N_ARTICLES_CAP)
        print(f"  random-cap to:              {N_ARTICLES_CAP:>10,}")
    df = pd.DataFrame(articles)
    stats = {"raw_articles": raw_n, "after_cap": len(df),
             "random_seed": RANDOM_SEED}

    # ----- Stage 2: article-level clean ----- #
    df["text"] = df["text"].apply(clean_article)
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    stats["after_article_clean"] = len(df)
    print(f"  after article cleanup:      {len(df):>10,}")
    if len(df) == 0:
        return stats

    # ----- Stage 3: article-level dedup ----- #
    df = df.drop_duplicates(subset="text", keep=False).reset_index(drop=True)
    stats["after_article_dedup"] = len(df)
    print(f"  after article dedup:        {len(df):>10,}")

    # ----- Stage 4: sentence split + lowercase-merge ----- #
    sent_records: list[dict] = []
    texts = df["text"].tolist()
    article_ids = df["article_id"].tolist()
    titles = df["title"].tolist()
    urls = df["url"].tolist()

    for i, doc in enumerate(
        tqdm(NLP.pipe(texts, batch_size=512),
             total=len(texts), desc=f"  sentencizer {code}")
    ):
        article_sents: list[str] = []
        for sent in doc.sents:
            t = sent.text.strip()
            if not t:
                continue
            if article_sents and t[:1].islower():
                article_sents[-1] = article_sents[-1] + " " + t
            else:
                article_sents.append(t)
        for t in article_sents:
            sent_records.append({
                "text": t, "label": label,
                "article_id": article_ids[i],
                "title": titles[i], "url": urls[i],
            })
    stats["raw_sentences"] = len(sent_records)
    print(f"  raw sentences from split:   {len(sent_records):>10,}")
    if not sent_records:
        return stats
    sent_df = pd.DataFrame(sent_records)

    # ----- Stage 5: pre-normalize filter (NATIVE, MIN_LEN_PRE_LANG=60) ----- #
    sent_df["text"] = sent_df["text"].apply(pre_norm_filter_lang)
    sent_df = sent_df.dropna(subset=["text"]).reset_index(drop=True)
    stats["after_pre_norm_filter"] = len(sent_df)
    print(f"  after pre-norm filter:      {len(sent_df):>10,}")
    if len(sent_df) == 0:
        return stats

    # ----- Stage 6: aggressive normalize ----- #
    sent_df["text"] = sent_df["text"].apply(aggressive_normalize)
    sent_df = sent_df[sent_df["text"].str.len() > 0].reset_index(drop=True)
    stats["after_normalize"] = len(sent_df)

    # ----- Stage 7: post-normalize length filter (no per-variety substrings) ----- #
    sent_df["text"] = sent_df["text"].apply(post_norm_filter_lang)
    sent_df = sent_df.dropna(subset=["text"]).reset_index(drop=True)
    stats["after_post_norm_filter"] = len(sent_df)
    print(f"  after post-norm filter:     {len(sent_df):>10,}")

    # ----- Stage 8: sentence dedup ----- #
    sent_df = sent_df.drop_duplicates(subset="text", keep="first")\
                     .reset_index(drop=True)
    stats["after_sentence_dedup"] = len(sent_df)
    print(f"  after sentence dedup:       {len(sent_df):>10,}")

    # ----- Stage 9: auto prefix-dedup PL=30, K=10 ----- #
    PREFIX_LEN, MIN_COUNT = 30, 10
    prefixes = Counter(t[:PREFIX_LEN] for t in sent_df["text"])
    keep_mask = sent_df["text"].str[:PREFIX_LEN].map(prefixes) < MIN_COUNT
    sent_df = sent_df[keep_mask].reset_index(drop=True)
    stats["after_prefix_dedup"] = len(sent_df)
    stats["final_sentences"] = len(sent_df)
    stats["final_articles"] = int(sent_df["article_id"].nunique())
    print(f"  after auto prefix-dedup:    {len(sent_df):>10,}")
    print(f"  final articles (with >=1):  {stats['final_articles']:>10,}")

    # ----- Stage 10: build meta + atomic save ----- #
    meta_df = (
        sent_df[["label", "article_id", "title", "url"]]
        .drop_duplicates(subset=["article_id"]).reset_index(drop=True).copy()
    )
    counts = sent_df.groupby("article_id").size().reset_index(name="n_sentences")
    meta_df = meta_df.merge(counts, on="article_id", how="left")

    tmp_main  = out_main.with_suffix(".csv.tmp")
    tmp_meta  = out_meta.with_suffix(".csv.tmp")
    tmp_stats = out_stats.with_suffix(".json.tmp")
    sent_df[["text", "label", "article_id"]].to_csv(tmp_main, index=False)
    meta_df[["article_id", "title", "url", "n_sentences"]].to_csv(tmp_meta, index=False)
    with tmp_stats.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    tmp_main.replace(out_main)
    tmp_meta.replace(out_meta)
    tmp_stats.replace(out_stats)

    print(f"  -> {out_main}  ({len(sent_df):,} sentences)")
    print(f"  -> {out_meta}  ({len(meta_df):,} articles)")
    print(f"  -> {out_stats}")
    return stats


# --------------------------------------------------------------------------- #
# CLI entry point.
# --------------------------------------------------------------------------- #
def main():
    print("Standard-language preprocessing pipeline (Wikipedia, normalized)")
    print("=" * 75)
    print(f"  MIN_LEN_PRE_LANG    = {MIN_LEN_PRE_LANG}")
    print(f"  MAX_LEN_POST        = {MAX_LEN_POST}")
    print(f"  N_ARTICLES_CAP      = {N_ARTICLES_CAP:,}")
    print(f"  RANDOM_SEED         = {RANDOM_SEED}")
    print(f"  per-variety filters = NONE (rely on Stages 8-9 dedup)")
    print(f"  output dir          = {OUT_DIR}")
    print()
    folders = sorted(d for d in os.listdir(".")
                     if os.path.isdir(d) and d in LANG_FOLDERS)
    if not folders:
        print("No '*_texts/' directory in cwd matching the language registry.")
        print("Run wikiextractor first on the downloaded dumps.")
        sys.exit(1)
    print(f"Found languages: {folders}\n")

    all_stats: dict[str, dict] = {}
    for folder in folders:
        s = process_language(folder)
        if s:
            all_stats[folder] = s

    if all_stats:
        print("\n" + "=" * 75)
        print("SUMMARY")
        print("=" * 75)
        cols = ["raw_articles", "after_cap", "after_article_dedup",
                "raw_sentences", "after_pre_norm_filter",
                "after_post_norm_filter", "final_sentences"]
        header = f"{'lang':<12}" + "".join(f"{c:>22}" for c in cols)
        print(header)
        for folder, s in all_stats.items():
            row = f"{folder:<12}" + "".join(
                f"{s.get(c, 0):>22,}" for c in cols
            )
            print(row)


if __name__ == "__main__":
    main()
