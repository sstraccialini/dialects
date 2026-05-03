"""
Wikipedia preprocessing pipeline for Italo-Romance varieties.

Architecture: **Camposampiero/ETHZ-first** (article-level cleaning + spaCy
sentence splitting), with SUKI used as a selective addition for markup
regex sets and per-variety boilerplate patterns.

Why this architecture (and not line-level / SUKI-first):
  When we used SUKI's line-level cleanup BEFORE the spaCy splitter, the
  articles we fed to spaCy were "moncated" — entire wikiextractor lines
  were missing in the middle. The resulting sentences were short,
  broken, and incoherent ("frasi buttate a caso"). Camposampiero/ETHZ
  cleans the entire article text first and only then splits, producing
  far more natural sentences.

References:
  - Jauhiainen et al. 2022 (SUKI):
      https://aclanthology.org/2022.vardial-1.13/
  - Camposampiero et al. 2022 (ETHZ):
      https://aclanthology.org/2022.vardial-1.10/

Pipeline:
  Stage 1. Load each wikiextractor article (full text, not split into lines).
  Stage 2. Article-level cleaning:
             a. html.unescape (Camposampiero/ETHZ — entities)
             b. drop wiki headers ==...==
             c. SUKI markup regex sets 1+2 applied to the full article
             d. strip stray <, br>, residual entities
             e. whitespace collapse
             f. drop articles shorter than 50 chars (Camposampiero/ETHZ)
  Stage 3. Article-level deduplication.
  Stage 4. Sentence split with spaCy IT (it_core_news_sm) on the cleaned
           full-article texts.
  Stage 5. Sentence-level filters:
             - drop sentences shorter than 20 chars
             - drop sentences without any lowercase ASCII letter
             - drop sentences without a word starting with lowercase ASCII
             - per-variety filters (SUKI VEC patterns + Camposampiero
               LMO/VEC substring patterns)
  Stage 6. Sentence-level deduplication (keep="first").
  Stage 7. Save <code>.csv, <code>_meta.csv, <code>_stats.json (atomic).

Per-variety filters:
  - VEC: SUKI French commune + SUKI Italian commune + SUKI Roman numbers
         + Camposampiero "el xe un comun" + Camposampiero "gregorian".
  - LMO: 24 Camposampiero substring patterns + SUKI len<14 short-line filter.
  - fur, lij, sc, scn: no per-variety filter (no published patterns yet).

Choices that depart from SUKI for FLORES/OLDI consistency:
  - NO digit→1 substitution (FLORES/OLDI keep real digits).
  - NO ł→l normalization (FLORES/OLDI veneto keep ł in 82-83% of rows).

The script is meant to be invoked from `Dataset/wiki/`, where the
wikiextractor `<lang>_texts/` folders sit. It writes its outputs (and the
stats JSON) right next to them, then `create.py` cleans up the
intermediates.
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import spacy
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Variety registry — italo-romance varieties shared by FLORES + OLDI.
# --------------------------------------------------------------------------- #
FOLD_LABEL = {
    "fur_texts": 0,   # Friulian
    "lij_texts": 1,   # Ligurian
    "lmo_texts": 2,   # Lombard
    "sc_texts":  3,   # Sardinian
    "scn_texts": 4,   # Sicilian
    "vec_texts": 5,   # Venetian
}
DIAL_LABEL = {v: k.replace("_texts", "").upper() for k, v in FOLD_LABEL.items()}


# --------------------------------------------------------------------------- #
# Article-level cleaning regexes.
# --------------------------------------------------------------------------- #
# Wiki headers like ==Storia==, ===Sotto-titolo===, ...
HEADER = re.compile(r"=+\s*[^=]+\s*=+")

# SUKI markup set 1: drop the matched span. Applied at article level here.
SUKI_MARKUP_1 = re.compile(
    r"<comment>.*?</comment>"
    r"|<contributor>"
    r"|</contributor>"
    r"|<format>.*?</format>"
    r"|<ip>.*?</ip>"
    r"|<minor\s*/>"
    r"|<model>.*?</model>"
    r"|<ns[^>]*>[^<]*</ns>"
    r"|<parentid>.*?</parentid>"
    r"|<revision>"
    r"|<timestamp>.*?</timestamp>"
    r"|<username>.*?</username>"
)

# SUKI markup set 2: drop the matched tag.
SUKI_MARKUP_2 = re.compile(
    r"</math>"
    r"|</[pP]oem>"
    r"|</small>"
    r"|<references>"
    r"|</html>"
    r"|</includeonly></onlyinclude>"
    r"|</table>"
    r"|<\?php"
    r"|<BR\s+C[^>]*>"
    r"|#redirect",
    re.IGNORECASE,
)

# Stray brackets / tags left over after the SUKI sweeps.
STRAY = re.compile(r"<|br>")


# --------------------------------------------------------------------------- #
# Sentence-level filters.
# --------------------------------------------------------------------------- #
HAS_LOWER_ASCII = re.compile(r"[a-z]")
HAS_WORD_LOWER = re.compile(r"(?:^|\s)[a-z]")


# --------------------------------------------------------------------------- #
# Per-variety patterns (SUKI + Camposampiero/ETHZ).
# --------------------------------------------------------------------------- #
# VEC — SUKI templates (regex).
VEC_FRENCH_COMMUNE = re.compile(
    r"el xe on comun de.*abitanti del departemento.*in Fransa\.",
    re.IGNORECASE,
)
VEC_ROMAN_NUMBERS = re.compile(
    r"\(L?[IVXC]+\s+(?:en|in)\s+numeri\s+romani\)", re.IGNORECASE,
)
VEC_ITALIAN_COMMUNE = re.compile(
    r"el xe (?:on|un) comun italian de.*abitanti", re.IGNORECASE,
)
# VEC — Camposampiero/ETHZ extras (substring).
VEC_CAMPOSAMPIERO_SUBSTRINGS = ("el xe un comun de", "gregorian")

# LMO — Camposampiero/ETHZ substrings.
LMO_CAMPOSAMPIERO_SUBSTRINGS = (
    "La Stazzion de",
    "La Stazion de",
    "El cumün",
    "El cümü",
    "El cumün de",
    "El Distret",
    "El Passaport",
    "El cunfìna coi cümü",
    "km²",
    "Al gh'ha pressapoch abitant",
    "L'andament del numer de abitant",
    "L'andament del nömer dei abitàncc",
    "L'andamènt del nömer dei abitàncc",
    "a l'è una cità",
    "l'è 'na stazion de la",
    "l'è un cumün",
    "l'è un cümü",
    "l'è 'n cümü",
    "l'è menziunaa la prima volta",
    "L'è taccada a stazione di",
    "La a l'è 'na strada",
    "a l'è 'na ferrovia",
    "la se tróa a 'na",
    "e 'na densità de",
)
LMO_MIN_LEN = 14  # SUKI


# --------------------------------------------------------------------------- #
# Cleaning functions.
# --------------------------------------------------------------------------- #
def clean_article(text: str) -> str | None:
    """Apply article-level cleanup (Camposampiero/ETHZ + SUKI markup).

    Returns the cleaned article text, or None if it should be discarded.
    """
    if not text:
        return None
    # 1. Decode HTML entities (Camposampiero/ETHZ).
    text = html.unescape(text)
    # 2. Drop wiki headers.
    text = HEADER.sub(" ", text)
    # 3. SUKI markup regex sets — drop matched spans.
    text = SUKI_MARKUP_1.sub(" ", text)
    text = SUKI_MARKUP_2.sub(" ", text)
    # 4. Strip stray < and br>.
    text = STRAY.sub(" ", text)
    # 5. Whitespace normalize.
    text = re.sub(r"\s+", " ", text).strip()
    # 6. Length filter.
    if len(text) < 50:
        return None
    return text


def filter_sentence(text: str, label: int) -> str | None:
    """Apply sentence-level cleanup. Returns text or None if discarded."""
    if len(text) <= 20:
        return None
    if not HAS_LOWER_ASCII.search(text):
        return None
    if not HAS_WORD_LOWER.search(text):
        return None
    code = DIAL_LABEL[label]
    if code == "VEC":
        if VEC_FRENCH_COMMUNE.search(text):
            return None
        if VEC_ITALIAN_COMMUNE.search(text):
            return None
        if VEC_ROMAN_NUMBERS.search(text):
            return None
        lower = text.lower()
        for pat in VEC_CAMPOSAMPIERO_SUBSTRINGS:
            if pat in lower:
                return None
    elif code == "LMO":
        if len(text) < LMO_MIN_LEN:
            return None
        for pat in LMO_CAMPOSAMPIERO_SUBSTRINGS:
            if pat in text:
                return None
    return text


# --------------------------------------------------------------------------- #
# Sentence splitter — rule-based (loaded once).
#
# We use spaCy's rule-based sentencizer (split on .!?) instead of the
# Italian statistical parser (`it_core_news_sm`). The parser is trained on
# Italian standard and on dialectal text it makes systematic segmentation
# errors — sentences come out fragmented or merged in non-obvious ways.
# The rule-based sentencizer is deterministic, fast, and produces more
# faithful "sentences between full stops". It does miss some abbreviations
# (e.g. "Art. 5"), but our downstream length / lowercase-word filters
# catch most of the resulting fragments.
# --------------------------------------------------------------------------- #
print("[generation] loading spaCy rule-based sentencizer ...")
NLP = spacy.blank("xx")
NLP.add_pipe("sentencizer")


# --------------------------------------------------------------------------- #
# Pipeline per variety.
# --------------------------------------------------------------------------- #
def process_dialect(folder: str) -> dict | None:
    label = FOLD_LABEL[folder]
    code = DIAL_LABEL[label].lower()
    out_dir = Path("wiki_new")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_main = out_dir / f"{code}.csv"
    out_meta = out_dir / f"{code}_meta.csv"
    out_stats = out_dir / f"{code}_stats.json"

    if out_main.exists() and out_meta.exists() and out_stats.exists():
        print(f"[skip] {folder}: outputs already present")
        return None

    print(f"\n=== {folder} (label={label}, output={code}.csv) ===")

    aa_dir = Path(folder) / "AA"
    if not aa_dir.is_dir():
        print(f"  [warn] {aa_dir} not found, skip")
        return None

    # --------------- Stage 1: load articles --------------- #
    articles: list[dict] = []
    for fname in sorted(os.listdir(aa_dir)):
        with (aa_dir / fname).open("r", encoding="utf-8") as f:
            for jline in f:
                obj = json.loads(jline)
                text = obj.get("text") or ""
                if not text.strip():
                    continue
                articles.append({
                    "article_id": int(obj["id"]),
                    "url": obj.get("url", ""),
                    "title": obj.get("title", ""),
                    "text": text,
                })
    raw_n = len(articles)
    print(f"  raw articles loaded:        {raw_n:>10,}")
    if raw_n == 0:
        return None
    df = pd.DataFrame(articles)
    stats = {"raw_articles": raw_n}

    # --------------- Stage 2: article-level clean --------------- #
    df["text"] = df["text"].apply(clean_article)
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    stats["after_article_clean"] = len(df)
    print(f"  after article cleanup:      {len(df):>10,}")
    if len(df) == 0:
        return stats

    # --------------- Stage 3: article-level dedup --------------- #
    # Camposampiero-style aggressive dedup: drop ALL occurrences of any
    # duplicated article text (keep=False). Two full-article texts being
    # byte-identical is essentially always boilerplate (bot-generated
    # template pages, accidental dupes, redirects). At sentence level we
    # are softer (keep="first"), but at article level Camposampiero's
    # choice is sound.
    df = df.drop_duplicates(subset="text", keep=False).reset_index(drop=True)
    stats["after_article_dedup"] = len(df)
    print(f"  after article dedup:        {len(df):>10,}")

    # --------------- Stage 4: sentence split (spaCy IT) --------------- #
    sent_records: list[dict] = []
    texts = df["text"].tolist()
    article_ids = df["article_id"].tolist()
    titles = df["title"].tolist()
    urls = df["url"].tolist()

    for i, doc in enumerate(
        tqdm(
            NLP.pipe(texts, batch_size=512),
            total=len(texts),
            desc=f"  sentencizer {code}",
        )
    ):
        for sent in doc.sents:
            t = sent.text.strip()
            if not t:
                continue
            sent_records.append({
                "text": t,
                "label": label,
                "article_id": article_ids[i],
                "title": titles[i],
                "url": urls[i],
            })
    stats["raw_sentences"] = len(sent_records)
    print(f"  raw sentences from split:   {len(sent_records):>10,}")
    if not sent_records:
        return stats
    sent_df = pd.DataFrame(sent_records)

    # --------------- Stage 5: sentence-level filters --------------- #
    sent_df["text"] = sent_df["text"].apply(lambda t: filter_sentence(t, label))
    sent_df = sent_df.dropna(subset=["text"]).reset_index(drop=True)
    stats["after_sentence_filter"] = len(sent_df)
    print(f"  after sentence filter:      {len(sent_df):>10,}")

    # --------------- Stage 6: sentence-level dedup --------------- #
    sent_df = sent_df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    stats["final_sentences"] = len(sent_df)
    stats["final_articles"] = int(sent_df["article_id"].nunique())
    print(f"  final sentences (kept):     {len(sent_df):>10,}")
    print(f"  final articles (with ≥1):   {stats['final_articles']:>10,}")

    # --------------- Stage 7: build meta + atomic save --------------- #
    meta_df = (
        sent_df[["label", "article_id", "title", "url"]]
        .drop_duplicates(subset=["article_id"])
        .reset_index(drop=True)
        .copy()
    )
    counts = sent_df.groupby("article_id").size().reset_index(name="n_sentences")
    meta_df = meta_df.merge(counts, on="article_id", how="left")

    tmp_main = out_main.with_suffix(".csv.tmp")
    tmp_meta = out_meta.with_suffix(".csv.tmp")
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
    print("Camposampiero/ETHZ-first preprocessing for Italo-Romance Wikipedia dumps")
    print("=" * 75)

    folders = sorted(
        d for d in os.listdir(".")
        if os.path.isdir(d) and d in FOLD_LABEL
    )
    if not folders:
        print("No dialect '*_texts/' directory found in cwd.")
        print("Run create.py first (which calls wikiextractor).")
        sys.exit(1)
    print(f"Found dialects: {folders}\n")

    all_stats: dict[str, dict] = {}
    for folder in folders:
        s = process_dialect(folder)
        if s:
            all_stats[folder] = s

    if all_stats:
        print("\n" + "=" * 75)
        print("SUMMARY")
        print("=" * 75)
        cols = ["raw_articles", "after_article_clean", "after_article_dedup",
                "raw_sentences", "after_sentence_filter", "final_sentences"]
        header = f"{'dialect':<10}" + "".join(f"{c:>22}" for c in cols)
        print(header)
        for folder, s in all_stats.items():
            row = f"{folder:<10}" + "".join(
                f"{s.get(c, 0):>22,}" for c in cols
            )
            print(row)


if __name__ == "__main__":
    main()
