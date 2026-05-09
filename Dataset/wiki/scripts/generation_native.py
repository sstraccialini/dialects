"""
Wikipedia preprocessing pipeline for Italo-Romance varieties — NATIVE TEXT.

This is the pre-aggressive-normalize pipeline (restored from git commit
2c641d9, Apr 2026).  It preserves diacritics, case, punctuation, digits,
and non-ASCII characters in the output CSVs.

Output: `Dataset/wiki/not_normalized/{dialects_in_both_OLDI_and_Flores,
others_dialects}/<code>.csv`.  The active aggressive-normalize variant
lives next to it under `Dataset/wiki/normalized/...` and is produced by
`generation.py`.

Why we keep both:
  - Surface-level methods (TF-IDF char/word, Word2Vec, FastText): use
    the normalized variant — diacritic + case + digit removal makes
    cross-variety n-gram overlap meaningful and matches what we do to
    FLORES/OLDI inputs.
  - Pretrained encoders (XLM-R, CANINE, Sentence-MiniLM, LaBSE): use
    the not_normalized variant — their tokenizers were pretrained on
    cased, accented, punctuated text, and lowercase-ASCII destroys
    sub-word identity.

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
  Stage 3. Article-level deduplication (keep=False — drop all duplicates).
  Stage 4. Sentence split with spaCy's rule-based sentencizer
           (`spacy.blank("xx") + sentencizer`), then merge fragments
           starting with lowercase into the previous sentence (fixes
           splits inside abbreviations like "s.p.a.", "a.C.", "D.O.C.G.").
  Stage 5. Sentence-level filters:
             - drop sentences shorter than 30 chars
             - drop sentences without any lowercase ASCII letter
             - drop sentences without a word starting with lowercase ASCII
             - drop sentences not ending with a real terminator
               (.!?")']”’»…) — excludes : and ; on purpose
             - per-variety filters (SUKI VEC patterns + Camposampiero
               LMO/VEC/PMS substring patterns)
  Stage 6. Sentence-level deduplication (keep="first" — softer than at
           article level, preserves legitimate repeated sentences).
  Stage 7. Auto prefix-based template dedup: drop sentences whose first
           30 characters appear ≥10 times (data-driven boilerplate catch).
  Stage 8. Fingerprint dedup: drop sentences whose fingerprint (digits→N,
           roman numerals→R, lowercased) appears ≥10 times — catches
           templates that vary only in numbers/years.
  Stage 9. Build per-article metadata + atomic save of <code>.csv,
           <code>_meta.csv, <code>_stats.json. Outputs are routed to
           `wiki/dialects_in_both_OLDI_and_Flores/` (Group A) or
           `wiki/others_dialects/` (Group B).

Per-variety filters:
  - VEC: SUKI French commune + SUKI Italian commune + SUKI Roman numbers
         (generalized to also match `par numari romani` and `MCMXV`-style
         year-page templates) + year-page template (`el xe on an del XX
         secolo` / `de el III secoło`) + day-of-year templates
         (`Ghe manca N dì par la fin de l'anno` and the symmetric
         `Par rivar al cao de l'an ghe vołe N dì`) + Camposampiero
         substrings (`el xe un comun de`, `gregorian`).
  - LMO: SUKI len<14 short-line filter + 24 Camposampiero substring
         patterns covering geographic / template stubs.
  - PMS: 11 Camposampiero substring patterns (Piemontese Wikipedia is
         heavily templated for municipal pages).
  - fur, lij, sc, scn, lld, nap, roa_tara: no per-variety filter — they
    rely on the general filters in Stage 5 plus the auto prefix-dedup
    (Stage 7) and fingerprint dedup (Stage 8) for boilerplate cleanup.

Choices that depart from SUKI for FLORES/OLDI consistency:
  - NO digit→1 substitution (FLORES/OLDI keep real digits).
  - NO ł→l normalization at the source (FLORES/OLDI veneto keep ł in
    82-83% of rows). When a downstream encoder needs `ł→l` (e.g. XLM-R
    subword), apply it at runtime via `Dataset.normalize.subword_safe`
    on BOTH training and eval inputs — see PIPELINE.md §12.

The script is meant to be invoked from the `_cache/` directory (set
by create.py), where the wikiextractor `<lang>_texts/` folders sit.
It writes its outputs to the two `wiki/` subfolders described in
Stage 9 above. The cache is preserved between runs so re-runs only
re-execute the cleaning stages.
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
# Variety registry — italo-romance varieties on Wikipedia.
#
# Group A (labels 0–5): the 6 varieties present in BOTH OLDI and FLORES,
#   our primary downstream training/eval set.
# Group B (labels 6–9): other italo-romance varieties that have a
#   Wikipedia edition but are missing from OLDI (and FLORES, except lld).
#   We process them with the same pipeline so they can be used as a
#   comparison set or for sanity checks against ITDI 2022.
# --------------------------------------------------------------------------- #
FOLD_LABEL = {
    # Group A: OLDI ∩ FLORES
    "fur_texts":      0,   # Friulian
    "lij_texts":      1,   # Ligurian
    "lmo_texts":      2,   # Lombard
    "sc_texts":       3,   # Sardinian
    "scn_texts":      4,   # Sicilian
    "vec_texts":      5,   # Venetian
    # Group B: other italo-romance varieties on Wikipedia
    "lld_texts":      6,   # Ladino (in FLORES but NOT in OLDI)
    "nap_texts":      7,   # Napoletano
    "pms_texts":      8,   # Piemontese
    "roa_tara_texts": 9,   # Tarantino (sub-variety of nap, separate Wiki edition)
    "eml_texts":     10,   # Emiliano-Romagnolo (added for edoardo/exp1_uriel_native)
}
DIAL_LABEL = {v: k.replace("_texts", "").upper() for k, v in FOLD_LABEL.items()}

# Group routing: where each variety's CSV ends up under wiki/.
GROUP_A = {"fur_texts", "lij_texts", "lmo_texts", "sc_texts", "scn_texts", "vec_texts"}
GROUP_B = {"lld_texts", "nap_texts", "pms_texts", "roa_tara_texts", "eml_texts"}


def _output_subdir_for(folder: str) -> Path:
    # Dataset/wiki/not_normalized/  — this script preserves native text
    # (diacritics, case, punctuation, digits).  The aggressive-normalized
    # counterpart lives at Dataset/wiki/normalized/, produced by
    # generation.py (which is the active extraction pipeline).
    base = Path(__file__).resolve().parents[1] / "not_normalized"
    if folder in GROUP_A:
        return base / "dialects_in_both_OLDI_and_Flores"
    else:
        return base / "others_dialects"


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
# Sentence terminators we accept as "this is a complete sentence". We
# deliberately exclude ":" and ";" — empirically a colon-ending fragment
# is almost always an intro to a list ("Persone inportanti che xe nate
# a Padoa:") or a heading-like incomplete fragment, not a real sentence.
SENTENCE_TERMINATORS = (".", "!", "?", '"', "'", "”", "’", "»", "…", ")", "]")


# --------------------------------------------------------------------------- #
# Per-variety patterns (SUKI + Camposampiero/ETHZ).
# --------------------------------------------------------------------------- #
# VEC — SUKI templates (regex).
VEC_FRENCH_COMMUNE = re.compile(
    r"el xe on comun de.*abitanti del departemento.*in Fransa\.",
    re.IGNORECASE,
)
VEC_ROMAN_NUMBERS = re.compile(
    r"\([MDCLXVI]+(?:\s+v\.C\.?)?\s+(?:en|in|par)\s+num[ae]ri\s+romani\)",
    re.IGNORECASE,
)
# VEC — year-page templates (TACL survey: ~1k pages for years 1 BC–999 BC).
# "El 1915 ... el xe un an del XX secoło"
VEC_YEAR_DESCRIPTION = re.compile(
    # Matches:  "el xe on an del XX sec..."  AND  "el xe on an de el III secoło v.C.."
    # The optional `(?:\s+\w+)?` allows for things like "an bisestile".
    # `an(?:n?o)?` matches "an" / "ano" / "anno" (all three attested forms).
    r"\bel\s+xe\s+(?:on|un)\s+an(?:n?o)?(?:\s+\w+)?\s+(?:del|de\s+el)\s+[IVX]+\s+sec",
    re.IGNORECASE,
)
# VEC — day-of-year templates (one per each of the 365 days).
# "Ghe manca 318 dì par la fin de l'anno (o 319 inte i anni bisestiłi)"
# The article body uses both `la` and `ła` (Venetian ł), and the noun
# is attested as `anno`, `ano`, and the apocopated `an`.
VEC_DAY_REMAINING = re.compile(
    r"ghe\s+manca\s+\d+\s+d[iì]\s+par\s+[lł][ae]?\s+fin\s+de\s+l['’]?\s*an(?:n?o)?\b",
    re.IGNORECASE,
)
# "Par rivar al cao de l'ano ghe vołe oncora 188 dì"
VEC_DAY_TO_YEAR_END = re.compile(
    r"par\s+rivar\s+al\s+cao\s+de\s+l['’]?\s*an[oó]?\s+ghe\s+vo[lł]+e",
    re.IGNORECASE,
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

# PMS — Camposampiero/ETHZ template substrings (carried over from the
# original Camposampiero generation.py). Piemontese Wikipedia is heavily
# templated for municipal pages.
PMS_CAMPOSAMPIERO_SUBSTRINGS = (
    "grup ëd popolassion.",
    "A confin-a con ",
    "a l'é na comun-a ëd",
    "con na densità",
    "A së stend",
    "As dëstend për",
    "a l'é na comun",
    "La lenga",
    "Në schema",
    "Ël sìndich a l'é",
    "a l'é un comun",
)


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
    if len(text) <= 30:
        return None
    if not HAS_LOWER_ASCII.search(text):
        return None
    if not HAS_WORD_LOWER.search(text):
        return None
    # Drop fragments that don't end with a real sentence terminator —
    # catches section titles ("Vangelo secondo Marco"), splitter glitches
    # ("Łe Nóve l'è zemełà co"), and the giant comune-list templates
    # that end with the last municipality name (5k+ chars, no period).
    if not text.rstrip().endswith(SENTENCE_TERMINATORS):
        return None
    code = DIAL_LABEL[label]
    if code == "VEC":
        if VEC_FRENCH_COMMUNE.search(text):
            return None
        if VEC_ITALIAN_COMMUNE.search(text):
            return None
        if VEC_ROMAN_NUMBERS.search(text):
            return None
        # Year and day-of-year placeholder pages (~2% of vec corpus).
        if VEC_YEAR_DESCRIPTION.search(text):
            return None
        if VEC_DAY_REMAINING.search(text):
            return None
        if VEC_DAY_TO_YEAR_END.search(text):
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
    elif code == "PMS":
        for pat in PMS_CAMPOSAMPIERO_SUBSTRINGS:
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
    # Routing: Group A → wiki/dialects_in_both_OLDI_and_Flores/,
    #          Group B → wiki/others_dialects/.
    # parents[1] of __file__ is Dataset/wiki/ regardless of cwd
    # (create.py runs us from the cache dir).
    out_dir = _output_subdir_for(folder)
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

    # --------------- Stage 4: sentence split (rule-based) ------------- #
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
        # First collect raw splits, then merge any sentence that starts
        # with a lowercase letter into the previous one — this fixes the
        # rule-based sentencizer's main failure mode (it splits on every
        # period, even inside abbreviations like "s.p.a.", "a.C.",
        # "D.O.C.G.", "es.", which produces fragments starting with
        # lowercase).
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
    stats["after_sentence_dedup"] = len(sent_df)
    print(f"  after sentence dedup:       {len(sent_df):>10,}")

    # --------------- Stage 7: auto prefix-based template dedup --------- #
    # Catches templated boilerplate that SUKI/Camposampiero patterns missed
    # (e.g. lmo's bot-generated French commune pages, scn's "Havi na
    # pupulazzioni" demographic stubs). Conservative thresholds:
    # PREFIX_LEN=30 + MIN_COUNT=10 → only drop when ≥10 sentences share
    # the same first 30 characters, which is statistically near-impossible
    # for natural sentences.
    PREFIX_LEN, MIN_COUNT = 30, 10
    from collections import Counter
    prefixes = Counter(t[:PREFIX_LEN].lower() for t in sent_df["text"])
    keep_mask = sent_df["text"].str[:PREFIX_LEN].str.lower().map(prefixes) < MIN_COUNT
    sent_df = sent_df[keep_mask].reset_index(drop=True)
    stats["after_prefix_dedup"] = len(sent_df)
    print(f"  after auto prefix-dedup:    {len(sent_df):>10,}")

    # --------------- Stage 8: fingerprint dedup ---------------------- #
    # Complementary to prefix-dedup: catches templates whose VARIABLE
    # part is in the middle/end of the sentence, like year-page stubs
    # ("El 64 v.C. (LXIV in numari romani) el xe on an ...") and
    # statistical templates ("L'abità el xe situà a 5 metri s.l.m.").
    # Fingerprint normalizes digits → "N" and roman numerals → "R",
    # so all sentences differing only in numbers collapse to one
    # fingerprint and get clustered. The CSV keeps the original
    # text — we only use the fingerprint for the dedup decision.
    NUM_RE = re.compile(r"\d+")
    ROMAN_RE = re.compile(r"\b[IVXLCDM]{2,}\b")
    def _fingerprint(t: str) -> str:
        # IMPORTANT: substitute roman numerals BEFORE lowercasing —
        # the regex character class is uppercase only, so applying it
        # after `t.lower()` silently matches nothing (year templates
        # like "(CCLIV v.C par numari romani)" survived because of this).
        t = ROMAN_RE.sub("R", t)
        t = NUM_RE.sub("N", t)
        return t.lower()
    fps = sent_df["text"].apply(_fingerprint)
    fp_counts = Counter(fps)
    keep_mask = fps.map(fp_counts) < MIN_COUNT
    sent_df = sent_df[keep_mask].reset_index(drop=True)
    stats["final_sentences"] = len(sent_df)
    stats["final_articles"] = int(sent_df["article_id"].nunique())
    print(f"  after fingerprint dedup:    {len(sent_df):>10,}")
    print(f"  final articles (with ≥1):   {stats['final_articles']:>10,}")

    # --------------- Stage 9: build meta + atomic save --------------- #
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
