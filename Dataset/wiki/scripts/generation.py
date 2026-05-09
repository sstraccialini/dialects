"""
Wikipedia preprocessing pipeline for Italo-Romance varieties + comparison
languages, with **aggressive normalization** as part of the pipeline.

Architecture: Camposampiero/ETHZ-first (article-level cleaning + spaCy
sentence splitting) + SUKI-style markup regexes, plus an explicit
**aggressive-normalize** step that strips digits, punctuation, symbols,
diacritics, and case to produce lowercase-ASCII-only text. This makes
the output directly comparable across languages for char-n-gram /
bag-of-features methods (TF-IDF, FastText, Word2Vec, ...).

References:
  - Jauhiainen et al. 2022 (SUKI):
      https://aclanthology.org/2022.vardial-1.13/
  - Camposampiero et al. 2022 (ETHZ):
      https://aclanthology.org/2022.vardial-1.10/

Pipeline (10 stages):
  Stage 1.  Load articles from wikiextractor JSON.
  Stage 2.  Article-level cleaning:
              a. html.unescape (HTML entities)
              b. drop wiki headers ==...==
              c. SUKI markup regex sets 1+2
              d. strip stray <, br>
              e. whitespace collapse
              f. drop articles shorter than 50 chars
  Stage 3.  Article-level dedup (keep=False — drop all duplicates).
  Stage 4.  Sentence split (spaCy rule-based sentencizer) + lowercase-merge
            (fixes splits inside abbreviations like "s.p.a.", "a.C.",
            "D.O.C.G.").
  Stage 5.  PRE-NORMALIZE filters (operate on NATIVE text):
              - len(text) >= 30
              - HAS_LOWER_ASCII (catches all-caps fragments)
              - HAS_WORD_LOWER (catches title-case fragments)
              - endswith(SENTENCE_TERMINATORS) — catches headings, splitter
                glitches, mega-list templates
  Stage 6.  AGGRESSIVE NORMALIZE: lowercase + strip diacritics + strip
            digits + strip romans + strip punctuation/symbols + collapse
            whitespace.
  Stage 7.  POST-NORMALIZE filters (operate on AGGRESSIVE text):
              - len(text) <= 500 (kill mega-list templates >500 chars)
              - per-variety lowercase substring matches (VEC, LMO, PMS).
                The previous regex-based per-variety patterns have been
                rewritten as substring lookups on aggressive text.
  Stage 8.  Sentence-level dedup keep="first" on aggressive text. Catches
            case/punctuation/diacritic variants that were distinct in
            native form.
  Stage 9.  Auto prefix-dedup PL=30, K=10 on aggressive text.
            (The old fingerprint dedup is REMOVED — on aggressive text
            digits and romans are already stripped, so it would be a
            no-op equivalent to Stage 8.)
  Stage 10. Save <code>.csv (single normalized text column),
            <code>_meta.csv, <code>_stats.json (atomic).

Per-variety substring lists (Stage 7), all in lowercase ASCII space:
  - VEC: 6 single-substring drops + 5 multi-substring (AND) drops covering
         French commune templates, Italian commune templates, Roman-numbers
         stubs, year-page stubs, day-of-year templates, Camposampiero
         comun-substrings, gregorian calendar stubs.
  - LMO: 22 substrings (was 24 — `km²` removed because aggressive collapses
         it to too-generic `km`; `L'andament` and `L'andamènt` collapse
         to the same `l andament` post-normalize).
  - PMS: 11 substrings (Camposampiero municipal templates).
  - All other dialects (fur, lij, sc, scn, lld, nap) and comparison
    languages (ita, spa, fra, eng, deu, cat, slv): no per-variety
    filter — rely on generic filters + Stages 8-9 dedup.

Routing (Stage 10):
  - GROUP_A dialects → wiki/dialects_in_both_OLDI_and_Flores/
  - GROUP_B dialects → wiki/others_dialects/
  - GROUP_LANGUAGES (comparison) → wiki/languages/

Choices that depart from SUKI for FLORES/OLDI consistency:
  - We DO substitute digits with " " (was: NO digit substitution). With
    the project's task being similarity (TF-IDF / FastText), digits are
    cross-language noise. Parallel normalize scripts apply the same to
    FLORES/OLDI in `Dataset/flores/scripts/normalize.py` and
    `Dataset/oldi/scripts/normalize.py`.
  - We DO collapse `ł → l` (and other letter-mappings ß→ss, æ→ae, ...) at
    extraction time. The non-normalized FLORES/OLDI files are preserved
    under `not_normalized/` for reference.

The script is meant to be invoked from the `_cache/` directory (set by
create.py), where the wikiextractor `<lang>_texts/` folders sit. It writes
its outputs to the appropriate subfolder under `Dataset/wiki/`. The cache
is preserved between runs so re-runs only re-execute the cleaning stages.
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
import spacy
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Variety registry — italo-romance varieties + comparison languages.
#
# Group A (labels 0-5): the 6 varieties in BOTH OLDI and FLORES — primary set.
# Group B (labels 6-9): other italo-romance varieties on Wikipedia.
# Comparison languages (labels 10-16): contemporary languages used as
#   reference points. Their dump URLs are commented in create.py — uncomment
#   when ready to re-extract.
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
    "eml_texts":     16,   # Emiliano-Romagnolo (added for edoardo/exp1_uriel_native)
    # Comparison languages (re-extract pending — see create.py)
    "ita_texts":      9,   # Italian
    "spa_texts":     10,   # Spanish
    "fra_texts":     11,   # French
    "eng_texts":     12,   # English
    "deu_texts":     13,   # German
    "cat_texts":     14,   # Catalan
    "slv_texts":     15,   # Slovenian
}
DIAL_LABEL = {v: k.replace("_texts", "").upper() for k, v in FOLD_LABEL.items()}

# Group routing: where each variety's CSV ends up under wiki/.
GROUP_A = {"fur_texts", "lij_texts", "lmo_texts",
           "sc_texts", "scn_texts", "vec_texts"}
GROUP_B = {"lld_texts", "nap_texts", "pms_texts", "eml_texts"}
GROUP_LANGUAGES = {"ita_texts", "spa_texts", "fra_texts", "eng_texts",
                   "deu_texts", "cat_texts", "slv_texts"}


def _output_subdir_for(folder: str) -> Path:
    # Dataset/wiki/normalized/  — this script applies aggressive normalize
    # (Stage 6).  The native-text counterpart lives at Dataset/wiki/not_normalized/
    # produced by generation_native.py.
    base = Path(__file__).resolve().parents[1] / "normalized"
    if folder in GROUP_A:
        return base / "dialects_in_both_OLDI_and_Flores"
    if folder in GROUP_B:
        return base / "others_dialects"
    if folder in GROUP_LANGUAGES:
        return base / "languages"
    raise ValueError(f"Unknown folder routing for {folder!r}")


# --------------------------------------------------------------------------- #
# Article-level cleaning regexes (Stage 2).
# --------------------------------------------------------------------------- #
HEADER = re.compile(r"=+\s*[^=]+\s*=+")

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

STRAY = re.compile(r"<|br>")


# --------------------------------------------------------------------------- #
# Pre-normalize filter constants (Stage 5, operate on NATIVE text).
# --------------------------------------------------------------------------- #
MIN_LEN_PRE = 30

HAS_LOWER_ASCII = re.compile(r"[a-z]")
HAS_WORD_LOWER = re.compile(r"(?:^|\s)[a-z]")

# Sentence terminators we accept as "this is a complete sentence".
# Excludes ":" and ";" — empirically those end fragments/intros, not real
# sentences.
SENTENCE_TERMINATORS = (".", "!", "?", '"', "'", "”", "’", "»", "…", ")", "]")


# --------------------------------------------------------------------------- #
# Aggressive normalization (Stage 6).
# --------------------------------------------------------------------------- #
# Explicit char mapping for letters that NFD does NOT decompose to ASCII.
_EXPLICIT_MAP = str.maketrans({
    "ß": "ss", "ł": "l", "Ł": "L",
    "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE",
    "ø": "o", "Ø": "O",
    "đ": "d", "Đ": "D",
    "ð": "d", "Ð": "D",
    "þ": "th", "Þ": "TH",
})

_DIACRITICS = re.compile(r"[̀-ͯ]")     # combining marks (NFD)
_DIGITS = re.compile(r"\d+")
_ROMAN_UPPER = re.compile(r"\b[IVXLCDM]{2,}\b")  # only uppercase, before lowercase()
_NON_LATIN_LOWER = re.compile(r"[^a-z\s]")       # anything not lowercase Latin or whitespace
_SPACES = re.compile(r"\s+")


def aggressive_normalize(text: str) -> str:
    """Aggressive normalize → lowercase ASCII letters + single spaces only.

    Pipeline:
      1. NFC + explicit mapping for non-decomposable letters (ß, ł, æ, œ, ...)
      2. Strip uppercase Roman numerals (regex needs uppercase, must run before lower())
      3. Lowercase everything
      4. NFD + strip combining diacritics (à è é ñ ç ü ä → a e e n c u a)
      5. Strip digits
      6. Strip everything not [a-z\\s] (punctuation, symbols, residual non-Latin)
      7. Collapse whitespace
    """
    if not isinstance(text, str):
        return text
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_EXPLICIT_MAP)
    text = _ROMAN_UPPER.sub(" ", text)
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = _DIACRITICS.sub("", text)
    text = _DIGITS.sub(" ", text)
    text = _NON_LATIN_LOWER.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


# --------------------------------------------------------------------------- #
# Post-normalize filter constants (Stage 7, operate on AGGRESSIVE text).
# --------------------------------------------------------------------------- #
MAX_LEN_POST = 500

# All substrings below are LOWERCASE ASCII (for matching against aggressive
# text). They are the post-normalize equivalents of the previous regex /
# substring patterns we used on native text.

# VEC: drop sentences containing any of these substrings (single match).
VEC_DROP_SUBSTRINGS = (
    "numari romani",         # was VEC_ROMAN_NUMBERS regex, post-aggressive substring
    "numeri romani",         # alternate Italian-influenced spelling
    "par rivar al cao",      # day-to-year-end template
    "el xe un comun de",     # Camposampiero generic-comun
    "el xe on comun de",     # alternate "on" form (case-folded same)
    "gregorian",             # calendar stub (Camposampiero)
    "xe na stasion feroviaria",  # train station infobox stub (~10 templates)
    "comun italian de aneme",     # template variant using "aneme" instead of "abitanti"
    "comune de abitanti inte",    # ~525 hits — Italian commune VEC variant with "comunE" (not "comun")
)

# VEC: drop only if ALL substrings in the tuple appear (AND match).
# Captures multi-piece templates whose specificity comes from co-occurring
# fragments rather than a single distinctive substring.
VEC_DROP_AND = (
    ("abitanti del departemento", "in fransa"),    # was VEC_FRENCH_COMMUNE
    ("abitanti", "departemento", "fransa"),        # broader: catches "abitanti inte el departemento... ile de france" variants
    ("comune fransese", "departemento"),           # broader: catches "el xe on comune fransese de X inte el departemento" form
    ("comun italian", "abitanti"),                 # broadened from ("comun italian de", "abitanti")
                                                    # — catches "comun italian de abitanti" + "comun italian colcerexa de abitanti"
                                                    # + "comun italian inte la rejon ... el conta abitanti" + "comun italian de aneme" forms
    ("ghe manca", "fin de l"),                     # was VEC_DAY_REMAINING (apostrophe form)
    ("ghe manca", "fin del"),                      # broader: catches "fin del an/ano" form (no apostrophe in source)
    ("el xe on an", "secolo"),                     # was VEC_YEAR_DESCRIPTION
    ("el xe un an", "secolo"),                     # alternate "un" form
)

# LMO: drop sentences containing any of these substrings.
LMO_DROP_SUBSTRINGS = (
    "la stazzion de",
    "la stazion de",
    "el cumun",
    "el cumu",                              # was "El cümü"  → ü→u
    "el cumun de",
    "el distret",
    "el passaport",
    "el cunfina coi cumu",                  # "cunfìna coi cümü"
    "al gh ha pressapoch",                  # broadened from "...abitant" — catches "...d abitant", "...miliun de abitant" variants
    "l andament del numer de abitant",
    "l andament del nomer dei abitancc",    # "nömer" + "abitàncc"
    "a l e una cita",                       # "a l'è una cità"
    "l e na stazion de la",                 # "l'è 'na stazion de la"
    "l e un cumun",
    "l e un cumu",
    "l e n cumu",
    "l e menziunaa la prima volta",
    "l e taccada a stazione di",
    "la a l e na strada",
    "a l e na ferrovia",
    "la se troa a na",                      # "tróa" → "troa"
    "e na densita de",                      # "densità"
    "el paes el se trova al cunfin",        # geographic stub (~8 hits)
    "altezza de meter sora el nivell del mar",  # altitude stub (~8 hits)
    "a l e una strada statal longa",        # road stub (~8 hits)
    "citadina todesca del stat federal",    # ~593 hits — German town stubs
    "comun de la croazzia",                 # ~510 hits — Croatian commune stubs
    "statunitens representant",             # ~251 hits — US politicians stubs
    "austriegh del land",                   # ~248 hits — Austrian regional stubs
    # Note: "km²" was in the native list but post-aggressive collapses to
    # "km" which is too generic — removed to avoid false positives.
)

# PMS: drop sentences containing any of these substrings.
PMS_DROP_SUBSTRINGS = (
    "grup ed popolassion",                  # "ëd"
    "a confin a con",                       # hyphen in "confin-a"
    "a l e na comun a ed",                  # "a l'é na comun-a ëd"
    "con na densita",                       # "densità"
    "a se stend",                           # "A së stend"
    "as destend per",                       # "As dëstend për"
    "a l e na comun",
    "la lenga",
    "ne schema",                            # "Në schema"
    "el sindich a l e",                     # "Ël sìndich a l'é"
    "a l e un comun",
)

# LLD: drop sentences containing any of these substrings.
# Ladin Wikipedia has massive auto-generated stub templates with the
# variable name at sentence start (so prefix-dedup misses them):
#   - "X ie n luech te [country]"           (place stubs)
#   - "X ie n videojuech svilupa..."        (video game stubs)
#   - "X ie n chemun de la [country] te..." (municipality stubs)
LLD_DROP_SUBSTRINGS = (
    "ie n luech te",                        # ~42k place stubs (~20% of lld!)
    "ie n videojuech svilupa",              # ~11k video-game stubs
    "ie n chemun de la",                    # ~15k municipality stubs (Spain/France/etc.)
)

# NAP: drop sentences containing any of these substrings.
# Italian comune templates plus residual table HTML markup.
NAP_DROP_SUBSTRINGS = (
    "colspan",                              # ~7 HTML table residue
    "e nu comune e",                        # ~3,400 hits — Italian comune nap form (catches both "abitante" and "crestiane" variants)
)

# PMS, FUR, LIJ, SC, SCN: per-variety substring filters.
PMS_EXTRA_DROP_SUBSTRINGS = (
    "parochia che a aparten al munissipi",  # ~1,700 hits — Galician parish stubs
)

FUR_DROP_SUBSTRINGS = (
    "comun di abitants logat tal",          # ~207 hits — Italian commune fur form
)

LIJ_DROP_SUBSTRINGS = (
    "comune inta provinsa de",              # ~1,300 hits — Italian commune lij form
    "comune inta cittae metropolitann",     # ~561 hits — metropolitan city
)

SC_DROP_SUBSTRINGS = (
    "comunu de bividores",                  # ~136 hits — Sardinian commune
    "comuna ispagnola de abitantes",        # ~92 hits — Spanish comuna
)

SCN_DROP_SUBSTRINGS = (
    "cumuni di abbitanti da pruvincia",     # ~510 hits — Sicilian commune
    "cumuni talianu da pruvincia",          # ~313 hits — Italian commune scn form
    "havi na pupulazzioni di abbita",       # ~9 hits — population stub
)


# --------------------------------------------------------------------------- #
# Stage 2 — article-level cleaning.
# --------------------------------------------------------------------------- #
def clean_article(text: str) -> str | None:
    """Apply article-level cleanup (HTML + SUKI markup + length filter).

    Returns the cleaned article text, or None if it should be discarded.
    """
    if not text:
        return None
    text = html.unescape(text)
    text = HEADER.sub(" ", text)
    text = SUKI_MARKUP_1.sub(" ", text)
    text = SUKI_MARKUP_2.sub(" ", text)
    text = STRAY.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 50:
        return None
    return text


# --------------------------------------------------------------------------- #
# Stage 5 — pre-normalize filters (NATIVE text).
# --------------------------------------------------------------------------- #
def pre_norm_filter(text: str, folder: str) -> str | None:
    """Generic filters that need NATIVE text (punctuation/case/digits)."""
    if len(text) < MIN_LEN_PRE:
        return None
    if not HAS_LOWER_ASCII.search(text):
        return None
    if not HAS_WORD_LOWER.search(text):
        return None
    if not text.rstrip().endswith(SENTENCE_TERMINATORS):
        return None
    return text


# --------------------------------------------------------------------------- #
# Stage 7 — post-normalize filters (AGGRESSIVE text).
# --------------------------------------------------------------------------- #
def post_norm_filter(text: str, label: int, folder: str) -> str | None:
    """Length max + per-variety substring drops on aggressive text."""
    if len(text) > MAX_LEN_POST:
        return None
    code = DIAL_LABEL[label]
    if code == "VEC":
        for sub in VEC_DROP_SUBSTRINGS:
            if sub in text:
                return None
        for and_pair in VEC_DROP_AND:
            if all(s in text for s in and_pair):
                return None
    elif code == "LMO":
        for sub in LMO_DROP_SUBSTRINGS:
            if sub in text:
                return None
    elif code == "PMS":
        for sub in PMS_DROP_SUBSTRINGS:
            if sub in text:
                return None
        for sub in PMS_EXTRA_DROP_SUBSTRINGS:
            if sub in text:
                return None
    elif code == "LLD":
        for sub in LLD_DROP_SUBSTRINGS:
            if sub in text:
                return None
    elif code == "NAP":
        for sub in NAP_DROP_SUBSTRINGS:
            if sub in text:
                return None
    elif code == "FUR":
        for sub in FUR_DROP_SUBSTRINGS:
            if sub in text:
                return None
    elif code == "LIJ":
        for sub in LIJ_DROP_SUBSTRINGS:
            if sub in text:
                return None
    elif code == "SC":
        for sub in SC_DROP_SUBSTRINGS:
            if sub in text:
                return None
    elif code == "SCN":
        for sub in SCN_DROP_SUBSTRINGS:
            if sub in text:
                return None
    return text


# --------------------------------------------------------------------------- #
# Sentence splitter (rule-based, loaded once at import).
# --------------------------------------------------------------------------- #
print("[generation] loading spaCy rule-based sentencizer ...")
NLP = spacy.blank("xx")
NLP.add_pipe("sentencizer")


# --------------------------------------------------------------------------- #
# Per-variety pipeline.
# --------------------------------------------------------------------------- #
def process_dialect(folder: str) -> dict | None:
    label = FOLD_LABEL[folder]
    code = DIAL_LABEL[label].lower()
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

    # ----- Stage 1: load articles ----- #
    articles: list[dict] = []
    for fname in sorted(os.listdir(aa_dir)):
        with (aa_dir / fname).open("r", encoding="utf-8") as f:
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
    df = pd.DataFrame(articles)
    stats = {"raw_articles": raw_n}

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

    # ----- Stage 5: pre-normalize filters (NATIVE) ----- #
    sent_df["text"] = sent_df["text"].apply(lambda t: pre_norm_filter(t, folder))
    sent_df = sent_df.dropna(subset=["text"]).reset_index(drop=True)
    stats["after_pre_norm_filter"] = len(sent_df)
    print(f"  after pre-norm filter:      {len(sent_df):>10,}")
    if len(sent_df) == 0:
        return stats

    # ----- Stage 6: aggressive normalize ----- #
    sent_df["text"] = sent_df["text"].apply(aggressive_normalize)
    # After normalize a few sentences may become empty — drop them.
    sent_df = sent_df[sent_df["text"].str.len() > 0].reset_index(drop=True)
    stats["after_normalize"] = len(sent_df)

    # ----- Stage 7: post-normalize filters ----- #
    sent_df["text"] = sent_df["text"].apply(
        lambda t: post_norm_filter(t, label, folder))
    sent_df = sent_df.dropna(subset=["text"]).reset_index(drop=True)
    stats["after_post_norm_filter"] = len(sent_df)
    print(f"  after post-norm filter:     {len(sent_df):>10,}")

    # ----- Stage 8: sentence dedup on (now-normalized) text ----- #
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
    print(f"  final articles (with ≥1):   {stats['final_articles']:>10,}")

    # ----- Stage 10: build meta + atomic save ----- #
    meta_df = (
        sent_df[["label", "article_id", "title", "url"]]
        .drop_duplicates(subset=["article_id"]).reset_index(drop=True).copy()
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
    print("Aggressive-normalize preprocessing for Italo-Romance Wikipedia dumps")
    print("=" * 75)
    folders = sorted(d for d in os.listdir(".")
                     if os.path.isdir(d) and d in FOLD_LABEL)
    if not folders:
        print("No '*_texts/' directory found in cwd.")
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
        cols = ["raw_articles", "after_article_dedup",
                "raw_sentences", "after_pre_norm_filter",
                "after_post_norm_filter", "final_sentences"]
        header = f"{'dialect':<12}" + "".join(f"{c:>22}" for c in cols)
        print(header)
        for folder, s in all_stats.items():
            row = f"{folder:<12}" + "".join(
                f"{s.get(c, 0):>22,}" for c in cols
            )
            print(row)


if __name__ == "__main__":
    main()
