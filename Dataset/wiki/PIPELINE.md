# Wikipedia preprocessing pipeline — methodology notes

This document records every choice we made when extracting and cleaning
the Wikipedia dumps for our six target Italo-Romance varieties. It is the
single source of truth for the group; whatever the paper claims about the
training corpus must be consistent with what is described here.

## 1. Scope

We process **10 Italo-Romance varieties** in two groups, plus a third
group of **9 comparison languages** that the script can also handle
(URLs commented in `create.py`, ready to activate when we want to
re-extract them):

**Group A** (labels 0-5): the 6 varieties in BOTH OLDI and FLORES — the
primary downstream training/eval set.

| Code | Wiki edition | Italian name |
|---|---|---|
| `fur` | furwiki | Friulano |
| `lij` | lijwiki | Ligure |
| `lmo` | lmowiki | Lombardo |
| `sc`  | scwiki  | Sardo |
| `scn` | scnwiki | Siciliano |
| `vec` | vecwiki | Veneto |

**Group B** (labels 6-9): other italo-romance varieties on Wikipedia.

| Code | Wiki edition | Italian name |
|---|---|---|
| `lld` | lldwiki | Ladino (in FLORES, not OLDI) |
| `nap` | napwiki | Napoletano |
| `pms` | pmswiki | Piemontese |
| `roa_tara` | roa_tarawiki | Tarantino |

**Comparison languages** (labels 10-16): not currently extracted.
URLs are present (commented) in `create.py` — uncomment to activate
when ready.

| Code | ISO 639-3 |
|---|---|
| `ita` | Italian |
| `spa` | Spanish |
| `fra` | French |
| `eng` | English |
| `deu` | German |
| `cat` | Catalan |
| `slv` | Slovenian |

## 2. Sources and snapshot

- **Snapshot**: 2026-04-01 — fixed for reproducibility.
- **Endpoint**: `https://dumps.wikimedia.org/other/mediawiki_content_current/<lang>wiki/2026-04-01/xml/bzip2/<lang>wiki-2026-04-01-pXpY.xml.bz2`
- **Exact URLs**: see `scripts/create.py`.
- **Files preserved on disk** (gitignored): the `.xml.bz2` raw dumps and
  the wikiextractor `*_texts/` intermediates are deleted at the end of
  `create.py`.

## 3. Pipeline at a glance

The pipeline is a **hybrid Camposampiero/ETHZ-first + SUKI-extras**:

- **Camposampiero/ETHZ** (Camposampiero et al. 2022) provides the
  *architecture*: load full articles, clean at article level, then split
  into sentences. Reference:
  [aclanthology.org/2022.vardial-1.10](https://aclanthology.org/2022.vardial-1.10/).
- **SUKI** (Jauhiainen et al. 2022) provides *additional filters*:
  more complete markup regex sets and the per-variety patterns that
  catch the Venetian French-commune templates (~33k lines). Reference:
  [aclanthology.org/2022.vardial-1.13](https://aclanthology.org/2022.vardial-1.13/).

We deliberately **inverted** an earlier draft of this pipeline that was
SUKI-first (line-level cleaning). Line-level filtering before the
sentence splitter produced "moncated" articles which spaCy then split
into fragmented, incoherent sentences. Article-level cleaning + then
splitting yields markedly more natural sentences (verified by sample
inspection).

**MAJOR CHANGE (May 2026)**: the pipeline now applies an **aggressive
normalize step** that strips digits, punctuation, symbols, diacritics,
and case to produce lowercase-ASCII-only text. This makes Wiki output
directly comparable to FLORES+ and OLDI normalized variants (parallel
files at `Dataset/flores/normalized/` and `Dataset/oldi/normalized/`,
produced by `Dataset/{flores,oldi}/scripts/normalize.py`). The original
text is NOT preserved in the Wiki CSV — only the normalized form. For
FLORES/OLDI the original files are kept under `not_normalized/`.

Pipeline summary (10 stages):

```
wikiextractor (cached)
  → Stage 1.  load articles
  → Stage 2.  article-level clean (HTML + SUKI markup + drop <50 chars)
  → Stage 3.  article-level dedup (keep=False)
  → Stage 4.  sentencizer + lowercase-merge (fix abbreviation splits)
  → Stage 5.  PRE-NORM filters [native]: len>=30, has-lowercase-ASCII,
                                          has-word-lowercase, endswith-terminator
  → Stage 6.  AGGRESSIVE NORMALIZE [→ aggressive]
  → Stage 7.  POST-NORM filters [aggressive]: len<=500 (mega-list killer)
                                              + per-variety lowercase substrings
                                              (VEC 6+5, LMO 22, PMS 11)
  → Stage 8.  sentence dedup (keep="first") on aggressive
  → Stage 9.  auto prefix-dedup PL=30 K=10 on aggressive
  → Stage 10. save: wiki/<group>/<lang>.csv (single text col) + _meta + _stats
```

The previous fingerprint dedup (digit→N, roman→R) is REMOVED — on
aggressive text those characters are already stripped, so the pass
becomes a no-op equivalent to Stage 8 dedup.

## 4. Step by step

Numbering matches the order of stages in `scripts/generation.py`.

### Stage 0 — Extraction
- Tool: `wikiextractor.WikiExtractor` with `--json`.
- Output: `<lang>_texts/AA/wiki_*` JSON files (one article per JSON line).
- Comes from `scripts/create.py` (which also handles the download).

### Stage 1 — Load full articles
- For each article, keep the full `text` field as one string. We do **not**
  split on `\n` here (that's Camposampiero-style; we work at article level).

### Stage 2 — Article-level cleaning
Applied to the full article text before any splitting:

1. **HTML entity decoding** (Camposampiero/ETHZ): `html.unescape(text)`.
   Turns `&lt;`, `&gt;`, `&amp;`, `&quot;`, `&apos;`, etc. into the
   real characters they encode. SUKI does not need this for char-n-gram
   NB; FLORES/OLDI never contain entities, so we decode for symmetry.
2. **Drop wiki headers** `==...==` (regex, ETHZ-style).
3. **SUKI markup set 1** (drop the matched span):
   `<comment>.*</comment>`, `<contributor>`, `</contributor>`,
   `<format>.*</format>`, `<ip>.*</ip>`, `<minor />`,
   `<model>.*</model>`, `<ns.*/ns>`, `<parentid>.*</parentid>`,
   `<revision>`, `<timestamp>.*</timestamp>`, `<username>.*</username>`.
4. **SUKI markup set 2** (drop the matched tag):
   `</math>`, `</[pP]oem>`, `</small>`, `<references>`, `</html>`,
   `</includeonly></onlyinclude>`, `</table>`, `<?php`, `<BR C.* >`,
   `#redirect`.
5. **Strip stray `<` and `br>` tags** (replace with space).
6. **Whitespace collapse** (`\s+ → " "`) and `strip()`.
7. **Drop articles shorter than 50 characters** (Camposampiero/ETHZ).

### Stage 3 — Article-level deduplication
- `df.drop_duplicates(subset="text", keep=False)` — Camposampiero-style.
- Drop **all** occurrences of any duplicated article text. Two
  byte-identical full articles are essentially always boilerplate
  (bot-generated template pages, accidental dupes, redirects), so we
  remove them entirely.
- This is the most aggressive step in the pipeline: vec drops 27% of
  articles here, lmo 30% — both are the "bot-generated commune pages"
  signature.

### Stage 4 — Sentence splitting (rule-based) + lowercase-merge

- We use **spaCy's rule-based sentencizer**, not the Italian statistical
  parser:

  ```python
  NLP = spacy.blank("xx")
  NLP.add_pipe("sentencizer")
  ```

- Camposampiero originally used `it_core_news_sm` (the Italian parser)
  for sentence segmentation, but it is trained on Italian standard text
  and on dialectal text it makes systematic errors — sentences come out
  fragmented or merged at unnatural boundaries. We verified empirically
  that on the same vec corpus the IT parser produces ~10-15% more
  fragments and 10-15% of "fake" sentences ending with `:` (the parser
  splits aggressively on colons inside lists), while the sentencizer
  yields fewer and longer sentences (median +12-20 chars) that are more
  natural.
- After splitting, we **merge any sentence that starts with a lowercase
  letter into the previous one**. This corrects the sentencizer's main
  failure mode: it splits on every period, including inside abbreviations
  like `s.p.a.`, `a.C.`, `D.O.C.G.`, `es.`, `Art.`, producing fragments
  starting with lowercase. The merge is generic — no hard-coded
  abbreviation list needed.

### Stage 5 — Pre-normalize generic filters (NATIVE text)

These run on the **native** sentence text, before aggressive normalize.
They use punctuation/case/digits/romans which are about to be stripped.

- **Drop sentences shorter than 30 characters**. We tightened from the
  Camposampiero/SUKI default of 20 to 30 because below that threshold
  almost everything is structured noise (scoreline tables, infobox
  values, single-row headings). We sampled the 21–30 char range on vec
  and found 10/10 to be such fragments.
- **Drop sentences without any lowercase ASCII letter** (SUKI). Catches
  all-caps fragments (`"ROMA"`, `"GIOVANNI XXIII"`).
- **Drop sentences without a word starting with lowercase ASCII** (SUKI).
  Catches title-case fragments (`"Roma Capitale d'Italia"`, `"Storia
  Della Toscana"`).
- **Drop sentences that do not end with a sentence terminator**
  (`.`, `!`, `?`, `"`, `'`, `”`, `’`, `»`, `…`, `)`, `]`). We
  deliberately exclude `:` and `;` — empirically a colon-ending fragment
  is almost always a section heading or an intro to a list, not a real
  sentence. This single filter catches three kinds of residual noise
  simultaneously: (i) section titles, (ii) splitter mid-sentence
  glitches, (iii) the giant comune-list templates that span thousands
  of characters and end with the last municipality name (no period).

The **per-variety patterns** were moved out of this stage — they now
live in Stage 7 (post-normalize) as lowercase substring matches on
aggressive text. See Stage 7 below for the full list.

### Stage 6 — Aggressive normalize

Single function `aggressive_normalize(text)` in `generation.py`. Pipeline:

1. **NFC** + explicit char mapping for letters that NFD does not
   decompose: `ß → ss`, `ł → l, Ł → L`, `æ → ae, œ → oe`, `ø → o`,
   `đ → d`, `ð → d`, `þ → th` (and uppercase variants).
2. **Strip uppercase Roman numerals** `\b[IVXLCDM]{2,}\b → " "` BEFORE
   lowercase (the regex is uppercase-only by design — case-insensitive
   would false-positive on `"il"`, `"vi"`, `"li"`, etc.).
3. **Lowercase**.
4. **NFD decompose** + strip combining diacritic marks
   (`[̀-ͯ]`). Handles all the regular accents:
   `à è é ñ ç ü ä ö è ì ò ù ï` → `a e e n c u a o e i o u i`.
5. **Strip digits** `\d+ → " "`.
6. **Strip everything not `[a-z\s]`** — punctuation, symbols, residual
   non-Latin characters.
7. **Collapse whitespace** to single spaces.

Output: only lowercase ASCII letters and single spaces.

Concrete example (vec):
```
INPUT:  "Ła cità de Padova ga 207,694 abitanti nel 2023 (CCXXIII° posto in Europa) — fonte: Istat."
OUTPUT: "la cita de padova ga abitanti nel posto in europa fonte istat"
```

### Stage 7 — Post-normalize filters (AGGRESSIVE text)

- **Drop sentences longer than 500 characters**. Catches the residual
  mega-list templates (`"comuni della provincia di padova padova abano
  terme albignasego ..."` for thousands of chars). The terminator
  filter in Stage 5 caught them when they ended without a period; this
  catches the rest.
- **Per-variety lowercase substring drops**. The previous regex /
  substring patterns were rewritten as plain lowercase substring lookups
  on aggressive text. Simpler, tolerant of small variants, and uniform
  across varieties:
  - **VEC** — 6 single-substring drops + 5 multi-substring (AND) drops:
    - Single: `"numari romani"`, `"numeri romani"`, `"par rivar al cao"`,
      `"el xe un comun de"`, `"el xe on comun de"`, `"gregorian"`.
    - AND: `("abitanti del departemento", "in fransa")` (was French
      commune regex), `("comun italian de", "abitanti")` (was Italian
      commune regex), `("ghe manca", "fin de l")` (was day-of-year
      template), `("el xe on an", "secolo")` and `("el xe un an",
      "secolo")` (was year-page template).
  - **LMO** — 22 substrings: `"la stazzion de"`, `"el cumun"`,
    `"el cumu"`, `"el distret"`, `"al gh ha pressapoch abitant"`,
    `"l andament del numer de abitant"`, `"e na densita de"`, ...
    (was 24, dropped `km²` which collapses to too-generic `km`, and
    merged `L'andament/L'andamènt` which collapse to the same form).
  - **PMS** — 11 substrings: `"grup ed popolassion"`, `"a confin a con"`,
    `"con na densita"`, `"el sindich a l e"`, ...
  - **fur, lij, sc, scn, lld, nap, roa_tara** + comparison languages —
    no per-variety patterns. Rely on Stages 8-9 (data-driven dedup).

### Stage 8 — Sentence-level dedup (AGGRESSIVE text)

- `drop_duplicates(subset="text", keep="first")` on aggressive text.
- Catches case/punctuation/diacritic variants that were distinct
  sentences in native form but collapse to the same string post-aggressive.
  Example: `"La Casa Rossa."` and `"la casa rossa"` (different in
  native, same after normalize).
- We use `keep="first"` (not `keep=False`) — Camposampiero used the
  latter at sentence level, but it was too aggressive: legitimate
  repeated sentences (e.g. *"Inoltre, è un noto attore."*) appearing
  in two unrelated articles got removed entirely. Soft dedup is enough.

### Stage 9 — Auto prefix-dedup PL=30 K=10 (AGGRESSIVE text)

- Compute `prefix(t) = t[:30]` for every sentence (already lowercase
  after Stage 6) and **drop sentences whose prefix appears ≥10 times**.
- 30/10 thresholds verified empirically (sweet spot between false
  positives on real sentences sharing prefixes by coincidence and false
  negatives on small template clusters).
- Catches data-driven boilerplate (LMO French commune pages, scn
  demographic stubs, vec province/commune templates) that the
  hard-coded substring lists in Stage 7 may not cover.

### Stage 10 — Save outputs (atomic)

- Write to `.tmp` first, then rename, so a crash mid-write never leaves
  truncated files.
- Per variety we save:
  - `<code>.csv` — `text,label,article_id` (text is **normalized only**;
    no native column).
  - `<code>_meta.csv` — `article_id,title,url,n_sentences` (titles and
    URLs preserve the native form for human readability).
  - `<code>_stats.json` — line counts after each stage.

### Removed: old Stage 8 fingerprint dedup

The previous fingerprint dedup (digit→N, roman→R, lowercased) was
removed. On aggressive text, digits and roman numerals are already
stripped to spaces, so the fingerprint computation is identity (no
substitutions happen). The pass would be exactly equivalent to Stage 8
exact dedup → no information gained, just slower.
## 5. Where we deliberately depart from the published pipelines

| Step | We do? | Reason |
|---|---|---|
| SUKI: substitute every digit with `1` | **STRIP entirely** (was: NO) | Aggressive normalize Stage 6 strips all digits. Decision driven by the embedding-similarity task: numbers are universal background noise across languages and inflate cross-language similarity in TF-IDF / FastText / Word2Vec methods (e.g., shared "1995", "2023" between Italian and Arabic Wikipedia). For methods that benefit from numbers as semantic context (CANINE, contextual encoders) we have separately the FLORES `not_normalized/` files. |
| SUKI + Camposampiero: normalize Venetian `ł → l, Ł → L` | **YES** (was: NO) | Aggressive normalize Stage 6 maps `ł → l` for ALL languages where it appears. With the project moving to a full normalization step, the Wiki↔FLORES↔OLDI consistency is preserved by applying the same `aggressive_normalize` function in `Dataset/{flores,oldi}/scripts/normalize.py`. The `not_normalized/` folders preserve the original `ł`-bearing text for any analysis that needs it. |
| Aggressive: lowercase + strip diacritics + strip punctuation + strip symbols | **YES** (NEW) | Same rationale as digit stripping: cross-language similarity benefits from removing surface noise. `año → ano`, `Schön → schon`, `cità → cita` is acceptable loss — within-language disambiguation is not the project's task; cross-language alignment is. |
| Camposampiero: `text.replace('"', " ")` | **NO** | This breaks CSV quoting in Camposampiero's output; we keep the original quotes (and they get stripped by Stage 6 anyway). |
| Camposampiero: greedy regex `\(.*\)` and `\[.*\]` to strip parentheticals | **NO** | Bug in the original Camposampiero script: the greedy `.*` matches from the first `(` to the *last* `)` in the article, deleting large stretches of legitimate text. We do not strip parentheticals at the article level (Stage 6 strips parens at sentence level cleanly). |
| Camposampiero: `it_core_news_sm` parser for sentence segmentation | **NO** | Trained on Italian standard, makes systematic errors on dialect. We use the rule-based sentencizer instead (see Stage 4). |
| Camposampiero: sentence-level `drop_duplicates(keep=False)` | **NO** | Drops legitimate repeated sentences. We use `keep="first"` (Stage 8). |
| SUKI: per-variety adaptive thresholding for the classifier | n/a | We are not training a Naive Bayes classifier; SUKI's adaptive splits are model-specific. |

### Diagnostic comparison with the legacy Camposampiero output

For the record (legacy `wiki_old/` snapshot, no longer kept on disk):
our pipeline produced **1.4–2.0×** as many sentences and **1.6–2.8×**
as much character content for the same six varieties on the same
Wikimedia snapshot (2026-04-01). The article counts agreed within 5%
across varieties (input dumps equivalent), so the gain is **legitimate
text** the legacy script was wrongly removing — primarily because of
two greedy-regex bugs (`\(.*\)` and `\[.*\]`) and the
`text.replace('"', " ")` call (see §5). Verified at the time by
per-article diffs (vec new: 978 chars/article; old: 540 chars/article).
The legacy folder has since been deleted to avoid confusion.

## 6. Where each addition comes from (origin map)

| Addition | Origin |
|---|---|
| Article-level architecture | Camposampiero/ETHZ |
| `html.unescape` (Stage 2) | Camposampiero/ETHZ |
| Wiki-header drop `==...==` (Stage 2) | Camposampiero/ETHZ |
| SUKI markup regex sets 1+2 (Stage 2) | SUKI |
| Stray `<`/`br>` strip (Stage 2) | SUKI + Camposampiero |
| Article length filter ≥50 (Stage 2) | Camposampiero/ETHZ |
| Article-level dedup `keep=False` (Stage 3) | Camposampiero/ETHZ |
| Rule-based sentencizer (Stage 4) | ours (after empirical comparison with parser IT) |
| Lowercase-start merge (Stage 4) | ours (fixes sentencizer split-on-abbreviation glitches) |
| Sentence length filter `>30` (Stage 5) | ours (tightened from Camposampiero's `>20`) |
| HAS_LOWER_ASCII filter (Stage 5) | SUKI |
| HAS_WORD_LOWER filter (Stage 5) | SUKI |
| Terminator-punctuation filter (Stage 5) | ours (catches titles, splitter glitches, mega-lists) |
| **Aggressive normalize (Stage 6)** | **ours (NEW: lowercase ASCII + strip diacritics/digits/punct/symbols + explicit map ß→ss, ł→l, æ→ae)** |
| Length max 500 filter (Stage 7) | ours (kills mega-list templates that survived terminator filter) |
| VEC post-norm substrings (Stage 7, 6+5 patterns) | rewritten from native regexes (SUKI French/Italian commune + SUKI Roman numbers + ours year/day templates + Camposampiero substrings) |
| LMO post-norm substrings (Stage 7, 22 patterns) | rewritten from Camposampiero/ETHZ 24 substrings (dropped `km²`, merged accent-equivalents) |
| PMS post-norm substrings (Stage 7, 11 patterns) | rewritten from Camposampiero/ETHZ |
| Sentence-level exact dedup `keep="first"` (Stage 8, on aggressive) | ours (`keep="first"` is softer than Camposampiero `keep=False`; on aggressive catches case/punct/diacritic variants) |
| Auto prefix-dedup PL=30 K=10 (Stage 9, on aggressive) | ours (data-driven, catches prefix-template clusters) |
| Three-folder routing (Group A / Group B / languages) | ours (separates OLDI ∩ FLORES varieties from "others" from comparison) |
| Atomic save + per-stage `_stats.json` (Stage 10) | ours |

## 7. Output format

```text
Dataset/wiki/
├── dialects_in_both_OLDI_and_Flores/   → Group A (fur, lij, lmo, sc, scn, vec)
│   ├── <code>.csv         → columns: text, label, article_id
│   ├── <code>_meta.csv    → columns: article_id, title, url, n_sentences
│   └── <code>_stats.json  → per-stage line counts
├── others_dialects/                    → Group B (lld, nap, pms, roa_tara)
│   └── (same three files per code)
├── languages/                          → comparison languages (ita/eng/fra/spa/cat/deu/slv)
│   ├── <iso>.csv          → columns: text, label, article_id (legacy Camposampiero)
│   └── <iso>_meta.csv     → columns: article_id, title, url, ...
├── _cache/                             → wikiextractor *_texts/ + .xml.bz2 (gitignored)
├── scripts/
│   ├── create.py                       → download + extract + invoke generation
│   └── generation.py                   → cleaning pipeline (§4)
└── PIPELINE.md                         → this file
```

`label` is a small integer defined in `scripts/generation.py:FOLD_LABEL`:
- Group A: `fur=0, lij=1, lmo=2, sc=3, scn=4, vec=5`
- Group B: `lld=6, nap=7, pms=8, roa_tara=9`
- Comparison: `ita=10, spa=11, fra=12, eng=13, deu=14, cat=15, slv=16`

Downstream code can map back to a code via `DIAL_LABEL`.

For FLORES and OLDI, the same `aggressive_normalize` function is applied
by `Dataset/flores/scripts/normalize.py` and `Dataset/oldi/scripts/normalize.py`,
producing parallel files:

```text
Dataset/flores/
├── not_normalized/         original FLORES+ files (.txt + parallel.tsv)
└── normalized/             aggressive-normalized parallels

Dataset/oldi/
├── not_normalized/         original OLDI parquet + pairs TSVs
└── normalized/             aggressive-normalized parallels
```

## 8. Reproduction

For Wiki:

```bash
cd Dataset/wiki
python scripts/create.py
```

`create.py` will:
1. download the dumps from Wikimedia for snapshot `2026-04-01`,
2. run `wikiextractor` for each,
3. invoke `scripts/generation.py` which produces the per-variety
   `.csv`, `_meta.csv`, and `_stats.json` (single text col, normalized).
4. cache `_cache/` is preserved between runs; delete it manually to
   force re-download.

For FLORES and OLDI normalization (run once after originals are in place):

```bash
python Dataset/flores/scripts/normalize.py
python Dataset/oldi/scripts/normalize.py
```

These read from `not_normalized/` and write to `normalized/` next to it.

Tool versions (current):
- `wikiextractor==3.0.6`
- `spacy==3.5.3`
- Python 3.9 (matches the Bocconi HPC `stud` partition).

## 9. Final stats

Final per-variety sentence counts after the full 10-stage pipeline
(snapshot 2026-04-01, with aggressive normalize). Output is split into
three subfolders under `Dataset/wiki/`.

**Group A — `dialects_in_both_OLDI_and_Flores/`** (6 varieties in BOTH
OLDI and FLORES, primary set):

| Variety | raw art. | after art.dedup | raw sent. | after pre-norm | after post-norm | **final** |
|---|---:|---:|---:|---:|---:|---:|
| fur | 4,979 | 4,678 | 30,228 | 24,062 | 23,969 | **22,468** |
| lij | 8,223 | 7,371 | 62,139 | 52,534 | 52,114 | **50,917** |
| lmo | 79,073 | 52,155 | 252,941 | 214,231 | 161,744 | **129,263** |
| sc  | 7,692 | 7,648 | 68,817 | 62,483 | 61,935 | **61,213** |
| scn | 23,457 | 19,373 | 94,075 | 82,067 | 81,714 | **78,125** |
| vec | 68,960 | 49,367 | 159,686 | 141,270 | 102,053 | **98,786** |

**Group B — `others_dialects/`** (italo-romance varieties NOT in OLDI;
lld is in FLORES):

| Variety | raw art. | after art.dedup | raw sent. | after pre-norm | after post-norm | **final** |
|---|---:|---:|---:|---:|---:|---:|
| lld      | 176,204 | 166,503 | 385,088 | 360,330 | 360,296 | **211,432** |
| nap      |  12,887 |  11,835 |  32,688 |  29,027 |  28,960 | **27,577** |
| pms      |  71,072 |  69,397 | 270,038 | 207,158 | 109,713 | **96,024** |
| roa_tara |   9,230 |   8,101 |  34,744 |  30,362 |  30,111 | **29,682** |

**Comparison languages — `languages/`** — currently legacy Camposampiero
files (ita 320k, deu 360k, eng 301k, fra 294k, spa 277k, slv 249k,
cat 104k). Re-extract with the new pipeline by uncommenting the dump
URLs in `scripts/create.py`.

Notable from the May 2026 normalized run vs. the previous Camposampiero
hybrid pipeline:
- **lmo** drops more (-4.7%) because Stage 6 normalize collapses many
  case/diacritic variants into duplicates that Stage 8 catches; also,
  the post-norm LMO substring list catches additional templates.
- **pms** drops more (-3.0%) for the same reason.
- All other varieties drop less than 1.5% — the new pipeline is mostly
  consistent with the previous one in terms of sentence count, but the
  output text is now lowercase-ASCII-only.

Quality indicators on final output:
- **Sentence length**: typically 60-120 chars (was 110-140 native — drop
  due to removed punctuation/digits/spaces).
- **Character classes in output**: only `[a-z\s]` for all varieties.
- **HTML entity residue**: 0 across all varieties.

For the historical comparison with the legacy Camposampiero output
(`wiki_old/`, no longer kept on disk), see §5 "Diagnostic comparison
with the legacy Camposampiero output". Summary: at the same Wikimedia
snapshot (2026-04-01) and same article counts (within 5%), the new
pipeline produced **1.4–2.0× more sentences** than the legacy one,
all from legitimate text the old greedy regexes were wrongly deleting.

## 10. License and attribution

The Wikipedia source text is **CC BY-SA 3.0**. Any derivative dataset
we publish or share must:
- preserve attribution to Wikipedia,
- remain CC BY-SA 3.0 compatible,
- record the snapshot date (2026-04-01),
- cite the methodology references below.

References to cite when the cleaning pipeline is described in the paper:

- Aepli, N. et al. (2022). *Findings of the VarDial Evaluation Campaign 2022*.
  [aclanthology.org/2022.vardial-1.1](https://aclanthology.org/2022.vardial-1.1/).
- Jauhiainen, T., Jauhiainen, H., Lindén, K. (2022). *Italian Language and
  Dialect Identification and Regional French Variety Detection using
  Adaptive Naive Bayes*.
  [aclanthology.org/2022.vardial-1.13](https://aclanthology.org/2022.vardial-1.13/).
  (SUKI — extra markup regexes and per-variety VEC patterns we adopt.)
- Camposampiero, G. et al. (2022). *The Curious Case of Logistic
  Regression for Italian Languages and Dialects Identification*.
  [aclanthology.org/2022.vardial-1.10](https://aclanthology.org/2022.vardial-1.10/).
  (Camposampiero/ETHZ — article-level pipeline architecture and per-variety
  LMO/VEC substring patterns we adopt.)

## 11. Open issues / future improvements

- **Re-extract comparison languages with the new pipeline**: dump URLs
  are commented in `scripts/create.py`. When ready, uncomment + fill in
  the actual `pXpY` page ranges from `dumps.wikimedia.org`. Output
  routes to `wiki/languages/<iso>.csv`.
- **Empirical boilerplate dictionaries for fur/lij/sc/scn**: we don't
  have explicit substring lists for these. Derive by inspecting top
  n-grams in the per-variety CSVs after each generation run. Estimated
  effort: ~30 min/variety.
- **FLORES↔Wiki overlap check**: a quick sanity check that no FLORES
  devtest sentence (after aggressive normalize) appears verbatim in the
  cleaned Wiki training set. Trivial to run on the normalized files.
- **External validation on ITDI dev/test**: train a TF-IDF char
  classifier on our Wiki and evaluate on ITDI dev (6,799 sentences,
  with our 6 varieties present) — sanity check that the pipeline
  produces classifiable data and a number comparable to published
  baselines.
- **`it_core_news_sm` no longer required**: now that we use the
  rule-based sentencizer, the model is no longer used. We could remove
  it from `requirements.txt` and shrink the venv (~50 MB).

## 12. Normalization function — single source of truth

The `aggressive_normalize` function is defined identically in three
places (kept in sync manually — short enough to copy-paste):

```
Dataset/wiki/scripts/generation.py     # for Wiki extraction (Stage 6)
Dataset/flores/scripts/normalize.py    # for FLORES not_normalized → normalized
Dataset/oldi/scripts/normalize.py      # for OLDI not_normalized → normalized
```

If you change one (e.g., add a new explicit char mapping), update the
other two so that Wiki/FLORES/OLDI normalized variants stay byte-equivalent
on shared text.

