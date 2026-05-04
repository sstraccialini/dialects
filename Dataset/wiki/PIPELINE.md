# Wikipedia preprocessing pipeline — methodology notes

This document records every choice we made when extracting and cleaning
the Wikipedia dumps for our six target Italo-Romance varieties. It is the
single source of truth for the group; whatever the paper claims about the
training corpus must be consistent with what is described here.

## 1. Scope

We process **6 Italo-Romance varieties** — the intersection of OLDI and
FLORES+ training/eval sets:

| Code | Wiki edition | Italian name |
|---|---|---|
| `fur` | furwiki | Friulano |
| `lij` | lijwiki | Ligure |
| `lmo` | lmowiki | Lombardo |
| `sc`  | scwiki  | Sardo |
| `scn` | scnwiki | Siciliano |
| `vec` | vecwiki | Veneto |

Italian standard, the other Romance languages (es/fr/ca), and the
non-Romance comparison languages (de/en/el/ar/sl) are NOT re-extracted
here. If we ever need them with the same cleaning pipeline, we can add
them by extending `FOLD_LABEL` in `scripts/generation.py`.

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

Pipeline summary:

```
wikiextractor (cached)
  → article-level clean (HTML entities + SUKI markup + drop short)
  → article-level dedup (keep=False, kills bot-generated duplicates)
  → sentencizer + lowercase-merge (fix abbreviation splits like "s.p.a.")
  → sentence-level filters (length>30 + lowercase + per-variety + must end with .!?)
  → sentence dedup (keep="first")
  → auto prefix-dedup PL=30 K=10 (data-driven catch of template clusters)
  → save: wiki/<group>/<lang>.csv + _meta + _stats
```

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

### Stage 5 — Sentence-level filters

- **Drop sentences shorter than 30 characters**. We tightened from the
  Camposampiero/SUKI default of 20 to 30 because below that threshold
  almost everything is structured noise (scoreline tables like
  `"Juve - Cagliari : 2 a 1."`, infobox values like
  `"Linea de costa: 296 km."`, single-row headings like
  `"Maria inte el Nòvo Testamento."`). We sampled the 21–30 char range
  on vec and found 10/10 to be such fragments, while the volume cost
  is just ~2-3% per variety.
- **Drop sentences without any lowercase ASCII letter** (SUKI).
- **Drop sentences without a word starting with lowercase ASCII** (SUKI).
- **Drop sentences that do not end with a sentence terminator**
  (`.`, `!`, `?`, `"`, `'`, `”`, `’`, `»`, `…`, `)`, `]`). We
  deliberately exclude `:` and `;` — empirically a colon-ending fragment
  is almost always a section heading or an intro to a list (e.g.
  `"Persone inportanti che xe nate a Padoa:"`), not a real sentence.
  This single filter catches three kinds of residual noise simultaneously:
  (i) section titles, (ii) splitter mid-sentence glitches, (iii) the
  giant comune-list templates that span thousands of characters and
  end with the last municipality name (no period).
- **Per-variety filters**:
  - **VEC** — eight drops total (no orthographic normalization, see §5):
    - SUKI: French commune template
      `el xe on comun de.*abitanti del departemento.*in Fransa\.`
      (33,701 lines in SUKI; biggest single boilerplate cluster of vec).
    - SUKI: Italian commune template
      `el xe (?:on|un) comun italian de.*abitanti`.
    - SUKI (generalized): Roman numbers stub
      `\([MDCLXVI]+(?:\s+v\.C\.?)?\s+(?:en|in)\s+num[ae]ri\s+romani\)`.
      Generalized from SUKI's original `L?[IVXC]+` to also match
      `MCMXV`-style year-page templates.
    - Year-page template `el xe (?:on|un) an(?:no)? del [IVX]+ sec`
      (TACL survey: ~1k pages for years 1 BC–999 BC).
    - Day-of-year template `ghe manca \d+ d[iì] par l[ae]? fin de l'ann?o`
      (one page for each of the 365 days).
    - Day-to-year-end template `par rivar al cao de l'an[oó] ghe vo[lł]+e`.
    - Camposampiero: generic-comun substring `el xe un comun de`.
    - Camposampiero: calendar-stub substring `gregorian`.
  - **LMO** — combination of SUKI and Camposampiero:
    - SUKI: drop lines shorter than 14 characters.
    - Camposampiero: 24 substring patterns covering geographic /
      template stubs (`El cumün`, `La Stazzion de`, `km²`,
      `El Distret`, `Al gh'ha pressapoch abitant`,
      `L'andament del numer de abitant`, ...). Full list in
      `scripts/generation.py:LMO_CAMPOSAMPIERO_SUBSTRINGS`.
  - **fur, lij, sc, scn** — no hard-coded per-variety patterns; for
    these varieties we rely on the general filters above plus the
    auto prefix-dedup in Stage 7 (which catches the same kind of
    bot-generated templates without needing a manual catalogue).

### Stage 6 — Sentence-level exact-duplicate dedup
- `drop_duplicates(subset="text", keep="first")` — keep the first
  occurrence and drop subsequent duplicates.
- A previous version of this pipeline used `keep=False` like
  Camposampiero, but at sentence level that turned out too aggressive:
  legitimate repeated sentences (e.g. *"Inoltre, è un noto attore."*)
  appearing in two unrelated articles got removed entirely. The
  boilerplate volume is already controlled by Stages 2, 3, and 5 —
  at this point a soft dedup is enough.

### Stage 7 — Auto prefix-based template dedup (data-driven)
- Compute `prefix_lower(t) = t[:30].lower()` for every surviving
  sentence, and **drop all sentences whose prefix appears ≥10 times**.
  Concretely: any cluster of 10+ sentences starting with the same first
  30 characters is treated as a templated boilerplate and removed.
- **Limitation**: only catches templates that begin with a fixed prefix.
  Templates whose variable part appears early (e.g. year-page stubs
  `"El 64 v.C. (LXIV in numari romani) el xe..."` where the year
  varies in the first few chars) escape — they are caught by Stage 8
  instead.
- Why 30/10 (and not 20/10 or 15/20):
  - 30 characters is long enough that natural sentences almost never
    coincide on the prefix by accident (verified empirically: in vec
    most prefix-30 clusters above 10 occurrences are bot-generated
    `"El xe in provincia de ..."` / `"L'abità el xe situà ..."`,
    while shorter prefixes like `el confina co i` would also match
    legitimate descriptive sentences and trigger false positives).
  - The 10-occurrence threshold filters out small accidental clusters
    while still catching template families (most templates produce
    50–1500 occurrences).
- Concrete catches per variety with 30/10:

  | Variety | dropped here | of which not already caught upstream |
  |---|---:|---|
  | fur |   ~35 | minimal |
  | lij |  ~220 | new |
  | lmo | ~3,600 | mostly new (LMO French commune template) |
  | sc  |  ~125 | minimal |
  | scn |  ~755 | mostly new ("Havi na pupulazzioni di...") |
  | vec | ~1,400 | mostly new (province / confina / abitato) |

  Crucially this catches the lmo bot-generated French commune pages
  ("El fa part del cantù de Romans-sur-Isère...", 963 occurrences)
  which neither SUKI nor Camposampiero patterns covered.
- This step makes the per-variety hard-coded substring lists in
  Stage 5 less load-bearing — they remain because they catch
  *substring* patterns (template content embedded mid-sentence,
  which prefix-matching cannot see), but the auto-dedup is the
  more powerful net for *prefix-templates*.

### Stage 8 — Fingerprint-based dedup (numbers / roman numerals)
- Complementary to Stage 7. Many templates do **not** share a fixed
  prefix because the variable part is in the middle (year-page stubs,
  statistical templates, geographic data). To catch these, we compute
  a normalized fingerprint:

  ```python
  fingerprint(t) = re.sub(r"\d+", "N",
                   re.sub(r"\b[IVXLCDM]{2,}\b", "R", t.lower()))
  ```

  i.e. all digit runs become `N` and all roman-numeral tokens of
  length ≥2 become `R`. Sentences whose fingerprint clusters with
  ≥10 others are dropped.
- Crucially, **the CSV keeps the original sentence** (with real digits
  and roman numerals). The fingerprint is used only for the dedup
  decision — coherence with FLORES/OLDI (which keep real digits) is
  preserved.
- Concrete catches per variety:

  | Variety | dropped here | of which |
  |---|---:|---|
  | fur | ~780 (3.2%) | statistical templates |
  | lij | ~700 (1.3%) | varied |
  | lmo | ~500 (0.3%) | residue (most caught upstream) |
  | sc  |   ~10 (0%)  | almost nothing |
  | scn | ~1,360 (1.7%) | song / film templates |
  | vec | ~460 (0.5%) | year/day templates that survived Stage 5 |
  | lld |  ~220 (0.1%) | minimal |
  | nap | ~700 (2.4%) | demographic templates |
  | pms | ~1,950 (1.9%) | municipal data templates |
  | roa_tara | 0 | already clean |

  Sample (vec): `"L'abità el xe situà a 5 metri s.l.m."` — 437 copies
  varying only in the `5`. Stage 7 (prefix-dedup) catches this too,
  but fingerprint also catches `"Ła ga na superfisie de 2467,35 km²
  e ła conprende 29 comuni."` (11 copies varying in numbers, prefix
  too short to cluster).

### Stage 9 — Save outputs (atomic)
- Write to `.tmp` first, then rename, so a crash mid-write never leaves
  truncated files.
- Per variety we save:
  - `<code>.csv` — `text,label,article_id` (the actual training data).
  - `<code>_meta.csv` — `article_id,title,url,n_sentences`.
  - `<code>_stats.json` — line counts after each stage, comparable
    column-by-column to SUKI Table 4.

## 5. Where we deliberately depart from the published pipelines

| Step | We do? | Reason |
|---|---|---|
| SUKI: substitute every digit with `1` | **NO** | SUKI does this so that "in 2013" and "in 2014" collapse for their language-ID classifier. For embedding/distance modeling, numbers carry signal. Most importantly, FLORES devtest and OLDI seed pairs **keep real digits** — substituting in the Wiki side would create a Wiki↔eval mismatch. |
| SUKI + Camposampiero: normalize Venetian `ł → l, Ł → L` | **NO** | SUKI normalizes to align with the ITDI dev set, which uses `l`. Camposampiero applies the same. But our evaluation targets do **not** use `l`: 82% of FLORES veneto sentences and 83% of OLDI veneto pairs contain `ł`. Normalizing the Wiki side would create a Wiki↔eval orthographic mismatch. If a downstream encoder is sensitive to it (e.g. XLM-R subword), we apply `ł→l` as an in-line preprocessing step at evaluation time, **on both training and eval inputs**, never on the source files. |
| Camposampiero: `text.replace('"', " ")` | **NO** | This breaks CSV quoting in Camposampiero's output; we keep the original quotes. |
| Camposampiero: greedy regex `\(.*\)` and `\[.*\]` to strip parentheticals | **NO** | This is a **bug in the original Camposampiero script** — the greedy `.*` matches from the first `(` to the *last* `)` in the article, which deletes large stretches of legitimate text whenever an article contains multiple parentheses. We do not strip parentheticals at all (they are usually informative). |
| Camposampiero: `it_core_news_sm` parser for sentence segmentation | **NO** | Trained on Italian standard, makes systematic errors on dialect. We use the rule-based sentencizer instead (see Stage 4). |
| Camposampiero: sentence-level `drop_duplicates(keep=False)` | **NO** | Drops legitimate repeated sentences. We use `keep="first"` at sentence level. |
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
| VEC French/Italian commune regexes (Stage 5) | SUKI |
| VEC Roman-numbers regex (Stage 5) | SUKI (generalized to `[MDCLXVI]+` and `numari`/`numeri`) |
| VEC year-page + day-of-year templates (Stage 5) | ours (data-driven, ~2k vec sentences) |
| VEC `el xe un comun` + `gregorian` substrings (Stage 5) | Camposampiero/ETHZ |
| LMO `len < 14` filter (Stage 5) | SUKI |
| LMO 24 boilerplate substrings (Stage 5) | Camposampiero/ETHZ |
| Sentence-level exact dedup `keep="first"` (Stage 6) | ours (Camposampiero used `keep=False`, too aggressive at sentence level) |
| Auto prefix-dedup PL=30 K=10 (Stage 7) | ours (data-driven, catches prefix-template clusters) |
| Fingerprint dedup K=10 (digits→N, romans→R) (Stage 8) | ours (data-driven, catches mid-sentence templates that vary only in numbers/roman numerals — vec year-page stubs, pms municipal data, etc.) |
| PMS Camposampiero substrings (Stage 5) | Camposampiero/ETHZ |
| Two-folder routing (Group A / Group B) | ours (separates OLDI ∩ FLORES varieties from "others") |
| Atomic save + per-stage `_stats.json` (Stage 9) | ours |

## 7. Output format

```text
Dataset/wiki/
├── dialects_in_both_OLDI_and_Flores/   → Group A (fur, lij, lmo, sc, scn, vec)
│   ├── <code>.csv         → columns: text, label, article_id
│   ├── <code>_meta.csv    → columns: article_id, title, url, n_sentences
│   └── <code>_stats.json  → per-stage line counts
├── others_dialects/                    → Group B (lld, nap, pms, roa_tara)
│   └── (same three files per code)
├── languages/                          → comparison languages (ita/eng/fra/spa/cat/deu/ell/ara/slv)
│   ├── <iso>.csv          → columns: text, label, article_id (legacy Camposampiero)
│   └── <iso>_meta.csv     → columns: article_id, title, url, ...
├── _cache/                             → wikiextractor *_texts/ + .xml.bz2 (gitignored)
├── scripts/
│   ├── create.py                       → download + extract + invoke generation
│   └── generation.py                   → cleaning pipeline (§4)
└── PIPELINE.md                         → this file
```

`label` is a small integer (`fur=0, lij=1, lmo=2, sc=3, scn=4, vec=5`)
defined in `scripts/generation.py:FOLD_LABEL`. Downstream code can map
back to a code via `DIAL_LABEL`.

## 8. Reproduction

One command, after the project venv is set up and `it_core_news_sm` is
installed (the model is no longer used for splitting but is in the
requirements; see future improvements §10):

```bash
cd Dataset/wiki
python scripts/create.py
```

`create.py` will:
1. download the six dumps from Wikimedia for snapshot `2026-04-01`,
2. run `wikiextractor` for each,
3. invoke `scripts/generation.py` which produces the per-variety
   `.csv`, `_meta.csv`, and `_stats.json`,
4. delete only its own intermediates (`*.xml.bz2` and `*_texts/`),
   never touching pre-existing files of other contributors.

Tool versions (current):
- `wikiextractor==3.0.6`
- `spacy==3.5.3`
- Python 3.9 (matches the Bocconi HPC `stud` partition).

## 9. Final stats

Final per-variety sentence counts after the full pipeline
(snapshot 2026-04-01, all 9 stages including fingerprint dedup).
Output is split into two subfolders under `Dataset/wiki/`.

**Group A — `dialects_in_both_OLDI_and_Flores/`** (the 6 italo-romance
varieties that appear in BOTH OLDI and FLORES, our primary set):

| Variety | raw articles | after article dedup | raw sentences | after sentence filter | after exact dedup | **final** |
|---|---:|---:|---:|---:|---:|---:|
| fur | 4,979 | 4,678 | 30,228 | 23,980 | 23,770 | **22,956** |
| lij | 8,223 | 7,371 | 62,139 | 52,426 | 52,265 | **51,351** |
| lmo | 79,073 | 52,155 | 252,941 | 159,250 | 139,751 | **135,642** |
| sc  | 7,692 | 7,648 | 68,817 | 62,346 | 62,168 | **62,031** |
| scn | 23,457 | 19,373 | 94,075 | 81,827 | 80,688 | **78,570** |
| vec | 68,960 | 49,367 | 159,686 | 103,145 | 102,121 | **100,268** |

**Group B — `others_dialects/`** (italo-romance varieties on Wikipedia
that are NOT in OLDI; lld is in FLORES but not OLDI):

| Variety | raw articles | after article dedup | raw sentences | after sentence filter | after exact dedup | **final** |
|---|---:|---:|---:|---:|---:|---:|
| lld      | 176,204 | 166,503 | 385,088 | 359,578 | 212,755 | **212,539** |
| nap      |  12,887 |  11,835 |  32,688 |  28,965 |  28,416 | **27,716** |
| pms      |  71,072 |  69,397 | 270,038 | 111,011 | 100,911 | **98,965** |
| roa_tara |   9,230 |   8,101 |  34,744 |  30,315 |  29,963 | **29,963** |

The auto prefix-dedup at the end (Stage 7) drops a further 0.1–2.5%
beyond exact dedup, with the bigger cuts on lmo (template French
commune pages) and vec (province / commune templates SUKI/Camposampiero
miss).

Quality indicators on final output:
- **Sentence length**: median ~110-140 chars, average ~130-160. Only
  ~2% of sentences are below 40 chars now (since min length is 30,
  and most short fragments fail the terminator-punct filter).
- **HTML entity residue**: 0 for fur/lij/sc/scn, ≤3 for lmo and vec.
- **Lowercase-start sentences**: ≤0.3% per variety (the lowercase-merge
  in Stage 4 nearly eliminates splitter glitches).

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

- **Empirical boilerplate dictionaries for fur/lij/sc/scn**: SUKI did
  not document patterns for these. We can derive them by inspecting top
  n-grams in the per-variety CSVs and codifying the obvious templated
  fragments. Estimated effort: ~30 min/variety.
- **In-line preprocessing module** — done. See §12 below.
- **FLORES↔Wiki overlap check**: a quick sanity check that no FLORES
  devtest sentence appears verbatim in the cleaned Wiki training set
  (potential data leakage). Run via exact-string match, no model needed.
- **External validation on ITDI dev/test**: train a TF-IDF char
  classifier on our Wiki and evaluate on ITDI dev (6,799 sentences,
  with our 6 varieties present) — sanity check that the pipeline
  produces classifiable data and a number comparable to published
  baselines.
- **`it_core_news_sm` no longer required**: now that we use the
  rule-based sentencizer, the model is no longer used for sentence
  segmentation. We could remove it from `requirements.txt` and shrink
  the venv (~50 MB). Currently kept in case a contributor wants to
  switch back for ablation.

## 12. Normalization & loader API

The cleaning pipeline above keeps source files in their **native
orthography** (PIPELINE.md §5). For analysis-time consistency across
encoders and corpora, we expose two modules at the package root:

```
Dataset/
├── normalize.py     # composable text-normalization functions
├── loaders.py       # unified API over wiki / FLORES / OLDI
└── __init__.py
```

### Four normalization levels

Defined in `Dataset/normalize.py`. Monotone — each strictly extends the
previous.

| Level | What it does | Loss | Recommended for |
|---|---|---|---|
| `none` | identity | — | debug / diff with sources |
| `hygiene` | NFC + curly quotes/dashes → ASCII + Unicode whitespace → `" "` | none | char-level encoders (CANINE-c), default everywhere |
| `subword_safe` | hygiene + `ł→l, Ł→L` (only when `lang=="vec"`) | minimal (cross-orthography variation in vec) | subword encoders (XLM-R, mBERT, LaBSE, mT5); also when ITDI is in pipeline |
| `tfidf_safe` | subword_safe + strip roman numerals + strip digits + strip punctuation/symbols + collapse whitespace | strips all non-letter content | bag-of-features methods (TF-IDF, FastText, Word2Vec, Naive Bayes, KenLM) |

Per-encoder defaults (use this table directly when wiring a method):

| Method family | Default normalize |
|---|---|
| Char-level contextual (CANINE-c, ByT5) | `hygiene` |
| Subword contextual (XLM-R, mBERT, LaBSE, mUSE, mT5) | `subword_safe` |
| Bag-of-features (TF-IDF, FastText, Word2Vec, NB, KenLM) | `tfidf_safe` |

### Loader API

`Dataset/loaders.py` wraps the three corpora behind a uniform interface:

```python
from Dataset.loaders import load_wiki, load_flores, load_oldi
from Dataset.loaders import load_flores_parallel, list_supported

df = load_wiki("vec", normalize="subword_safe")        # Wiki Group A/B auto-routed
df = load_flores("vec", normalize="hygiene")           # FLORES+ one-sentence-per-row
df = load_oldi("vec", normalize="tfidf_safe")          # OLDI parquet
pp = load_flores_parallel(normalize="subword_safe")    # 16-language aligned (2009 rows)
list_supported()                                       # availability matrix
```

`lang` is always the **ISO 639-3 code** we use internally (`vec`, `fur`,
`lij`, `lmo`, `sc`, `scn`, `lld`, `nap`, `pms`, `roa_tara` for dialects;
`ita`, `eng`, `fra`, `spa`, `cat`, `deu`, `ell`, `ara`, `slv` for
comparison languages). The loaders translate to each corpus's native
naming convention (FLORES uses Italian names like `veneto.txt`, OLDI
uses BCP47 like `vec_Latn.parquet`, Sardinian's OLDI release uses the
macrolanguage code `srd`).

### Coverage

|  | Wiki | FLORES+ | OLDI |
|---|:---:|:---:|:---:|
| **Group A dialects** (fur, lij, lmo, sc, scn, vec) | ✓ | ✓ | ✓ |
| **Group B dialects** (lld, nap, pms, roa_tara) | ✓ | only `lld` | — |
| **Comparison Romance** (ita, fra, spa) | ✓ (legacy)¹ | ✓ | ✓ |
| **Comparison Romance** (cat) | ✓ (legacy)¹ | ✓ | — |
| **Non-Romance** (eng, deu, ell, ara, slv) | ✓ (legacy)¹ | ✓ | only `eng` |

¹ The Wiki CSVs for the 9 comparison languages live under
`wiki/languages/` and were produced with the **original Camposampiero
script** (legacy pipeline). They are ~30-50% smaller than a re-extraction
with the current Stage 1-9 pipeline would yield, but suffice for
comparison-language baselines where the same surgical cleaning is not
needed. To re-extract them with the current pipeline, add their dump
URLs to `scripts/create.py` and their `_texts` codes to `FOLD_LABEL`
in `scripts/generation.py`.

Comparison-language sentence counts (legacy Camposampiero output):

| Lang | sentences |
|---|---:|
| ita | 320,307 |
| deu | 360,568 |
| eng | 301,112 |
| fra | 294,414 |
| spa | 276,661 |
| slv | 249,497 |
| cat | 103,854 |
| ell |  18,257 |
| ara |  16,165 |

Expanding coverage is just a matter of adding entries to the `_FLORES_NAME`
or `_OLDI_NAME` dictionaries in `loaders.py` (no code change to the
loader functions themselves).

### Why runtime normalization (and not materialized variants)

We considered materializing `<lang>_norm.csv` files alongside the
sources, but the variant count explodes: 4 normalization levels × 4
sources × 19 languages = up to 304 files to keep in sync. Runtime
normalization via the loader API gives the same guarantees with one
source of truth and zero disk duplication. Source files therefore
remain native (`ł` preserved in `vec`, accents preserved everywhere).
