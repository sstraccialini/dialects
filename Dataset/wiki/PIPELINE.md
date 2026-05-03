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
wikiextractor → article-level clean → article dedup → sentence split → sentence filter → sentence dedup
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

### Stage 4 — Sentence splitting (rule-based)
- We use **spaCy's rule-based sentencizer**, not the Italian statistical
  parser:

  ```python
  NLP = spacy.blank("xx")
  NLP.add_pipe("sentencizer")
  ```

- Camposampiero originally used `it_core_news_sm` (the Italian parser)
  for sentence segmentation, but it is trained on Italian standard text
  and on dialectal text it makes systematic errors — sentences come out
  fragmented or merged at unnatural boundaries.
- The rule-based sentencizer splits on `.`, `!`, `?` with a few generic
  rules. It is deterministic, ~5-10× faster, and on dialectal text it
  produces sentences that are **mediana 12-20 characters longer** and
  more faithful to natural sentence boundaries (verified empirically:
  comparing the same vec corpus split with the parser vs the sentencizer,
  the latter produces fewer, longer, more complete sentences).
- The downside is that the sentencizer cannot tell that abbreviations
  ending with `.` (e.g. `Art. 5`) are not full stops. The downstream
  length and lowercase-word filters catch most of the resulting
  fragments.

### Stage 5 — Sentence-level filters
- **Drop sentences shorter than 20 characters** (Camposampiero/ETHZ +
  SUKI both apply this).
- **Drop sentences without any lowercase ASCII letter** (SUKI):
  removes pure-numeric / all-caps fragments.
- **Drop sentences without a word starting with lowercase ASCII**
  (SUKI): catches single-headings.
- **Per-variety filters**:
  - **VEC** — five drops (no orthographic normalization, see §5):
    - SUKI: French commune template
      `el xe on comun de.*abitanti del departemento.*in Fransa\.`
      (33,701 lines in SUKI; biggest single boilerplate cluster of vec).
    - SUKI: Italian commune template
      `el xe (?:on|un) comun italian de.*abitanti`.
    - SUKI: Roman numbers stub
      `\(L?[IVXC]+\s+(?:en|in)\s+numeri\s+romani\)`.
    - Camposampiero: generic-comun substring `el xe un comun de`.
    - Camposampiero: calendar-stub substring `gregorian`.
  - **LMO** — combination of SUKI and Camposampiero:
    - SUKI: drop lines shorter than 14 characters.
    - Camposampiero: 24 substring patterns covering geographic /
      template stubs (`El cumün`, `La Stazzion de`, `km²`,
      `El Distret`, `Al gh'ha pressapoch abitant`,
      `L'andament del numer de abitant`, ...). Full list in
      `scripts/generation.py:LMO_CAMPOSAMPIERO_SUBSTRINGS`.
  - **fur, lij, sc, scn** — no per-variety filter at the moment;
    SUKI does not document custom rules for these and we have not
    derived empirical ones yet. To add them later: dump top n-grams
    of `wiki_old/<lang>.csv`, inspect manually, codify the obvious
    template patterns.

### Stage 6 — Sentence-level deduplication
- `drop_duplicates(subset="text", keep="first")` — keep the first
  occurrence and drop subsequent duplicates.
- A previous version of this pipeline used `keep=False` like
  Camposampiero, but at sentence level that turned out too aggressive:
  legitimate repeated sentences (e.g. *"Inoltre, è un noto attore."*)
  appearing in two unrelated articles got removed entirely. The
  boilerplate volume is already controlled by Stages 2, 3, and 5 —
  at this point a soft dedup is enough.

### Stage 7 — Save outputs (atomic)
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

### Diagnostic comparison with `wiki_old/`

The `wiki_old/` snapshot was produced with the original Camposampiero
script. Our pipeline produces **1.4–2.0×** as many sentences and
**1.6–2.8×** as much character content for the same six varieties on
the same Wikimedia snapshot (2026-04-01). This is **not** because we
loaded a different dump or kept duplicates: the article counts agree
within 5% (vec is the only outlier at 1.27× — a side-effect of the
two greedy-regex bugs combined with `text.replace('"', " ")` having a
larger impact on Venetian where `"` and parentheses are more frequent).

The 1.5–2.0× content gain per article is essentially **legitimate text
that the old greedy regex was wrongly deleting**. Verified by
inspecting article-level diffs and per-article character counts (vec
new: 978 chars/article; old: 540 chars/article). The new pipeline is
therefore **strictly better** than `wiki_old/` for the same input
data, not just a different partitioning of it.

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
| Sentence length filter `>20` (Stage 5) | Camposampiero/ETHZ + SUKI |
| HAS_LOWER_ASCII filter (Stage 5) | SUKI |
| HAS_WORD_LOWER filter (Stage 5) | SUKI |
| VEC French/Italian commune + Roman regexes (Stage 5) | SUKI |
| VEC `el xe un comun` + `gregorian` substrings (Stage 5) | Camposampiero/ETHZ |
| LMO `len < 14` filter (Stage 5) | SUKI |
| LMO 24 boilerplate substrings (Stage 5) | Camposampiero/ETHZ |
| Sentence-level dedup `keep="first"` (Stage 6) | ours (Camposampiero used `keep=False`, too aggressive) |
| Atomic save + per-stage `_stats.json` | ours |

## 7. Output format

```text
Dataset/wiki/
├── <code>.csv          → columns: text, label, article_id
├── <code>_meta.csv     → columns: article_id, title, url, n_sentences
├── <code>_stats.json   → per-stage line counts
└── wiki_old/           → previous extraction (legacy, kept for diff)
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

Expected line counts at each stage (snapshot 2026-04-01,
sentencizer + article-dedup keep=False):

| Variety | raw articles | after clean | after article dedup | raw sentences | after sentence filter | **final sentences** |
|---|---:|---:|---:|---:|---:|---:|
| fur | 4,979 | 4,680 | 4,678 | 30,601 | 25,911 | **25,392** |
| lij | 8,223 | 7,371 | 7,371 | 62,993 | 55,404 | **55,102** |
| lmo | 79,073 | 75,575 | 52,155 | 255,048 | 175,248 | **146,829** |
| sc  | 7,692 | 7,648 | 7,648 | 69,617 | 64,920 | **64,666** |
| scn | 23,457 | 21,767 | 19,373 | 95,679 | 87,708 | **86,184** |
| vec | 68,960 | 68,670 | 49,367 | 161,341 | 112,742 | **111,320** |

Sentence length: median 105-141 chars, average 125-160. Only **2-3% of
sentences are under 30 characters** (residual fragments).

HTML entity residue (post-pipeline): 0 for fur/lij/sc/scn, 3 for lmo,
2 for vec — trivial, likely literal `&abc;` strings rather than
standard entities.

Comparison with `wiki_old/` (legacy Camposampiero output, same
Wikimedia snapshot 2026-04-01):

| Variety | new sentences | wiki_old sentences | new chars | old chars | char ratio |
|---|---:|---:|---:|---:|---:|
| fur | 25,392 | 16,530 | 3,295,272 | 1,723,973 | 1.91× |
| lij | 55,102 | 26,615 | 8,798,427 | 3,165,564 | 2.78× |
| lmo | 146,829 | 105,153 | 19,752,649 | 12,347,241 | 1.60× |
| sc  | 64,666 | 35,899 | 10,281,975 | 4,579,203 | 2.25× |
| scn | 86,184 | 60,373 | 10,777,050 | 6,142,725 | 1.75× |
| vec | 111,320 | 57,271 | 15,010,440 | 6,539,719 | 2.30× |

The 1.6–2.8× gain is content the legacy script was wrongly removing
(see §5: greedy-regex bug + quote-stripping). Article counts agree to
~5% across varieties, confirming that the input dumps are equivalent.

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
  n-grams in `wiki_old/<lang>.csv` and codifying the obvious templated
  fragments. Estimated effort: ~30 min/variety.
- **Optional in-line preprocessing module** (`evaluation/normalization.py`):
  for subword encoders that suffer from rare characters (e.g. XLM-R on
  `ł`), we apply orthographic normalization consistently to **both**
  training and eval inputs at runtime. The source `.csv` files remain
  untouched.
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
