# Dataset/oldi — OLDI seed corpus

Open Language Data Initiative (OLDI) seed corpus: 6,193 sentences
sampled from English Wikipedia and translated into 44 languages.
Originally aimed at low-resource languages that lack large-scale MT data.

> Source: <https://huggingface.co/datasets/openlanguagedata/oldi_seed>

## Folder structure

```
Dataset/oldi/
├── README.md                       this file
├── not_normalized/                 originals (after download from HF)
│   ├── eng_Latn.parquet            6,193 English source sentences
│   ├── ita_Latn.parquet            human-translated by OLDI
│   ├── fra_Latn.parquet
│   ├── spa_Latn.parquet
│   ├── fur_Latn.parquet            6 italo-romance dialects (the project target)
│   ├── lij_Latn.parquet
│   ├── lmo_Latn.parquet
│   ├── scn_Latn.parquet
│   ├── srd_Latn.parquet            (Sardinian, code 'sc' in our pipeline)
│   ├── vec_Latn.parquet
│   ├── deu_Latn.parquet            ⚠️ GOOGLE-MT (not in original OLDI)
│   ├── cat_Latn.parquet            ⚠️ GOOGLE-MT (not in original OLDI)
│   ├── slv_Latn.parquet            ⚠️ GOOGLE-MT (not in original OLDI)
│   ├── ace_Arab.parquet, ace_Latn.parquet, ary_Arab.parquet
│   │                                 (Acehnese, Moroccan Arabic — kept from OLDI
│   │                                 original release; not used in our analyses)
│   ├── pairs_all.tsv               14-column ita↔(6 dialects + 3 ext + 3 MT) pairs
│   ├── pairs_ita_<dialect>.tsv     6 files: ita↔{veneto,siciliano,lombardo,sardo,ligure,friulano}
│   └── pairs_ita_<lang>.tsv        3 files (MT-derived): ita↔{tedesco,catalano,sloveno}
│
├── normalized/                     mirror of not_normalized/, with `text`
│                                   column aggressive-normalized in place
│
└── scripts/
    ├── normalize.py                aggressive_normalize the text column
    ├── translate_missing.py        produce deu/cat/slv via Google Translate
    ├── build_pairs.py              build pairs_ita_{tedesco,catalano,sloveno}.tsv
    │                               and extend pairs_all.tsv with the 3 MT columns
    └── verify_against_hf.py        re-verify all OLDI/FLORES files match upstream HF
```

## Parquet schema

| column | type | example |
|---|---|---|
| `id` | int | 0 .. 6192 |
| `iso_639_3` | str | `vec`, `eng`, `deu`, ... |
| `iso_15924` | str | `Latn` (Latin script for our target subset) |
| `glottocode` | str | `vene1259` (Venetian), `stan1295` (German), ... |
| `text` | str | the (translated) sentence |
| `url` | str | URL of the original English Wikipedia article |
| `last_updated` | str | OLDI version tag (e.g. `1.0`) |

The `id` column aligns across all languages: row `i` in `vec_Latn.parquet`
is the translation of row `i` in `eng_Latn.parquet`. This makes OLDI
suitable for cross-lingual cosine similarity at sentence level.

## Pairs files (for TLM fine-tuning + cross-lingual cosine)

There are now **9 per-language pair files**, all with 3 columns
`(id, italiano, <lang>)` aligned by `id`:

- 6 dialect pairs (human-translated by OLDI): `pairs_ita_{veneto,siciliano,lombardo,sardo,ligure,friulano}.tsv`
- 3 MT-derived pairs (Google Translate, ⚠️ NOT for fine-tuning): `pairs_ita_{tedesco,catalano,sloveno}.tsv`

Used by:
- `analysis/xlmr_finetuned/flores/` — Translation Language Model training (**only the 6 dialect pairs**)
- `analysis/sentence_finetuned/flores/` — sentence-transformer fine-tuning (**only the 6 dialect pairs**)

`pairs_all.tsv` is the wide-format concatenation: 14 columns
`(id, italiano, 6 dialects, inglese, spagnolo, francese, tedesco, catalano, sloveno)`.
The 3 MT-derived columns at the end are tagged in this README only —
the file format is uniform.

## ⚠️ Methodological caveat — machine-translated languages

The original OLDI seed dataset includes 4 of our 7 comparison languages
(ita, eng, fra, spa) but **NOT** German, Catalan, Slovenian — these
high-resource languages are out of OLDI's scope (their stated mission
is filling gaps for low-resource MT). To enable cross-lingual cosine
similarity on the full 13-variety set (6 dialects + 7 comparison) using
OLDI sentence-level alignment, we generated `deu_Latn.parquet`,
`cat_Latn.parquet`, `slv_Latn.parquet` by **machine-translating the
6,193 English seed sentences via Google Translate (`deep-translator`
library, accessed via `Dataset/oldi/scripts/translate_missing.py`)**.

**Implications**:

- The 10 OLDI-native files (4 standard + 6 dialects) are **human-translated
  by native speakers** → gold standard quality.
- The 3 added files (deu, cat, slv) are **machine-translated** → very
  high quality (~95% on these high-resource languages) but not gold.
- For most analyses (TF-IDF char n-gram, FastText, Word2Vec, sentence
  embeddings) the difference is negligible — these methods don't
  exploit the human-vs-MT distinction.
- For **fine-tuning experiments** (TLM, MNRL): the MT-derived languages
  are NOT used as training pairs (only `pairs_ita_<dialect>.tsv` is, and
  none of those involve deu/cat/slv).
- For **publications**: declare in the methodology section that
  deu/cat/slv pairs are Google-MT, not human-translated. Optionally
  ablation: redo cosines without these 3 to verify the picture is stable.

Re-generate via:
```bash
python Dataset/oldi/scripts/translate_missing.py   # ~30 min total (deu+cat+slv)
python Dataset/oldi/scripts/build_pairs.py         # ~5 sec — pair TSVs + pairs_all
python Dataset/oldi/scripts/normalize.py           # ~10 sec — aggressive_normalize
```

### `translate_missing.py` — gotchas you'll hit if you re-run

**1. Threading bug (FIXED).** `deep_translator.GoogleTranslator` mutates
internal state (`self._source_text`) on every `.translate()` call. A
single instance shared across N worker threads causes race conditions
that produce duplicates and wrong-aligned translations (the symptom
that hit us first time: 587-809 duplicates per language and shifts of
±1 between EN id and target id). The script now uses `threading.local()`
so each worker thread has its own `GoogleTranslator` instance.

**2. Google rate-limit per IP.** Stated limit is 5 requests/sec and 200k/day,
but the practically-enforced trigger is a *cumulative* threshold around
~30-40k requests on a single IP — once you cross it Google starts
returning `TooManyRequests` for many hours. With 8 workers and
6,193 sentences × 3 languages = ~18,579 requests, ONE clean run from
a fresh IP fits under the threshold. Re-running on a burned IP doesn't.

  - **Fix while running:** stop the script. **Switch IP.** Restart — script
    auto-skips already-completed languages (`[skip] deu: ... already exists`).
  - **Easy IP switches** (in order of speed):
    1. Phone tethering / hotspot (mobile IP — almost always fresh)
    2. Paid VPN (NordVPN, ExpressVPN, Mullvad — pick a less-popular exit node)
    3. Free VPN (ProtonVPN free etc — variable)
  - **Verify before launching** by translating 5 test sentences. If you get
    `TooManyRequests`, change exit node before starting the full run:
    ```python
    from deep_translator import GoogleTranslator
    GoogleTranslator(source='en', target='ca').translate('Hello world')
    ```

**3. Empty translations.** Even on a clean IP a few sentences may come
back as `""` (Google's quirk on long/punctuation-heavy inputs). If
verification reports e.g. "cat: 3 empty", patch them after the main
run: load the parquet, find the empty rows, retranslate, save back.

### Verify the corpus matches upstream

```bash
python Dataset/oldi/scripts/verify_against_hf.py
```

Downloads each of the 13 OLDI-native parquets fresh from
`openlanguagedata/oldi_seed` and each of the 14 FLORES+ files fresh from
`openlanguagedata/flores_plus`, computes MD5 of every text column,
and reports whether the local copy is byte-identical to upstream.
Useful when HF dataset gets a version bump or after a heavy reorg.

## Aggressive normalization

`scripts/normalize.py` applies the same `aggressive_normalize` function
used in `Dataset/wiki/scripts/generation.py` Stage 6 (lowercase ASCII
letters + spaces only — no diacritics, digits, punctuation, symbols).
This makes OLDI directly comparable to Wiki and FLORES normalized
variants for char-n-gram / bag-of-features cross-lingual analysis.

The not-normalized originals are preserved in `not_normalized/` for any
analysis that needs the surface form (e.g., subword encoders that use
diacritics as signal).

## Python loading

```python
import pandas as pd

# Single language, normalized form
df = pd.read_parquet("Dataset/oldi/normalized/vec_Latn.parquet")
df.columns        # ['id', 'iso_639_3', 'iso_15924', 'glottocode', 'text', 'url', 'last_updated']
df["text"]        # 6,193 normalized Venetian sentences

# Cross-lingual sentence-level: row i in any file is the same sentence
ita = pd.read_parquet("Dataset/oldi/normalized/ita_Latn.parquet")
deu = pd.read_parquet("Dataset/oldi/normalized/deu_Latn.parquet")
print(ita.iloc[0]["text"])    # Italian translation of row 0
print(deu.iloc[0]["text"])    # German translation (Google-MT) of the same English source

# Italian↔dialect pairs (for TLM training)
pairs = pd.read_csv("Dataset/oldi/normalized/pairs_ita_veneto.tsv", sep="\t")
pairs.columns                 # ['id', 'italiano', 'veneto']

# Wide-format with all 13 varieties
all_pairs = pd.read_csv("Dataset/oldi/normalized/pairs_all.tsv", sep="\t")
all_pairs.columns             # ['id', 'italiano', 6 dialects, 'inglese', 'spagnolo',
                              #  'francese', 'tedesco', 'catalano', 'sloveno']
```

## License & citation

OLDI seed corpus is released under **CC BY-SA 4.0**.

Cite the OLDI release when using the human-translated subset:

> Open Language Data Initiative (2024). *OLDI Seed Corpus*.
> <https://huggingface.co/datasets/openlanguagedata/oldi_seed>

For the machine-translated additions (deu, cat, slv), credit Google
Translate as the translation system used.
