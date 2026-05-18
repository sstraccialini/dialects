# `Dataset/flores/` — FLORES+ parallel evaluation set

FLORES+ is a professionally translated parallel corpus from the Open
Language Data Initiative: 2009 raw sentences per variety (997 `dev`
+ 1012 `devtest`). We use the union of the two splits as the
**shared evaluation set** across every embedding method, and ship
the cleaned form with **1827** sentences per variety after manual
review (see the *Methodological note* below for the cleaning rules).

> Source: <https://huggingface.co/datasets/openlanguagedata/flores_plus>

The 17 varieties we keep in the paper are the 6 Italo-Romance dialects
(Friulian, Ligurian, Lombard, Sardinian, Sicilian, Venetian) plus
11 standards: Italian, Spanish, French, Catalan, Portuguese, Occitan,
German, English, Slovenian, Croatian, Hungarian.

## Folder layout

```
Dataset/flores/
├── README.md                   this file
├── stats.csv                   per-variety length stats
├── before_cleaning/            raw FLORES+ (dev+devtest union) as downloaded
│   ├── normalized/             from HF; one .txt per variety, plus parallel.tsv
│   │   ├── <variety>.txt         2009 sentences, one per line
│   │   └── parallel.tsv          wide-format: sentence_id + split + one column per variety
│   └── not_normalized/         same, with case/diacritics preserved
│
├── cleaned/                    cleaned single-CSV form used by every
│   └── flores.csv                experiment, 1827 rows after manual review. Columns
│                                  in file order:
│                                  unique_id, dataset, original_id,
│                                  italiano, veneto, siciliano, lombardo,
│                                  sardo, ligure, friulano,
│                                  inglese, spagnolo, francese, tedesco, catalano,
│                                  sloveno, croato, occitano, portoghese, ungherese
│
├── cleaned_normalized/         same as cleaned/ but with aggressive
│   └── flores.csv                normalization (lowercase ASCII + spaces)
│
└── scripts/
    ├── download_flores.py        downloads the HF release, writes the .txt
    │                               files + parallel.tsv
    ├── normalize.py              applies aggressive_normalize to produce the
    │                               normalized variant
    ├── complete_cleaned_columns.ipynb
    │                             stitches the per-variety files into the
    │                               single CSV used by experiments
    └── requirements.txt          isolated requirements for the scripts
```

## Variety codes

The Italian-name slug (e.g. `friulano`) is the column name in the cleaned
CSVs and the filename in `before_cleaning/`. The ISO 639-3 code is what
the rest of the pipeline uses. The mapping lives in
`analysis/_shared/varieties.py::FLORES_SLUG`.

| ISO | Slug | English |
|---|---|---|
| fur | friulano | Friulian |
| lij | ligure | Ligurian |
| lmo | lombardo | Lombard |
| sc  | sardo | Sardinian |
| scn | siciliano | Sicilian |
| vec | veneto | Venetian |
| ita | italiano | Italian |
| spa | spagnolo | Spanish |
| fra | francese | French |
| cat | catalano | Catalan |
| por | portoghese | Portuguese |
| oci | occitano | Occitan |
| deu | tedesco | German |
| eng | inglese | English |
| slv | sloveno | Slovenian |
| hrv | croato | Croatian |
| hun | ungherese | Hungarian |

## Methodological note — splits and cleaning

FLORES+ originally separates `dev` (tuning, 997 sentences) from `devtest`
(final test, 1012 sentences) to avoid overfitting when training MT
models. For distance-based analysis the distinction is not needed: we
union the splits to get 2009 sentences of statistical power. The
`split` column in `before_cleaning/.../parallel.tsv` preserves the
original split if ever needed.

The 2009-sentence union is the starting point. Manual review (described
in the paper, Section *Cleaning and normalization*) drops sentences
that carry no usable linguistic signal — too-short fragments,
boilerplate, content dominated by proper nouns/foreign quotations, and
sentences localised only on the Italian side. After review the cleaned
CSV contains **1827** sentences per variety.

## Re-downloading

```bash
source venv/bin/activate
python -m pip install -r Dataset/flores/scripts/requirements.txt
hf auth login                                # the first time only
python Dataset/flores/scripts/download_flores.py
python Dataset/flores/scripts/normalize.py
# then complete_cleaned_columns.ipynb stitches per-variety files into the CSV
```

## Loading from Python

The canonical loaders are in `analysis/_shared/dataset_loaders.py`:

```python
from analysis._shared.dataset_loaders import load_flores
data, stats = load_flores(text_variant="native")        # or "normalized"
data["vec"]   # 1827 Venetian sentences, aligned with every other variety
```

If you need the raw per-variety files instead of the cleaned CSV,
they live under `before_cleaning/{normalized,not_normalized}/<slug>.txt`.

## License

FLORES+ is released under **CC BY-SA 4.0** by the Open Language Data
Initiative.

> Open Language Data Initiative (2024). *FLORES+*.
> <https://huggingface.co/datasets/openlanguagedata/flores_plus>
