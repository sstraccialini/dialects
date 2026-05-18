# `Dataset/oldi/` — OLDI seed corpus

Open Language Data Initiative (OLDI) seed corpus: ~6,200 sentences
sampled from English Wikipedia and translated into many languages by
native speakers. Originally built to fill data gaps for low-resource
machine translation.

> Source: <https://huggingface.co/datasets/openlanguagedata/oldi_seed>

We use OLDI exclusively as a source of **Italian↔dialect parallel
data** for fine-tuning encoder models on the 6 Italo-Romance dialects.
For this reason the repo ships only the 7 OLDI varieties relevant to
that task (`ita` plus `fur`, `lij`, `lmo`, `sc`, `scn`, `vec`); other
OLDI languages have been pruned.

## Folder layout

```
Dataset/oldi/
├── README.md                  this file
├── before_cleaning/            raw HF originals, one parquet per variety
│   ├── normalized/             7 parquets + 6 ita↔dialect pair TSVs
│   │   ├── ita_Latn.parquet      Italian (human-translated)
│   │   ├── fur_Latn.parquet      \\
│   │   ├── lij_Latn.parquet       \\
│   │   ├── lmo_Latn.parquet        ─── 6 Italo-Romance dialects
│   │   ├── scn_Latn.parquet       /
│   │   ├── srd_Latn.parquet      /
│   │   ├── vec_Latn.parquet      (Sardinian is 'srd' in BCP47, 'sc' in our pipeline)
│   │   └── pairs_ita_<dialect>.tsv  one TSV per dialect (id, italiano, <dialect>)
│   └── not_normalized/         same set, with case/diacritics preserved
│
├── cleaned/                    cleaned single-CSV form used by every
│   └── oldi.csv                  experiment. Columns:
│                                  unique_id, dataset, original_id,
│                                  italiano, veneto, siciliano, lombardo,
│                                  sardo, ligure, friulano
│
├── cleaned_normalized/         same with aggressive normalization
│   └── oldi.csv                  (lowercase ASCII + spaces only)
│
└── scripts/
    ├── normalize.py              applies aggressive_normalize to the text
    │                               column of each parquet
    ├── translate_missing.py      legacy: backfilled Italian sides via
    │                               deep-translator when an alignment row
    │                               had Italian missing in HF
    ├── build_pairs.py            assembles pairs_ita_<dialect>.tsv from the
    │                               aligned parquets
    └── verify_against_hf.py      double-checks the local copy is
                                    byte-identical to the upstream HF release
```

## Parquet schema

Each parquet has the same schema as the upstream HF release:

| column | type | example |
|---|---|---|
| `id` | int | 0 … 6192 (aligns row-by-row across all OLDI variety parquets) |
| `iso_639_3` | str | `vec`, `ita`, … |
| `iso_15924` | str | `Latn` |
| `glottocode` | str | `vene1259` (Venetian), … |
| `text` | str | the (translated) sentence |
| `url` | str | URL of the source English Wikipedia article |
| `last_updated` | str | OLDI version tag |

The `id` column aligns across all parquets: row `i` in `vec_Latn.parquet`
is the OLDI translation of row `i` in any other variety's parquet.

## Pair files

`pairs_ita_<dialect>.tsv`, one per dialect, has 3 columns `(id, italiano, <dialect>)` aligned by `id`:

* `pairs_ita_friulano.tsv`
* `pairs_ita_ligure.tsv`
* `pairs_ita_lombardo.tsv`
* `pairs_ita_sardo.tsv`
* `pairs_ita_siciliano.tsv`
* `pairs_ita_veneto.tsv`

These are used by encoder fine-tuning experiments
(`analysis/canine/experiments/canine_finetuned_*`,
`analysis/multilingual_xlmr/experiments/xlmr_finetuned_*`,
`analysis/labse/experiments/labse_finetuned_*`) via
`analysis._shared.dataset_loaders.load_oldi_pairs`.

## Loading from Python

The canonical loaders are in `analysis/_shared/dataset_loaders.py`:

```python
from analysis._shared.dataset_loaders import load_oldi, load_oldi_pairs

# Italian + 6 dialects, sentence-aligned
data, stats = load_oldi(text_variant="native")          # or "normalized"
data["vec"]   # ~5,167 Venetian sentences

# Italian↔dialect pairs for fine-tuning
pairs = load_oldi_pairs(dialects=["fur", "lij", "lmo", "sc", "scn", "vec"])
pairs["vec"]  # list of (italian, venetian) tuples
```

## Re-running the cleaning step

```bash
python -m Dataset.oldi.scripts.normalize        # produce normalized/ from not_normalized/
python -m Dataset.oldi.scripts.build_pairs      # assemble pair TSVs
python -m Dataset.oldi.scripts.verify_against_hf  # sanity-check vs. HF
```

## License

OLDI seed corpus is released under **CC BY-SA 4.0**.

> Open Language Data Initiative (2024). *OLDI Seed Corpus*.
> <https://huggingface.co/datasets/openlanguagedata/oldi_seed>
