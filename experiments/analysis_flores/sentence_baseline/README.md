# Sentence-embedding baseline (FLORES+)

Multilingual sentence-embedding pipeline on FLORES+ parallel data.
Mirrors the structure of the other four `analysis_flores/` methods
(TF-IDF, Word2Vec, subword/FastText, multilingual) so the outputs
are directly comparable.

## Method

1. Load 2009 parallel sentences per variety (997 dev + 1012 devtest)
   from `flores_data/flores_plus/<slug>.txt`.
2. Encode every sentence with a pretrained multilingual
   sentence-transformer — by default
   `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
3. Average the (L2-normalised) sentence embeddings to obtain one
   vector per variety. Sentence parallelism across varieties is the
   strong anchor here — position *i* in every file is the translation
   of the same source sentence.
4. Compute the cosine distance matrix across varieties.
5. Cluster (hierarchical, average linkage) + silhouette score.
6. Save 2D projections (MDS + t-SNE), a dialect-vs-language similarity
   table and ranked nearest-language table.

## Layout

```
analysis_flores/sentence_baseline/
├── README.md
├── requirements.txt
├── results/
│   ├── sentence/
│   └── shared/
└── src/
    ├── config.py
    ├── data_loader.py
    ├── sentence_vectorize.py
    ├── similarity.py
    ├── cluster.py
    ├── visualize.py
    └── run_sentence_baseline.py
```

## How to run

```bash
cd analysis_flores/sentence_baseline/src
python run_sentence_baseline.py
```

Outputs land in `analysis_flores/sentence_baseline/results/sentence/`
(plus `results/shared/silhouette_report.txt`).

## Configuration

All hyperparameters live in `src/config.py`:

- `SAMPLE_SIZE` (default 2009) — sentences per variety
- `SENTENCE_MODEL` — backbone checkpoint
- `VARIETY_AGGREGATION` — `"mean"` or `"median"`
- `DIALECT_CODES` / `MODERN_LANGUAGE_CODES` — subsets used in the
  dialect-vs-language similarity table
