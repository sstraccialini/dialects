# Sentence-embedding baseline (Wikipedia)

Multilingual sentence-embedding pipeline on Wikipedia data.
Mirrors the structure of the other four `analysis_wiki/` methods
(TF-IDF, Word2Vec, subword/FastText, multilingual) so the outputs
are directly comparable.

## Method

1. Load per-variety CSVs from `wiki_data/{code}.csv` preserving the
   `article_id` column (`data_loader.load_all_varieties_with_article_ids`).
2. Encode every sampled sentence with a pretrained multilingual
   sentence-transformer — by default
   `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
3. Average sentence embeddings inside each `article_id`, then average
   article embeddings to obtain one vector per variety. This two-stage
   aggregation is more robust than a flat mean over ~16k unrelated
   sentences because each article gets equal weight regardless of how
   many sentences it produced.
4. Compute the cosine distance matrix across varieties.
5. Cluster (hierarchical, average linkage) + silhouette score.
6. Save 2D projections (MDS + t-SNE), a dialect-vs-language similarity
   table and ranked nearest-language table.

## Layout

```
analysis_wiki/sentence_baseline/
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
cd analysis_wiki/sentence_baseline/src
python run_sentence_baseline.py
```

Outputs land in `analysis_wiki/sentence_baseline/results/sentence/`
(plus `results/shared/silhouette_report.txt`).

## Configuration

All hyperparameters live in `src/config.py`:

- `SAMPLE_SIZE` (default 16000) — sentences per variety
- `SENTENCE_MODEL` — backbone checkpoint
- `ARTICLE_AGGREGATION` / `VARIETY_AGGREGATION` — `"mean"` or `"median"`
- `DIALECT_CODES` / `MODERN_LANGUAGE_CODES` — subsets used in the
  dialect-vs-language similarity table
