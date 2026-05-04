# TF-IDF Baseline (Person 1)

TF-IDF baseline for the project "Modeling Historical and Linguistic
Relations Between Italian Dialects and Contemporary Languages Through
Embedding Spaces".

This sub-folder contains the TF-IDF pipeline with two vectorizers:

1. **Word n-grams** (1-2): word-level unigrams and bigrams.
2. **Character n-grams** (3-5, `char_wb`): character n-grams within
   word boundaries. Particularly important for dialects, where
   orthographic and phonetic variation are captured at the character
   level.

## Goal

For each of the 14 varieties, produce a TF-IDF vector and compute the
14x14 cosine distance matrix. This matrix serves as a **reference
baseline** for the more sophisticated vectorizers used by the other
team members (Word2Vec, SBERT, mBERT/XLM-R, FastText).

## Design decisions

All documented (with motivation) in `config.py`. Summary:

| Choice | Value | Motivation |
|---|---|---|
| Aggregation | One document per variety | Standard in computational dialectology, interpretable |
| Sample | 16k sentences / variety (default) | Minimum common (ar ~16k), fair comparison |
| Random state | 42 | Reproducibility |
| Lowercase | Yes | Standard |
| Mask numbers | Yes (`<NUM>`) | Avoids bias from Wikipedia dates |
| Punctuation | Stripped for word, kept for char | Apostrophes are distinctive in dialects |
| Diacritics | Kept | Accents and spellings are distinctive dialect traits |
| Char n-gram | (3, 5), `char_wb` | Standard VarDial / language ID |
| Word n-gram | (1, 2) | Unigrams + bigrams |
| `sublinear_tf` | True | Dampens Wikipedia hyper-frequent terms |
| `min_df` / `max_df` | 1 / 1.0 | We keep variety-unique features (they ARE the signal) |
| `norm` | l2 | Stable cosine distance |

## How to run

From the repo root, with the project's `venv` activated:

```powershell
# First time only (adds sklearn + scipy + matplotlib)
pip install -r tfidf_baseline/requirements.txt

# Run the full baseline
python tfidf_baseline/src/run_baseline.py

# Sensitivity check with a different sample size
python tfidf_baseline/src/run_baseline.py --sample-size 50000

# Run only one pipeline
python tfidf_baseline/src/run_baseline.py --pipeline char
```

## Structure

```
tfidf_baseline/
|- README.md              this file
|- requirements.txt       extra dependencies (sklearn/scipy/matplotlib)
|- src/                   all Python sources
|   |- config.py          paths, hyperparameters, varieties list
|   |- data_loader.py     loads CSVs, sub-samples to N sentences/variety
|   |- preprocess.py      lowercase, mask numbers, strip punct (word only)
|   |- vectorize.py       TfidfVectorizer word + char, fit + transform
|   |- similarity.py      14x14 cosine distance matrix + nearest neighbors
|   |- cluster.py         hierarchical clustering, silhouette, dendrogram
|   |- visualize.py       MDS / t-SNE 2D projections
|   |- run_baseline.py    end-to-end orchestrator
|- results/               outputs, split by pipeline
|   |- word/              word n-gram pipeline outputs
|   |- char/              char n-gram pipeline outputs
|   |- shared/            cross-pipeline outputs
|- notebooks/             interactive exploration (optional)
```

## Expected outputs in `results/`

`word/` and `char/` each contain:
- `distances.csv` (14x14 cosine distance)
- `top_features.csv` (top-30 features per variety)
- `nearest_neighbors.csv` (top-3 nearest per variety)
- `dendrogram.png`
- `projection_mds.png`, `projection_tsne.png`

`shared/` contains:
- `silhouette_report.txt`
- `run_stats.csv`

## What to look for in the output

The baseline should:

1. Separate Romance (italo_romance + italian + romance) from non-Romance.
2. Place `Italian` near the Italo-Romance cluster (it is its standard form).
3. Put `Catalan` between `Spanish` and `Italian` / Italo-Romance.
4. Keep `Arabic` and `Greek` as outliers (different scripts mean nearly
   orthogonal char n-gram spaces).
5. Show char n-grams beating word n-grams (tighter cluster structure,
   higher silhouette for the romance-vs-rest label).

What the baseline will NOT capture (and what the other models should):
- Historical contact signals across different scripts (Arabic influence
  in Sicilian, Greek in Neapolitan, etc.).
- Deep structural / syntactic similarity.
- Cross-lingual semantic alignment.

These are precisely the dimensions where Word2Vec / SBERT / mBERT /
FastText are expected to go beyond the surface signal of TF-IDF.
