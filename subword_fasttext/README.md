# Subword / FastText Approach (Person 5)

Subword and BPE-based embedding approach for the project
"Modeling Historical and Linguistic Relations Between Italian Dialects
and Contemporary Languages Through Embedding Spaces".

This sub-folder contains two complementary pipelines:

1. **FastText** — gensim FastText with character n-gram subwords
   (skip-gram, min_n=3, max_n=6, vector_size=200). A single model is
   trained on all 14 varieties; each variety is represented as the
   mean of its sentence vectors (mean-pooled word embeddings).
2. **BPE** — SentencePiece Byte-Pair Encoding tokenization
   (vocab_size=8000) followed by TF-IDF on BPE pieces. Each variety
   is represented as a TF-IDF vector over its BPE-tokenized text.

## Why subword models?

Italian dialects exhibit strong morphological variation:
- vowel alternations (`parlare` / `parrà` / `parlari`)
- suffix variation (`-zione` → `-zzione`, `-ssione`)
- dialectal clitics and particles

**FastText** handles morphology natively via character n-gram hashing:
even unseen words receive a meaningful embedding from their character
substrings. This is critical for low-resource dialects where rare forms
would be OOV in a standard word2vec model.

**BPE** discovers shared subword units across languages without any
prior linguistic knowledge. The same BPE piece (e.g. `▁parl`) will
appear in Italian, Neapolitan, Sicilian, and Spanish, making the TF-IDF
vector directly comparable across varieties.

## Design decisions

| Choice | Value | Motivation |
|---|---|---|
| FastText architecture | skip-gram (`sg=1`) | Better for rare words and morphologically diverse corpora |
| FastText char n-grams | min_n=3, max_n=6 | Covers typical Romance morpheme sizes |
| FastText vector size | 200 | Standard in FastText literature |
| FastText training | Shared model (all 14 varieties) | Per-variety model would be undertrained (~16k sentences each) |
| FastText variety repr. | Mean of sentence vectors | Simple, effective aggregation |
| BPE vocab size | 8000 | Large enough for 14 languages; small enough to stay interpretable |
| BPE + TF-IDF | Same params as Person 1 word pipeline | Isolates the tokenization unit as the only variable |
| Preprocessing | Lowercase + mask numbers + keep diacritics/punct | Same as Person 1; subword models benefit from punctuation and diacritics |

All hyperparameters are centralized in `src/config.py`.

## Can this run locally?

**Yes.** Both pipelines are designed to run on a standard laptop:

| Step | Estimated time (MacBook M1/M2) |
|---|---|
| Data loading (16k × 14 varieties) | ~30 s |
| FastText training (10 epochs, ~224k sentences) | ~3–8 min |
| BPE training (sentencepiece, 8k vocab) | ~30 s |
| BPE TF-IDF + distance + clustering | ~30 s |

No GPU required. Total: approximately 5–10 minutes.

## How to run

From the repo root, with the project's virtual environment activated:

```bash
# First time only: install extra dependencies
pip install -r subword_fasttext/requirements.txt

# Run both pipelines (default)
python subword_fasttext/src/run_approach.py

# Run only FastText
python subword_fasttext/src/run_approach.py --pipeline fasttext

# Run only BPE
python subword_fasttext/src/run_approach.py --pipeline bpe

# Sensitivity check with a larger sample
python subword_fasttext/src/run_approach.py --sample-size 50000
```

## Structure

```
subword_fasttext/
├── README.md               this file
├── requirements.txt        extra dependencies
├── INTERPRETATION_RESULTS.md  filled after running
├── src/
│   ├── config.py           paths, hyperparameters, variety list
│   ├── data_loader.py      loads CSVs, sub-samples to N sentences/variety
│   ├── preprocess.py       lowercase, mask numbers (keeps punct + diacritics)
│   ├── embed_fasttext.py   gensim FastText training + variety embeddings
│   ├── embed_bpe.py        sentencepiece BPE training + TF-IDF on BPE pieces
│   ├── similarity.py       cosine distance matrix + nearest neighbors
│   ├── cluster.py          hierarchical clustering, silhouette, dendrogram
│   ├── visualize.py        MDS / t-SNE 2D projections
│   └── run_approach.py     end-to-end orchestrator
└── results/
    ├── fasttext/           FastText pipeline outputs
    ├── bpe/                BPE pipeline outputs
    ├── shared/             cross-pipeline outputs (silhouette, run stats)
    └── models/             trained model artefacts
```

## Expected outputs

### `results/fasttext/`
| File | Description |
|---|---|
| `distances.csv` | 14 × 14 cosine distance matrix |
| `nearest_neighbors.csv` | Top-3 nearest neighbors per variety |
| `variety_vectors.csv` | 14 × 200 mean-pooled FastText vectors |
| `dendrogram.png` | Hierarchical clustering dendrogram |
| `projection_mds.png` | MDS 2D projection |
| `projection_tsne.png` | t-SNE 2D projection |

### `results/bpe/`
| File | Description |
|---|---|
| `distances.csv` | 14 × 14 cosine distance matrix |
| `nearest_neighbors.csv` | Top-3 nearest neighbors per variety |
| `top_features.csv` | Top-30 BPE pieces per variety (TF-IDF weight) |
| `dendrogram.png` | Hierarchical clustering dendrogram |
| `projection_mds.png` | MDS 2D projection |
| `projection_tsne.png` | t-SNE 2D projection |

### `results/shared/`
- `silhouette_report.txt` — silhouette scores for both pipelines
- `run_stats.csv` — sample sizes and random seed

### `results/models/`
- `fasttext_model.bin` — gensim FastText model (~50–200 MB)
- `bpe_model.model`, `bpe_model.vocab` — SentencePiece BPE model

## What to look for in the output

**FastText** should:
1. Cluster Romance languages together (shared subword units like `▁parlo`, `▁parl-`)
2. Place Italo-Romance dialects near standard Italian
3. Better distinguish dialects from each other than TF-IDF word n-grams
   (morphological variation → different subword n-grams)

**BPE** should:
1. Show similar Romance clustering as TF-IDF word, but with stronger
   cross-language alignment (shared BPE pieces across cognates)
2. Separate Italo-Romance from standard Italian more cleanly than
   character n-grams (BPE respects morpheme boundaries)
3. `top_features.csv` reveals which subword units are most
   characteristic of each dialect — useful for linguistic analysis

## Comparison with other approaches

| Approach | Tokenization | Representation | Strengths |
|---|---|---|---|
| Person 1 (TF-IDF word) | Whitespace | Sparse TF-IDF | Interpretable; fast |
| Person 1 (TF-IDF char) | Char 3-5 grams | Sparse TF-IDF | Good for orthographic variation |
| **Person 5 (FastText)** | **Subword char n-grams** | **Dense mean-pool** | **Handles OOV; captures morphology** |
| **Person 5 (BPE)** | **BPE pieces** | **Sparse TF-IDF** | **Data-driven subwords; links to mBERT** |
| Person 4 (mBERT/XLM-R) | BPE (WordPiece) | Contextual embeddings | State-of-the-art; pretrained |
