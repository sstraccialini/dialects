# How Do Italian Varieties Behave in Embedding Spaces?

Code and data for the final project of *20879 — Language Technology* (Spring
2026, Bocconi University). We compare six families of embedding methods
(TF-IDF, FastText, Word2Vec, XLM-R, CANINE, LaBSE) on six low-resource
Italo-Romance dialects (Friulian, Ligurian, Lombard, Sardinian, Sicilian,
Venetian) plus 11 standard languages, and quantify how well each method
recovers known linguistic structure.

The accompanying paper covers methodology, evaluation, results, and
limitations; this README is the operational counterpart explaining how to
re-run every step that produced the numbers in the paper.

---

## 1. Setup

The repo was developed and tested with **Python 3.9**. A working virtual
environment is shipped under `venv/` (not used by the commands below — every
command below installs into a fresh venv).

```bash
# 1) Create a virtualenv
python3.9 -m venv .venv
source .venv/bin/activate

# 2) Install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3) Install the spaCy Italian model (used by Wiki text cleaning)
python -m spacy download it_core_news_sm
```

GPU is required for the pretrained-encoder experiments (XLM-R, CANINE,
LaBSE). The surface methods (TF-IDF, FastText, Word2Vec) run on CPU.

---

## 2. Repository layout

```
Language-Technology-Project/
├── Dataset/                       Raw and cleaned data sources
│   ├── flores/                      FLORES+ parallel evaluation set
│   │   ├── before_cleaning/         Raw FLORES+ devtest, one .txt per variety
│   │   ├── cleaned/flores.csv         Single CSV (1827 sentences × 17 varieties)
│   │   ├── cleaned_normalized/...     Same, with lowercase-ASCII normalization
│   │   ├── scripts/                   download + normalize + cell-completion notebook
│   │   └── stats.csv                  Per-variety length stats
│   ├── oldi/                        Open Language Data Initiative seed corpus
│   │   ├── before_cleaning/           Raw parquets (ita + 6 dialects) + ita↔dialect pairs
│   │   ├── cleaned/oldi.csv           Single CSV (5167 sentences × ita + 6 dialects)
│   │   ├── cleaned_normalized/...     Same, normalized
│   │   └── scripts/                   pair building, normalization, ita backfill
│   └── wiki/                        Wikipedia per-variety sentences (NOT in git)
│       ├── PIPELINE.md                Dump-to-CSV pipeline documentation
│       ├── scripts/                   Dump extraction, cleaning, sampling
│       └── normalized|not_normalized/{dialects_in_both_OLDI_and_Flores,
│                                       languages, others_dialects}/<code>.csv
│
├── analysis/                      One folder per method family
│   ├── _shared/                     Variety registry + canonical dataset loaders
│   ├── _paper_results/              Scripts that recompute the headline numbers
│   ├── tfidf/                       Surface — sentence-level TF-IDF (word + char)
│   ├── fasttext/                    Surface — gensim FastText (subword skip-gram)
│   ├── word2vec/                    Surface — gensim Word2Vec
│   ├── canine/                      Deep — google/canine-c (tokenizer-free char encoder)
│   ├── multilingual_xlmr/           Deep — xlm-roberta-base
│   └── labse/                       Deep — sentence-transformers/LaBSE
│
├── evaluation/                    Method-agnostic evaluation suite
│   ├── evaluation.py                Variety-level eval entry point used by every method
│   ├── _bootstrap_core.py           Bootstrap CIs on Spearman gold correlations
│   ├── aggregate_bootstrap.py       Aggregate per-experiment bootstraps + Mantel p-values
│   ├── mantel_pvalues.py            Mantel permutation test
│   ├── correlate_against_gold.py    Spearman ρ vs. lexicostatistical / geographic gold
│   ├── _gold_correlation.py         Helpers for the above
│   ├── parallel_alignment.py        Aligned-sentence similarity on FLORES+
│   ├── compare_methods.py           Cross-method distance-matrix comparison
│   ├── cli.py                       CLI front-end
│   └── USAGE.md                     Detailed evaluation-suite documentation
│
├── gold/                          Reference distance matrices
│   ├── lexicostatistical/           LDND on Swadesh-100 (Wichmann 2010)
│   └── _correlations/               Per-method Spearman ρ vs. lexicostat / geographic gold
│
├── llm_translations/              LLM (ChatGPT, Gemini) dialect-translation evaluation
│   ├── oldi-selector.py             Interactive sentence selection
│   ├── data-chatgpt/, data-gemini/  Collected translations
│   ├── compute_metrics.py           BLEU / chrF over the parallel triples
│   ├── bootstrap_significance.py    Within-system bootstrap CIs
│   └── cross_llm_bootstrap.py       Paired bootstrap between systems
│
├── pair_sentences/                Manual annotation tool used to curate a small
│                                    parallel subset of OLDI for the LLM evaluation
│
├── requirements.txt               Pinned Python dependencies
├── README.md                      This file
├── LICENSE                        MIT
└── 20879_Project_Proposal.pdf     Original project proposal
```

For every method, the folder layout is identical:

```
analysis/<method>/
├── core/                Method-specific configuration, embedding, evaluation,
│                          and bootstrap logic.
└── experiments/<exp>/   One folder per "cell" of the experimental matrix.
    ├── run.py             Entry point — train (or load), embed, evaluate.
    ├── method_outputs/    Vectors, models, run metadata.
    └── evaluation_results/<source>/<aggregator>/
                            distances.csv, similarity.csv, nearest_neighbours.csv,
                            silhouette.csv, dendrogram.png, projection_*.png,
                            bootstrap_results.csv, gold_correlations.csv,
                            run_meta.json
```

---

## 3. Data

### 3.1 FLORES+ (`Dataset/flores/`)

The shared parallel evaluation set: 1827 sentences in each of 17 varieties
(6 dialects + 11 standards). After cleaning we ship:

* `cleaned/flores.csv` — native text (case, diacritics, punctuation preserved)
* `cleaned_normalized/flores.csv` — aggressive normalization (lowercase ASCII)

The pre-cleaning per-variety `.txt` files live under
`before_cleaning/{normalized,not_normalized}/`. To re-run the cleaning step:

```bash
python Dataset/flores/scripts/download_flores.py
python Dataset/flores/scripts/normalize.py
# then complete_cleaned_columns.ipynb stitches per-variety files into the CSV
```

### 3.2 OLDI (`Dataset/oldi/`)

Open Language Data Initiative seed corpus — Italian ↔ dialect parallel data.
We keep only the 6 dialects we evaluate (plus Italian) since OLDI is used for
dialect training data, never as a standalone evaluation set. 5167 sentences
in `cleaned/oldi.csv`.

```bash
python Dataset/oldi/scripts/normalize.py
python Dataset/oldi/scripts/build_pairs.py
# translate_missing.py fills in missing Italian sides via deep-translator
# verify_against_hf.py double-checks alignment vs. the HF release
```

### 3.3 Wikipedia (`Dataset/wiki/`)

Per-variety Wikipedia sentences, sub-sampled to ~100k per variety. **Not
shipped in this repo** — the cleaned CSVs alone exceed 50 GB. To regenerate:

```bash
# Pipeline documented in Dataset/wiki/PIPELINE.md
python Dataset/wiki/scripts/create.py            # download + WikiExtractor
python Dataset/wiki/scripts/generation.py        # clean (normalized variant)
python Dataset/wiki/scripts/generation_native.py # clean (native variant)
python Dataset/wiki/scripts/generate_sampled_100k.py
```

After running, the expected layout is

```
Dataset/wiki/{normalized,not_normalized}/{dialects_in_both_OLDI_and_Flores,
                                          languages, others_dialects}/<code>.csv
```

with one CSV per variety, each having a `text` column.

---

## 4. Reproducing the experiments

Every experiment is a self-contained `run.py` invoked as a Python module from
the repository root. The 12 cells of the experimental matrix are:

| Method | Cell | Command |
|---|---|---|
| TF-IDF | `tfidf_wikiOLDI_normalized` | `python -m analysis.tfidf.experiments.tfidf_wikiOLDI_normalized.run` |
| TF-IDF | `tfidf_wikiOLDI_native`     | `python -m analysis.tfidf.experiments.tfidf_wikiOLDI_native.run` |
| FastText | `fasttext_wikiOLDI_normalized` | `python -m analysis.fasttext.experiments.fasttext_wikiOLDI_normalized.run` |
| FastText | `fasttext_wikiOLDI_native`     | `python -m analysis.fasttext.experiments.fasttext_wikiOLDI_native.run` |
| Word2Vec | `word2vec_wikiOLDI_normalized` | `python -m analysis.word2vec.experiments.word2vec_wikiOLDI_normalized.run` |
| Word2Vec | `word2vec_wikiOLDI_native`     | `python -m analysis.word2vec.experiments.word2vec_wikiOLDI_native.run` |
| XLM-R    | `xlmr_zeroshot_native`                     | `python -m analysis.multilingual_xlmr.experiments.xlmr_zeroshot_native.run` |
| XLM-R    | `xlmr_finetuned_wikiOLDI_dialects_native`  | `python -m analysis.multilingual_xlmr.experiments.xlmr_finetuned_wikiOLDI_dialects_native.run` |
| CANINE   | `canine_zeroshot_native`                     | `python -m analysis.canine.experiments.canine_zeroshot_native.run` |
| CANINE   | `canine_finetuned_wikiOLDI_dialects_native`  | `python -m analysis.canine.experiments.canine_finetuned_wikiOLDI_dialects_native.run` |
| LaBSE    | `labse_zeroshot_native`                      | `python -m analysis.labse.experiments.labse_zeroshot_native.run` |
| LaBSE    | `labse_finetuned_oldi_dialects_native`       | `python -m analysis.labse.experiments.labse_finetuned_oldi_dialects_native.run` |

All deep-model fine-tuning uses a single GPU. Each `run.py`:

1. loads its dataset (Wiki + OLDI for surface methods, Wiki + OLDI dialect
   for encoder fine-tuning, FLORES for evaluation),
2. trains or loads the model,
3. writes per-sentence and per-variety embeddings,
4. computes the cosine-distance matrix on the 17 FLORES centroids,
5. calls `evaluation.run_evaluation`, which produces the standard outputs
   under `analysis/<method>/experiments/<exp>/evaluation_results/`.

### 4.1 Bootstrap confidence intervals

Each method has a `core/bootstrap.py` module that resamples FLORES sentences
B times to put confidence intervals on the gold-correlation Spearman ρ:

```bash
python -m analysis.canine.core.bootstrap \
    --experiment canine_finetuned_wikiOLDI_dialects_native --n-boot 1000

python -m analysis.tfidf.core.bootstrap \
    --experiment tfidf_wikiOLDI_normalized --variant char --n-boot 1000
```

The aggregate table is produced by:

```bash
python -m evaluation.aggregate_bootstrap \
    --analysis-root analysis/ \
    --out gold/_correlations/correlation_<gold>_with_bootstrap.csv
```

### 4.2 Mantel permutation tests

```bash
python -m evaluation.mantel_pvalues \
    --analysis-root analysis/ \
    --gold-root gold/ \
    --n-permutations 10000
```

---

## 5. Evaluation against gold

```bash
# Spearman ρ vs. lexicostatistical (LDND) and geographic (Haversine) gold
python -m evaluation.correlate_against_gold \
    --analysis-root analysis/ \
    --gold-root gold/
```

Results are written to `gold/_correlations/correlation_<gold>.csv`.

### 5.1 Lexicostatistical gold (`gold/lexicostatistical/`)

```bash
python -m gold.lexicostatistical.fetch_swadesh      # pull Swadesh-100 from Wiktionary
python -m gold.lexicostatistical.expand_gendered    # split gendered forms
python -m gold.lexicostatistical.build_ldnd         # build the LDND matrix
```

See `gold/lexicostatistical/README.md` for the underlying methodology
(ASJP transcription + chance-normalised Levenshtein, Wichmann et al. 2009).

---

## 6. Reproducing the paper findings

The "What embeddings recover" subsection cites cross-method statistics that
are not stored anywhere as a single artifact. They are recomputed on demand
from the saved per-method `distances.csv` files by four scripts under
`analysis/_paper_results/scripts/`:

```bash
python -m analysis._paper_results.scripts.01_trustworthy_consensus
python -m analysis._paper_results.scripts.02_bidirectional_gold
python -m analysis._paper_results.scripts.03_per_method_recovery
python -m analysis._paper_results.scripts.04_comprehensive_analysis
```

Each script prints to stdout the agreement count, the diverging methods, and
the underlying numbers. See `analysis/_paper_results/README.md` for what each
script answers.

---

## 7. LLM-translation evaluation (`llm_translations/`)

Independent of the embedding-space pipeline: we collected ChatGPT and Gemini
translations on a curated parallel subset and measured BLEU / chrF.

```bash
# Compute metrics over already-collected translations
python llm_translations/compute_metrics.py

# Bootstrap CIs (within system)
python llm_translations/bootstrap_significance.py

# Paired bootstrap (system A vs. system B)
python llm_translations/cross_llm_bootstrap.py \
    --pred_dir_a data-chatgpt --label_a chatgpt \
    --pred_dir_b data-gemini  --label_b gemini \
    --out cross_llm_chatgpt_vs_gemini.csv
```

The interactive sentence-selection tool used to build the evaluation subset
lives at `llm_translations/oldi-selector.py`.

---

## 8. License

MIT — see [LICENSE](LICENSE).
