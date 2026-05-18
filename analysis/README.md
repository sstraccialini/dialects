# `analysis/` — methods, experiments, cross-method scripts

Six embedding-method families. Each method is a self-contained sub-package
with shared `core/` logic and one folder per experiment under `experiments/`.

## Layout

```
analysis/
├── _shared/                Single source of truth for the variety registry,
│                             dataset paths, sampling defaults, and the
│                             canonical dataset loaders (load_flores, load_oldi,
│                             load_wiki, load_wiki_plus_oldi_dialect, ...).
│                             Imported by every method's core/config.py.
│
├── _comparisons/           Scripts that read multiple methods' distance
│                             matrices and recompute the cross-method findings
│                             reported in the paper.
│
└── <method>/               One folder per method family:
    │                         tfidf, fasttext, word2vec, canine,
    │                         multilingual_xlmr, labse.
    │
    ├── core/                 Shared logic for every experiment of this method:
    │                           config (re-exports _shared + method-specific
    │                           knobs), embedder/preprocessor, evaluate
    │                           (wraps the central evaluation/ suite), and
    │                           bootstrap (Spearman gold-correlation CIs).
    │
    └── experiments/<exp>/    One sub-folder per cell of the experimental
        ├── run.py              matrix. Each contains a self-contained run.py.
        ├── method_outputs/    Vectors, models, run metadata.
        └── evaluation_results/<source>/<aggregator>/
                                Standard outputs produced by
                                evaluation/run_evaluation: distances.csv,
                                similarity_matrix.csv, nearest_neighbors.csv,
                                silhouette_report.txt, dendrogram.png,
                                projection_*.png, gold_correlations.csv,
                                bootstrap_results.csv, run_meta.json.
```

## Experimental matrix

12 cells total:

| Method | Cells |
|---|---|
| TF-IDF | `tfidf_wikiOLDI_{normalized,native}` (× word + char vectorizer = 4 outputs) |
| FastText | `fasttext_wikiOLDI_{normalized,native}` |
| Word2Vec | `word2vec_wikiOLDI_{normalized,native}` |
| XLM-R | `xlmr_zeroshot_native`, `xlmr_finetuned_wikiOLDI_dialects_native` |
| CANINE | `canine_zeroshot_native`, `canine_finetuned_wikiOLDI_dialects_native` |
| LaBSE | `labse_zeroshot_native`, `labse_finetuned_oldi_dialects_native` |

## Conventions

### Output uniformity

Every experiment writes:

- `method_outputs/variety_vectors.npz` — `{matrix: (n_varieties, D), labels: (n_varieties,)}`
- `method_outputs/run_meta.json` — produced by `analysis._shared.run_meta.write_run_meta`
- `evaluation_results/...` — produced by the central `evaluation.run_evaluation` via the method's `core/evaluate.variety_eval`.

Cross-method scripts in `_comparisons/` rely on these files being there
under exactly these names.

### Where helpers live

- **Shared between every experiment of a method** → `core/`.
- **Specific to a single experiment** → next to that experiment's `run.py`
  (e.g. fine-tuning trainers live under their own experiment folder).
- **Shared across all methods** → `analysis/_shared/`.

## Quick reference

| Module | Imported as | Use |
|---|---|---|
| `_shared.varieties` | `from analysis._shared.varieties import VARIETY_CODES, FLORES_SLUG, ...` | Variety registry + paths |
| `_shared.dataset_loaders` | `from analysis._shared.dataset_loaders import load_flores, load_oldi, load_wiki, load_wiki_plus_oldi_dialect` | Canonical data loaders |
| `_shared.run_meta` | `from analysis._shared.run_meta import write_run_meta` | Reproducibility metadata |
| `<m>.core.config` | `from analysis.<m>.core.config import ...` | Method config (re-exports `_shared` + adds knobs) |
| `<m>.core.evaluate` | `from analysis.<m>.core.evaluate import variety_eval` | Wraps central `evaluation/` with the method's taxonomy |
| `evaluation` | `from evaluation.evaluation import run_evaluation` | Central evaluation primitives, shared by every method |
