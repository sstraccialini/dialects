# `analysis/` — methods, experiments, comparisons

This directory holds the comparative analysis: 8 active methods, each
with its own self-contained sub-package, plus two cross-cutting
support folders.

## Layout

```
analysis/
├── _shared/                    single source of truth for the variety
│                               registry, paths, sampling defaults, and
│                               the `experiment_dirs` / `write_run_meta`
│                               helpers. Imported by every method's
│                               `core/config.py`.
│
├── _comparisons/               scripts that read multiple methods' outputs
│                               and produce cross-method tables / plots.
│
└── <method>/                   one folder per method (tfidf, word2vec,
    │                           fasttext, sentence_minilm, sentence_labse,
    │                           multilingual_xlmr, multilingual_xlmr_adapted,
    │                           canine).
    │
    ├── core/                   shared logic for ALL experiments of this
    │                           method: config (re-exports `_shared` +
    │                           method-specific knobs), data_loader,
    │                           embedder/vectorize, evaluate (taxonomy
    │                           injection over the central `evaluation/`).
    │
    ├── experiments/            one sub-folder per experiment.
    │   └── <experiment>/       autonomous: run.py + outputs.
    │       ├── run.py
    │       ├── method_outputs/
    │       │   ├── variety_vectors.{npz,csv}
    │       │   ├── run_stats.csv
    │       │   └── run_meta.json
    │       └── evaluation_results/
    │           ├── distances.csv
    │           ├── dendrogram*.png
    │           └── ...
    │
    └── old_experiments/        historical work to keep for reference,
                                will be removed once the new experiments
                                fully replace them.
```

## Conventions

### Experiment naming

Experiments under `analysis/<m>/experiments/<exp>/` should follow one of
two patterns:

| Pattern | When to use | Example |
|---|---|---|
| `<train>_to_<eval>` | when the experiment differs by which corpora are used for training vs evaluation | `wiki_to_flores`, `flores_only`, `wiki_to_oldi_and_flores` |
| `<adapt_strategy>` | when the experiment differs by training/adaptation strategy of the model itself | `mlm_wiki`, `tlm_oldi`, `tsdae_then_mnrl` |

Pick the pattern that makes the contrast between experiments most
explicit.

### Output uniformity

Each experiment **must** write at least:

- `method_outputs/variety_vectors.npz`  — `{matrix: (n_varieties, D),
                                            labels: (n_varieties,)}`
- `method_outputs/run_stats.csv`         — per-variety load/sampling stats
- `method_outputs/run_meta.json`         — produced by
                                           `analysis._shared.run_meta.write_run_meta`
                                           (timestamp, git commit, params)
- `evaluation_results/`                  — produced by
                                           `evaluation.run_evaluation(...)`
                                           via the method's
                                           `core/evaluate.variety_eval`.

Cross-method scripts in `_comparisons/` rely on these files being there
under exactly these names, so don't rename them.

### Where method-specific helpers live

- **Shared between every experiment of the method** → `core/`.
- **Specific to a single experiment** → inside that experiment's folder
  alongside its `run.py` (e.g. an experiment that fine-tunes can have
  its own `trainer.py`, `pipeline.py`, `models/` checkpoint subfolder
  next to `run.py`).
- **Shared across all methods** → `analysis/_shared/`.

The two `old_experiments/<m>/old_experiments/finetuned_flores/` folders
already follow this pattern: every artefact of the legacy fine-tuning
work lives inside the experiment's own folder, not at method level.

## Quick reference

| Module | Imported as | Use |
|---|---|---|
| `_shared.varieties` | `from analysis._shared.varieties import VARIETY_CODES, ...` | variety registry, paths |
| `_shared.run_meta` | `from analysis._shared.run_meta import write_run_meta` | reproducibility |
| `<m>.core.config` | `from analysis.<m>.core.config import ...` | method config (re-exports `_shared` + adds knobs) |
| `<m>.core.data_loader` | `from analysis.<m>.core.data_loader import load_*` | wiki/flores/oldi loaders |
| `<m>.core.evaluate` | `from analysis.<m>.core.evaluate import variety_eval, parallel_eval` | wraps central `evaluation/` with this method's taxonomy |
| `evaluation` | `from evaluation import run_evaluation, run_parallel_alignment` | central evaluation primitives, shared by every method |
