# `evaluation/` — central evaluation suite

Method-agnostic evaluation entry point. Every embedding method calls
`run_evaluation` at the end of its pipeline, so the comparison is
apples-to-apples across methods.

## Modules

| File | Purpose |
|---|---|
| `evaluation.py` | `run_evaluation()` — full variety-level eval; also `run_sentence_evaluation()` for sentence-level analysis |
| `parallel_alignment.py` | `run_parallel_alignment()` — aligned-sentence cosine similarity on FLORES+ |
| `compare_methods.py` | `run_cross_method_comparison()` — Spearman ρ, Mantel, Procrustes, CKA between methods |
| `correlate_against_gold.py` | Spearman ρ vs. the lexicostatistical (LDND) gold matrix |
| `_gold_correlation.py` | Helpers for the above |
| `mantel_pvalues.py` | Mantel permutation test (B = 10 000) |
| `_bootstrap_core.py` | Bootstrap CIs on Spearman gold-correlation, shared by per-method `bootstrap.py` |
| `aggregate_bootstrap.py` | Merge per-experiment `bootstrap_results.csv` → final `correlation_<gold>_with_bootstrap.csv` |
| `cli.py` | CLI front-end exposing the four `run_*` entry points |

## What `run_evaluation` writes

For every experiment, under `analysis/<m>/experiments/<exp>/evaluation_results/<source>/<aggregator>/`:

| Artefact | Content |
|---|---|
| `distances.csv` | (N, N) cosine-distance matrix between variety centroids |
| `similarity_matrix.csv` | (N, N) cosine similarity (1 - distance) |
| `similarity_heatmap.png` | Heatmap with rows/columns grouped by family |
| `nearest_neighbors.csv` | Top-k nearest variety per row, ranked |
| `dendrogram.png` | Hierarchical clustering of varieties |
| `projection_mds.png`, `projection_tsne.png`, `projection_umap.png` | 2-D projections |
| `per_variety_profiles.csv` + `per_variety_plots/<code>.png` | Per-variety ranked-distance bar charts |
| `family_stats.csv` | Intra-/inter-family cohesion ratios |
| `clustering_metrics.csv` | Davies-Bouldin, Calinski-Harabasz, ARI, NMI, V-measure, cophenetic r |
| `silhouette_report.txt` | Silhouette over the macro-family partition + per-variety silhouette ranking |
| `gold_correlations.csv` | Spearman ρ vs. every gold matrix found under `gold/*/matrices/` |
| `bootstrap_results.csv` | Bootstrap CIs on the Spearman gold correlations (written by `analysis/<m>/core/bootstrap.py`) |

## Bootstrap + aggregation

Each method has `analysis/<m>/core/bootstrap.py` that resamples FLORES sentences B times to put CIs on the Spearman gold-correlation. After all per-experiment `bootstrap_results.csv` are written, `aggregate_bootstrap.py` merges them into a final table under `gold/_correlations/correlation_<gold>_with_bootstrap.csv`.

See the project root `README.md` (sections 4.1, 4.2) for runnable commands.
