# `_comparisons/` — cross-method analyses

Scripts here read the per-method outputs (under `analysis/<m>/experiments/<exp>/method_outputs/`
and `evaluation_results/`) and produce **comparisons across methods** — for
example:

- a leaderboard CSV with `silhouette_family`, `silhouette_romance_vs_rest`
  for every (method, experiment) pair
- a joint dendrogram or projection that overlays the variety vectors of
  several methods
- a heatmap of pairwise correlation between method distance matrices

By convention every script:

1. Globs `analysis/*/experiments/*/method_outputs/variety_vectors.npz` to
   find available results
2. Loads the matching `run_meta.json` (when present) for hyperparameter
   provenance
3. Writes its outputs under `analysis/_comparisons/results/<script_name>/`
   — never inside individual methods' folders.

Cross-method evaluation primitives (e.g. `run_cross_method_comparison`) live
in the central `evaluation/` package; this folder is the place where you
**call** them with the right inputs.
