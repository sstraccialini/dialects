# Evaluation Suite — Usage Guide

This guide covers every function, parameter, and output file in the `evaluation/`
package.  The suite is organized into four modules:

| Module | Entry point | Purpose |
|---|---|---|
| `evaluation.py` | `run_evaluation()` | Variety-level evaluation for any embedding method |
| `evaluation.py` | `run_sentence_evaluation()` | Sentence-level analysis when full embeddings are available |
| `parallel_alignment.py` | `run_parallel_alignment()` | Aligned-sentence similarity for parallel corpora (FLORES+) |
| `compare_methods.py` | `run_cross_method_comparison()` | Compare distance structures across methods |
| `cli.py` | CLI sub-commands | Run any of the above from the terminal |

All modules are also importable as `from evaluation import run_evaluation`, etc.

---

## Table of Contents

1. [Quick-start commands](#1-quick-start-commands)
2. [CLI reference](#2-cli-reference)
   - [run](#21-run--variety-level-evaluation)
   - [sentence](#22-sentence--sentence-level-evaluation)
   - [compare](#23-compare--cross-method-comparison)
3. [Python API](#3-python-api)
   - [run_evaluation](#31-run_evaluation)
   - [run_sentence_evaluation](#32-run_sentence_evaluation)
   - [run_parallel_alignment](#33-run_parallel_alignment)
   - [run_cross_method_comparison](#34-run_cross_method_comparison)
4. [Output files reference](#4-output-files-reference)
5. [Metrics glossary](#5-metrics-glossary)
6. [Taxonomy reference](#6-taxonomy-reference)
7. [Recommended workflows](#7-recommended-workflows)
8. [Dependencies and optional packages](#8-dependencies-and-optional-packages)

---

## 1. Quick-start commands

```bash
# Evaluate Word2Vec variety vectors (simplest case)
python evaluation/cli.py run \
  --vectors  analysis/word2vec/flores/method_outputs/variety_vectors.npz \
  --method   "Word2Vec (FLORES+)" \
  --out-dir  analysis/word2vec/flores/evaluation_results

# Evaluate XLM-R — complete linkage, more neighbors
python evaluation/cli.py run \
  --vectors   analysis/multilingual_xlmr/flores/method_outputs/variety_vectors.npz \
  --method    "XLM-R (FLORES+)" \
  --out-dir   analysis/multilingual_xlmr/flores/evaluation_results \
  --linkage   complete \
  --nearest-k 5

# Sentence-level analysis from XLM-R sentence embeddings
python evaluation/cli.py sentence \
  --vectors  analysis/multilingual_xlmr/flores/method_outputs/sentence_vectors.npz \
  --method   "XLM-R sentences (FLORES+)" \
  --out-dir  analysis/multilingual_xlmr/flores/evaluation_results/sentence_level

# Compare Word2Vec, FastText, XLM-R side by side
python evaluation/cli.py compare \
  --method-dirs \
    "Word2Vec:analysis/word2vec/flores/evaluation_results,\
FastText:analysis/fasttext/flores/evaluation_results/fasttext,\
XLM-R:analysis/multilingual_xlmr/flores/evaluation_results" \
  --vector-paths \
    "Word2Vec:analysis/word2vec/flores/method_outputs/variety_vectors.npz,\
XLM-R:analysis/multilingual_xlmr/flores/method_outputs/variety_vectors.npz" \
  --out-dir analysis/comparison/flores
```

---

## 2. CLI reference

Run from the **project root**.  The `run` sub-command can be omitted: if
the first argument starts with `--`, the CLI inserts `run` automatically.

```
python evaluation/cli.py [run | sentence | compare] [options]
```

### 2.1 `run` — variety-level evaluation

Loads a saved variety-vector file and runs the full evaluation suite.

```
python evaluation/cli.py run
    --vectors   PATH          (required)
    --out-dir   PATH          (required)
    --method    STRING        (default: parent directory name)
    --linkage   CHOICE        (default: average)
    --nearest-k INT           (default: 3)
```

#### Parameters

| Flag | Type | Default | Description |
|---|---|---|---|
| `--vectors` | path | — | Variety vector file. Accepted formats: `.npz` (keys `matrix`, `labels`) or `.csv` / `.tsv` (index = variety code, columns = dimensions). |
| `--out-dir` | path | — | Output directory. Created if it does not exist. |
| `--method` | string | parent dir name | Label shown in plot titles (e.g. `"TF-IDF char"`, `"XLM-R (FLORES+)"`). Has no effect on computed values. |
| `--linkage` | choice | `average` | Linkage criterion for hierarchical clustering. See table below. |
| `--nearest-k` | int ≥ 1 | `3` | Number of nearest neighbours written per variety in `nearest_neighbors.csv`. |

#### `--linkage` options

| Value | Description | When to use |
|---|---|---|
| `average` | Distance between two clusters = mean of all pairwise distances (UPGMA). **Default.** | General use; coherent with cosine distance; avoids chaining |
| `complete` | Distance = maximum pairwise distance between clusters | Compact, spherical clusters; highlights outliers |
| `single` | Distance = minimum pairwise distance (nearest-neighbour chaining) | Detects elongated or chain-like clusters; prone to chaining |
| `ward` | Minimises within-cluster variance (uses squared Euclidean internally) | Suited for Euclidean space; less appropriate with cosine distance |

> **Note:** `average` is the project default because it is mathematically coherent with
> cosine distance.  Use `complete` when you want to stress-test whether clusters remain
> tight under the worst-case pairwise distance.

---

### 2.2 `sentence` — sentence-level evaluation

Loads a `sentence_vectors.npz` file whose `matrix` key is shape `(M, D)`
and `labels` key is length-`M` array of variety codes, then runs sentence-level
analysis.

```
python evaluation/cli.py sentence
    --vectors   PATH          (required)
    --out-dir   PATH          (required)
    --method    STRING        (default: "")
    --n-sample  INT           (default: 5000)
```

#### Parameters

| Flag | Type | Default | Description |
|---|---|---|---|
| `--vectors` | path | — | `.npz` file with keys `matrix` (M, D) and `labels` (M,). |
| `--out-dir` | path | — | Output directory. |
| `--method` | string | `""` | Label for plot titles. |
| `--n-sample` | int | `5000` | Sentences to subsample for silhouette and projections. Large `M` makes the full distance matrix intractable. Set `0` to use all sentences (slow for M > 10 000). |

> `--n-sample` affects only the silhouette score and the UMAP/t-SNE plots.
> The within/between distance table is always computed on all sentences.

---

### 2.3 `compare` — cross-method comparison

Loads `distances.csv` from multiple evaluation results directories and
computes pairwise agreement metrics across methods.

```
python evaluation/cli.py compare
    --method-dirs   "Name:path[,Name:path,...]"    (required)
    --out-dir       PATH                           (required)
    --vector-paths  "Name:path[,Name:path,...]"    (optional)
    --mantel-perms  INT                            (default: 999)
```

#### Parameters

| Flag | Type | Default | Description |
|---|---|---|---|
| `--method-dirs` | string | — | Comma-separated `Name:path` pairs, one per method. Each `path` must point to a directory containing `distances.csv`. |
| `--out-dir` | path | — | Output directory. |
| `--vector-paths` | string | `""` | Optional comma-separated `Name:path` pairs pointing to `variety_vectors.npz` or `.csv` files. Enables **Procrustes disparity** and **Linear CKA**. |
| `--mantel-perms` | int | `999` | Permutations for the Mantel test. Set `0` to skip the test (faster). More permutations = more precise p-value. |

#### Example

```bash
python evaluation/cli.py compare \
  --method-dirs \
    "TF-IDF (char):analysis/tfidf/flores/evaluation_results/char,\
TF-IDF (word):analysis/tfidf/flores/evaluation_results/word,\
Word2Vec:analysis/word2vec/flores/evaluation_results,\
FastText:analysis/fasttext/flores/evaluation_results/fasttext,\
FastText BPE:analysis/fasttext/flores/evaluation_results/bpe,\
XLM-R:analysis/multilingual_xlmr/flores/evaluation_results,\
Sentence-MiniLM:analysis/sentence_baseline/flores/evaluation_results/sentence" \
  --vector-paths \
    "Word2Vec:analysis/word2vec/flores/method_outputs/variety_vectors.npz,\
FastText:analysis/fasttext/flores/method_outputs/variety_vectors.npz,\
XLM-R:analysis/multilingual_xlmr/flores/method_outputs/variety_vectors.npz,\
Sentence-MiniLM:analysis/sentence_baseline/flores/method_outputs/variety_vectors.npz" \
  --out-dir analysis/comparison/flores \
  --mantel-perms 9999
```

---

## 3. Python API

All functions are importable directly:

```python
from evaluation import (
    run_evaluation,
    run_sentence_evaluation,
    run_parallel_alignment,
    run_cross_method_comparison,
)
```

Or from the sub-modules:

```python
from evaluation.evaluation import run_evaluation, run_sentence_evaluation
from evaluation.parallel_alignment import run_parallel_alignment
from evaluation.compare_methods import run_cross_method_comparison
```

---

### 3.1 `run_evaluation`

The main entry point.  Every method's `run_*.py` calls this at the end of
its pipeline.

```python
results = run_evaluation(
    variety_vectors,          # np.ndarray (N, D)  — one row per language/dialect
    variety_codes,            # list[str] length N — stable identifiers
    out_dir,                  # str | Path

    # metadata
    method_label   = "",

    # taxonomy (all optional; plots are greyscale without them)
    family_groups        = None,
    family_colors        = None,
    family_display_names = None,
    display_names        = None,
    romance_families     = None,

    # algorithmic
    linkage_method   = "average",
    tsne_perplexity  = 4.0,
    nearest_k        = 3,
    random_state     = 42,
    normalise        = True,
)
```

#### Parameter reference

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `variety_vectors` | `np.ndarray (N, D)` | — | Per-variety embedding matrix. Rows are varieties; columns are embedding dimensions. |
| `variety_codes` | `list[str]` | — | Stable identifier for each row (e.g. `"veneto"`, `"italiano"`). Must align with `variety_vectors`. |
| `out_dir` | `str \| Path` | — | All output files are written here. Created automatically. |
| `method_label` | `str` | `""` | Human-readable name appended to every plot title. No effect on values. |
| `family_groups` | `dict[str, str] \| None` | `None` | Maps each variety code to its language family (e.g. `"veneto" → "italo_romance"`). Required for coloured plots, silhouette scores, family stats, and clustering metrics. |
| `family_colors` | `dict[str, str] \| None` | `None` | Maps each family name to a hex colour string. If omitted, all points are grey. |
| `family_display_names` | `dict[str, str] \| None` | `None` | Maps each family name to a human-readable legend entry. |
| `display_names` | `dict[str, str] \| None` | `None` | Maps each variety code to a pretty label used on plot axes. Falls back to the code itself. |
| `romance_families` | `set[str] \| None` | `None` | Set of family names counted as "Romance". When provided, a second silhouette score (Romance vs. rest) is computed. Recommended: `{"italo_romance", "italian", "romance"}`. |
| `linkage_method` | `str` | `"average"` | Hierarchical clustering linkage. Options: `"average"`, `"complete"`, `"single"`, `"ward"`. See §2.1 table. |
| `tsne_perplexity` | `float` | `4.0` | t-SNE perplexity. For N ≈ 16 varieties, values in [3, 6] work well. The function clamps it to `max(2.0, min(value, (N−1)/3))` automatically. Higher values = more global structure; lower = tighter local clusters. |
| `nearest_k` | `int` | `3` | How many nearest neighbours to list per variety in `nearest_neighbors.csv`. |
| `random_state` | `int` | `42` | Seed for MDS, t-SNE, UMAP, and K-Means. Fix to reproduce identical plots across runs. |
| `normalise` | `bool` | `True` | L2-normalise rows before computing cosine distances. Should almost always be `True`; set to `False` only if vectors are already normalised. |

#### Return value

A `dict` with the following keys (all values are strings for file paths,
or `float | None` for scalar metrics):

```
distances_path               → distances.csv
similarity_path              → similarity_matrix.csv
nearest_neighbors_path       → nearest_neighbors.csv
similarity_heatmap_path      → similarity_heatmap.png
dendrogram_path              → dendrogram.png
projection_mds_path          → projection_mds.png
projection_tsne_path         → projection_tsne.png
projection_umap_path         → projection_umap.png  (None if umap-learn absent)
per_variety_profiles_path    → per_variety_profiles.csv
per_variety_plots_dir        → per_variety_plots/
family_stats_path            → family_stats.csv      (None if no family_groups)
clustering_metrics_path      → clustering_metrics.csv (None if no family_groups)
silhouette_path              → silhouette_report.txt
silhouette_family            → float | None
silhouette_romance_vs_rest   → float | None
n_varieties                  → int
method_label                 → str
```

---

### 3.2 `run_sentence_evaluation`

Analyses the full sentence-embedding matrix rather than variety centroids.

```python
results = run_sentence_evaluation(
    sentence_vectors,        # np.ndarray (M, D) — one row per sentence
    sentence_labels,         # list[str] length M — variety code per sentence

    out_dir,

    method_label         = "",
    family_groups        = None,
    family_colors        = None,
    family_display_names = None,
    display_names        = None,
    romance_families     = None,
    normalise            = True,
    random_state         = 42,
    n_sample             = 5000,
    tsne_perplexity      = 30.0,
    umap_n_neighbors     = 15,
)
```

#### Parameter reference

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `sentence_vectors` | `np.ndarray (M, D)` | — | One sentence embedding per row. `M` is the total number of sentences across all varieties. |
| `sentence_labels` | `list[str]` | — | Variety code for each row (length M). |
| `n_sample` | `int \| None` | `5000` | Subsample size for silhouette computation and projections. The full (M × M) pairwise distance matrix is intractable for M > ~20 000. Set to `None` or `0` to use all sentences. |
| `tsne_perplexity` | `float` | `30.0` | t-SNE perplexity for the sentence projection. Larger than in `run_evaluation` because M >> N. Typical range: 10–50. |
| `umap_n_neighbors` | `int` | `15` | UMAP `n_neighbors` for the sentence projection. Controls the balance between local and global structure. Typical range: 5–50. |

All other parameters are the same as in `run_evaluation`.

#### Return value

```
within_between_path      → sentence_within_between.csv
silhouette_sentence      → float | None
sentence_report_path     → sentence_silhouette_report.txt
sentence_tsne_path       → sentence_projection_tsne.png
sentence_umap_path       → sentence_projection_umap.png  (None if umap-learn absent)
```

---

### 3.3 `run_parallel_alignment`

Computes sentence-pair cosine similarity on an **aligned** corpus (same
sentence `i` in every variety, as in FLORES+).  Unlike centroid-based
evaluation, this does not aggregate into variety representations first —
it measures directly how close each translation is to the others.

```python
results = run_parallel_alignment(
    sentence_vectors,        # dict[str, np.ndarray (N, D)]
                             # ALL arrays must have the same N (aligned sentences)
    out_dir,

    method_label         = "",
    family_groups        = None,
    family_colors        = None,
    family_display_names = None,
    display_names        = None,
    normalise            = True,
)
```

#### Parameter reference

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `sentence_vectors` | `dict[str, ndarray]` | — | Maps each variety code to its `(N, D)` sentence matrix. Every matrix must have the same `N` (number of aligned sentences). |
| `normalise` | `bool` | `True` | L2-normalise sentence vectors before computing dot products. |

All other parameters are the same as `run_evaluation`.

#### Building the input dict

```python
import numpy as np

# Option A: load per-variety sentence .npy files
sentence_vecs = {
    "veneto":   np.load("analysis/multilingual_xlmr/flores/method_outputs/veneto_sent.npy"),
    "italiano": np.load("analysis/multilingual_xlmr/flores/method_outputs/italiano_sent.npy"),
    "inglese":  np.load("analysis/multilingual_xlmr/flores/method_outputs/inglese_sent.npy"),
}

# Option B: slice from a consolidated sentence_vectors.npz
data   = np.load("analysis/multilingual_xlmr/flores/method_outputs/sentence_vectors.npz",
                 allow_pickle=True)
matrix = data["matrix"]   # (M_total, D)
labels = data["labels"]   # (M_total,)

N_per_variety = 2009      # FLORES+ has 2009 aligned sentences per variety
sentence_vecs = {}
for code in sorted(set(labels)):
    idx = [i for i, l in enumerate(labels) if l == code]
    sentence_vecs[code] = matrix[idx]   # should be (2009, D) if aligned
```

#### Return value

```
alignment_csv           → parallel_alignment.csv
alignment_heatmap       → parallel_alignment_heatmap.png
alignment_pairs_csv     → parallel_alignment_pairs.csv
alignment_dialects_csv  → parallel_alignment_dialects.csv
alignment_report        → parallel_alignment_report.txt
n_sentences             → int
n_varieties             → int
```

---

### 3.4 `run_cross_method_comparison`

```python
results = run_cross_method_comparison(
    method_eval_dirs,        # dict[str, str | Path]  name → eval_results_dir
    out_dir,

    variety_vector_paths = None,   # dict[str, str | Path] name → vector_file
    mantel_permutations  = 999,
    random_state         = 42,
)
```

#### Parameter reference

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `method_eval_dirs` | `dict[str, path]` | — | Maps each method name to its evaluation results directory. Each directory must contain `distances.csv` (produced by `run_evaluation`). |
| `variety_vector_paths` | `dict[str, path] \| None` | `None` | Maps method names to raw variety-vector files (`.npz` or `.csv`). **Required** for Procrustes disparity and Linear CKA. Methods not listed here are skipped for those two metrics. |
| `mantel_permutations` | `int` | `999` | Number of random permutations used to estimate the Mantel p-value. Higher = more precise. Set `0` to skip the test entirely. |
| `random_state` | `int` | `42` | RNG seed used by the Mantel test permutations. |

#### Return value

```
rank_correlation_csv        → method_rank_correlation.csv
rank_correlation_heatmap    → method_rank_correlation_heatmap.png
mantel_csv                  → method_mantel.csv
procrustes_csv              → method_procrustes.csv        (None if no vectors given)
procrustes_heatmap          → method_procrustes_heatmap.png
cka_csv                     → method_cka.csv               (None if no vectors given)
cka_heatmap                 → method_cka_heatmap.png
report                      → comparison_report.txt
methods                     → list[str]
n_methods                   → int
```

---

## 4. Output files reference

### From `run_evaluation`

| File | Format | Description |
|---|---|---|
| `distances.csv` | N × N CSV | Symmetric pairwise **cosine distance** matrix. Values in [0, 2]; 0 = identical, 2 = opposite. Diagonal is 0. |
| `similarity_matrix.csv` | N × N CSV | Pairwise **cosine similarity** = 1 − distance. Values in [−1, 1]; 1 = identical. More intuitive than distance for manual inspection. |
| `nearest_neighbors.csv` | Tabular CSV | For each variety: its `nearest_k` closest neighbours with their distances. Columns: `code`, `nn_1`, `dist_1`, `nn_2`, `dist_2`, … |
| `similarity_heatmap.png` | PNG | Annotated colour grid of the similarity matrix. Rows and columns are sorted by language family (Italo-Romance first). Colour scale: red = dissimilar, yellow = neutral, green = similar. |
| `dendrogram.png` | PNG | Hierarchical clustering tree computed from the cosine distance matrix. Leaf labels are coloured by family. Y-axis = cosine distance at merge. |
| `projection_mds.png` | PNG | 2D MDS (multidimensional scaling) projection. Preserves pairwise distances as faithfully as possible in 2D. |
| `projection_tsne.png` | PNG | 2D t-SNE projection. Better at revealing local cluster structure; distances between clusters are not interpretable. |
| `projection_umap.png` | PNG | 2D UMAP projection (only if `umap-learn` is installed). Balances local and global structure. Often cleaner than t-SNE for small N. |
| `per_variety_profiles.csv` | Long CSV | All `N × (N−1)` ordered variety pairs: `source`, `target`, `distance`, `similarity`, `target_family`, `target_display`. Sorted by `(source, distance)`. |
| `per_variety_plots/<code>.png` | PNG × N | One bar chart per variety. Bars show cosine distance to each other variety, sorted ascending. Bars are coloured by the target's family. Dashed line = mean distance. |
| `family_stats.csv` | CSV | Per-family: `n_members`, `mean_intra_dist`, `std_intra_dist`, `mean_inter_dist`, `std_inter_dist`, `cohesion_ratio` (= inter / intra). Sorted by cohesion ratio descending. |
| `clustering_metrics.csv` | Two-section CSV | **Section 1 — global metrics** (one row each): silhouette global, Davies-Bouldin, Calinski-Harabasz, cophenetic correlation, ARI / NMI / V-measure for hierarchical cut and K-Means. **Section 2 — per-variety silhouette** (one row per variety). |
| `silhouette_report.txt` | Plain text | Global silhouette (family) and silhouette (Romance vs. rest), followed by per-variety silhouette scores sorted from highest to lowest. |

### From `run_sentence_evaluation`

| File | Description |
|---|---|
| `sentence_within_between.csv` | Per variety: `n_sentences`, `mean_within_dist` (avg. distance among sentences of that variety), `mean_between_dist` (avg. distance to all other varieties), `separation_ratio` (between / within; higher = more distinct). |
| `sentence_silhouette_report.txt` | Sentence-level silhouette score + the within/between table. |
| `sentence_projection_tsne.png` | t-SNE of subsampled sentence embeddings coloured by variety. |
| `sentence_projection_umap.png` | UMAP of subsampled sentence embeddings (if available). |

### From `run_parallel_alignment`

| File | Description |
|---|---|
| `parallel_alignment.csv` | N × N matrix of **mean sentence-pair cosine similarity**. Entry [i, j] = mean over all aligned sentences of `cos_sim(sent_i_lang_A, sent_i_lang_B)`. |
| `parallel_alignment_heatmap.png` | Heatmap of the above matrix, sorted by family. |
| `parallel_alignment_pairs.csv` | All N(N−1)/2 variety pairs ranked by mean similarity. Columns: `variety_A`, `variety_B`, `mean_similarity`, `family_A`, `family_B`. |
| `parallel_alignment_dialects.csv` | Subset: each Italian dialect vs. each non-dialect reference variety. Useful for the "dialect vs. the world" analysis. |
| `parallel_alignment_report.txt` | Top/bottom ranked pairs, per-variety mean alignment to all others. |

### From `run_cross_method_comparison`

| File | Description |
|---|---|
| `method_rank_correlation.csv` | M × M Spearman ρ matrix. Entry [i, j] = rank correlation between method i and method j's pairwise distances on shared varieties. |
| `method_rank_correlation_heatmap.png` | Heatmap of the above. |
| `method_mantel.csv` | One row per method pair: `mantel_r` (Pearson correlation of distance upper triangles), `p_value` (permutation test), `n_varieties`, `significant_005`. |
| `method_procrustes.csv` | M × M matrix of Procrustes disparity (0 = same geometry, 1 = unrelated). Only computed for methods that have vector files. |
| `method_procrustes_heatmap.png` | Heatmap of the above (inverted colour: green = low disparity = similar). |
| `method_cka.csv` | M × M matrix of Linear CKA (0 = unrelated, 1 = identical). |
| `method_cka_heatmap.png` | Heatmap of the above. |
| `comparison_report.txt` | All four matrices printed in plain text for quick reading. |

---

## 5. Metrics glossary

### Cosine distance

```
d(x, y) = 1 − cos(x, y) = 1 − (x · y) / (‖x‖ ‖y‖)
```

Range [0, 2]; 0 = same direction, 1 = orthogonal, 2 = opposite.
With L2-normalised vectors this reduces to `d = 1 − x·y`.
All clustering and projection methods in this suite use cosine distance.

### Silhouette score

For each sample `i`:

```
s(i) = (b(i) − a(i)) / max(a(i), b(i))
```

Where `a(i)` = mean distance to other samples in the same cluster,
`b(i)` = mean distance to samples in the nearest other cluster.

| Range | Interpretation |
|---|---|
| > 0.5 | Excellent — varieties cluster cleanly by family |
| 0.2 – 0.5 | Good — families distinguishable |
| 0.0 – 0.2 | Weak — overlapping families |
| < 0 | Varieties are closer to another family than their own |

`run_evaluation` computes two global silhouette scores:
- **family** — uses the full family taxonomy as labels
- **romance vs. rest** — binary: Romance (Italo-Romance + Italian + Other Romance) vs. all others

It also computes per-variety scores (in `clustering_metrics.csv` and `silhouette_report.txt`).

### Davies-Bouldin index

Average ratio of within-cluster scatter to between-cluster distance.
**Lower is better.**  Range [0, ∞).  Computed on raw (L2-normalised) vectors.
Does not require setting a threshold, making it useful for comparing methods directly.

### Calinski-Harabasz score

Ratio of between-cluster dispersion to within-cluster dispersion.
**Higher is better.**  Computed on raw vectors.
Tends to favour compact, well-separated clusters.

### Cophenetic correlation

Pearson correlation between the original pairwise distances and the
distances implied by the dendrogram (the height at which two samples
first merge).  **Higher is better** (max 1.0).
Measures how faithfully the hierarchical clustering tree represents
the actual distance structure.

### Adjusted Rand Index (ARI)

Measures agreement between predicted cluster labels and true family labels,
corrected for chance.  Range [−1, 1]; 1 = perfect agreement, 0 = chance.
Computed for two clusterers: hierarchical (dendrogram cut at K families)
and K-Means.

### Normalized Mutual Information (NMI)

Information-theoretic measure of agreement between predicted and true labels.
Range [0, 1]; 1 = perfect, 0 = independent.  Unlike ARI, NMI is symmetric.

### V-measure

Harmonic mean of homogeneity (each cluster contains only one class) and
completeness (all samples of a class are in one cluster).  Range [0, 1].

### Spearman ρ (rank correlation)

Correlation between the **ranks** of pairwise distances from two methods.
Range [−1, 1]; 1 = identical ordering.  Robust to outliers.
Used in `compare_methods.py` to ask: "Do two methods agree on which variety
pairs are most/least similar?"

### Mantel test

Permutation test for the Pearson correlation between two distance matrices.
Quantifies whether the correlation in `rank_correlation.csv` is statistically
significant.  `p < 0.05` means the two methods' distance orderings agree
beyond what random permutation can explain.

### Procrustes disparity

After finding the optimal orthogonal rotation `R` that aligns matrix A onto
matrix B (`min ‖AR − B‖_F`), the **normalised Frobenius residual** is returned:

```
disparity = ‖AR − B‖_F / ‖B‖_F
```

Range [0, 1] approximately; 0 = perfect geometric alignment.
Answers: "After removing rotation differences, do the two methods place varieties
in the same relative positions?"  Requires raw vector files.

### Linear CKA

Centered Kernel Alignment on linear kernels `K = XX^T`:

```
CKA(A, B) = HSIC(K_A, K_B) / sqrt(HSIC(K_A, K_A) · HSIC(K_B, K_B))
```

Where `HSIC` = centred Frobenius inner product.  Range [0, 1].
Invariant to orthogonal transformations **and** isotropic scaling.
Stronger invariance than Procrustes — two spaces can differ by any
orthogonal rotation or scale and still score CKA = 1 if their relative
point geometry is the same.

### Parallel alignment score

Mean cosine similarity between aligned sentence pairs:

```
align(A, B) = (1/N) Σ_i cos(sent_A_i, sent_B_i)
```

Range [−1, 1] (typically [0.5, 1.0] for related languages with a good model).
Directly measures cross-lingual alignment in the embedding space without
aggregating to variety centroids.  High values mean translations land near
each other; low values mean the model treats them as unrelated.

### Within / between variety distance

From `run_sentence_evaluation`:

- **within** = mean cosine distance among all sentence pairs belonging to the
  same variety
- **between** = mean cosine distance from each sentence to all sentences of all
  other varieties
- **separation ratio** = between / within; ratio > 1 means the variety is more
  separated from others than internally spread

---

## 6. Taxonomy reference

The standard taxonomy used by the CLI and recommended for the Python API:

| Code | Display | Family | Color |
|---|---|---|---|
| `veneto` | Veneto | `italo_romance` | #d62728 (red) |
| `siciliano` | Siciliano | `italo_romance` | #d62728 |
| `lombardo` | Lombardo | `italo_romance` | #d62728 |
| `sardo` | Sardo | `italo_romance` | #d62728 |
| `ligure` | Ligure | `italo_romance` | #d62728 |
| `friulano` | Friulano | `italo_romance` | #d62728 |
| `ladino` | Ladino | `italo_romance` | #d62728 |
| `napolitano` | Napolitano | `italo_romance` | #d62728 |
| `italiano` | Italiano | `italian` | #ff7f0e (orange) |
| `spagnolo` | Spagnolo | `romance` | #2ca02c (green) |
| `francese` | Francese | `romance` | #2ca02c |
| `catalano` | Catalano | `romance` | #2ca02c |
| `tedesco` | Tedesco | `germanic` | #1f77b4 (blue) |
| `inglese` | Inglese | `english` | #17becf (cyan) |
| `greco` | Greco | `greek` | #9467bd (purple) |
| `arabo` | Arabo | `semitic` | #8c564b (brown) |
| `sloveno` | Sloveno | `slavic` | #e377c2 (pink) |

`romance_families = {"italo_romance", "italian", "romance"}` covers all
Italo-Romance dialects, Standard Italian, and the other Romance languages
for the binary silhouette analysis.

---

## 7. Recommended workflows

### A. Evaluate a single method end-to-end

```bash
# From the project root, after the method's run_*.py has saved variety_vectors.npz
python evaluation/cli.py run \
  --vectors  analysis/<method>/flores/method_outputs/variety_vectors.npz \
  --method   "<Method Name> (FLORES+)" \
  --out-dir  analysis/<method>/flores/evaluation_results
```

Produces 13 output files in `evaluation_results/`.

---

### B. Sentence-level + centroid comparison (same method)

```bash
# 1. Centroid-level
python evaluation/cli.py run \
  --vectors analysis/multilingual_xlmr/flores/method_outputs/variety_vectors.npz \
  --method  "XLM-R centroids" \
  --out-dir analysis/multilingual_xlmr/flores/evaluation_results

# 2. Sentence-level
python evaluation/cli.py sentence \
  --vectors analysis/multilingual_xlmr/flores/method_outputs/sentence_vectors.npz \
  --method  "XLM-R sentences" \
  --out-dir analysis/multilingual_xlmr/flores/evaluation_results/sentence_level
```

Compare `distances.csv` (centroid-based) with `sentence_within_between.csv`
(sentence-level) to see whether aggregation distorts the structure.

---

### C. Parallel corpus alignment (FLORES+)

```python
import numpy as np
from evaluation import run_parallel_alignment
from evaluation.cli import FAMILY_GROUPS, FAMILY_COLORS, FAMILY_DISPLAY_NAMES, DISPLAY_NAMES

# Load sentence vectors (one .npy per variety, shape (2009, D))
data   = np.load("analysis/multilingual_xlmr/flores/method_outputs/sentence_vectors.npz",
                 allow_pickle=True)
matrix = data["matrix"]
labels = list(data["labels"])

codes = sorted(set(labels))
sentence_vecs = {c: matrix[[i for i, l in enumerate(labels) if l == c]] for c in codes}

run_parallel_alignment(
    sentence_vecs,
    out_dir="analysis/multilingual_xlmr/flores/parallel_alignment",
    method_label="XLM-R (FLORES+)",
    family_groups=FAMILY_GROUPS,
    family_colors=FAMILY_COLORS,
    family_display_names=FAMILY_DISPLAY_NAMES,
    display_names=DISPLAY_NAMES,
)
```

Key output: `parallel_alignment_dialects.csv` shows each Italian dialect's
sentence-pair similarity to Standard Italian, Spanish, French, Arabic, etc. —
directly measuring contact and distance without any centroid aggregation.

---

### D. Cross-method comparison (FLORES+ all methods)

```bash
python evaluation/cli.py compare \
  --method-dirs \
    "TF-IDF word:analysis/tfidf/flores/evaluation_results/word,\
TF-IDF char:analysis/tfidf/flores/evaluation_results/char,\
Word2Vec:analysis/word2vec/flores/evaluation_results,\
FastText:analysis/fasttext/flores/evaluation_results/fasttext,\
FastText BPE:analysis/fasttext/flores/evaluation_results/bpe,\
XLM-R:analysis/multilingual_xlmr/flores/evaluation_results,\
Sentence-MiniLM:analysis/sentence_baseline/flores/evaluation_results/sentence,\
XLM-R finetuned:analysis/xlmr_finetuned/flores/evaluation_results" \
  --vector-paths \
    "Word2Vec:analysis/word2vec/flores/method_outputs/variety_vectors.npz,\
FastText:analysis/fasttext/flores/method_outputs/variety_vectors.npz,\
XLM-R:analysis/multilingual_xlmr/flores/method_outputs/variety_vectors.npz,\
Sentence-MiniLM:analysis/sentence_baseline/flores/method_outputs/variety_vectors.npz" \
  --out-dir analysis/comparison/flores
```

Key questions answered by the outputs:

- `method_rank_correlation_heatmap.png` — Do surface-level methods (TF-IDF,
  Word2Vec) agree with deep contextual methods (XLM-R)?  High correlation
  implies the signal is mostly orthographic/phonetic; low correlation implies
  the contextual model captures additional structure.
- `method_procrustes_heatmap.png` — Are the embedding geometries structurally
  similar after rotation alignment?
- `method_cka_heatmap.png` — Most invariant view: ignores rotation AND scale.

---

### E. Wikipedia vs. FLORES+ comparison (same method)

```python
from evaluation.compare_methods import run_cross_method_comparison

run_cross_method_comparison(
    {
        "XLM-R (FLORES+)": "analysis/multilingual_xlmr/flores/evaluation_results",
        "XLM-R (wiki)":    "analysis/multilingual_xlmr/wiki/evaluation_results",
    },
    out_dir="analysis/comparison/xlmr_corpus",
    variety_vector_paths={
        "XLM-R (FLORES+)": "analysis/multilingual_xlmr/flores/method_outputs/variety_vectors.npz",
        "XLM-R (wiki)":    "analysis/multilingual_xlmr/wiki/method_outputs/variety_vectors.npz",
    },
)
```

High Spearman ρ + high CKA means the dialect structure is stable across
corpora (more reliable finding).  Low agreement suggests the model is
sensitive to domain rather than language.

---

### F. Dialect-only zoomed analysis

To restrict any evaluation to just the Italian dialects + Italian, pass only
those varieties:

```python
from evaluation import run_evaluation
from evaluation.cli import FAMILY_GROUPS, FAMILY_COLORS, FAMILY_DISPLAY_NAMES, DISPLAY_NAMES
import numpy as np, pandas as pd

df = pd.read_csv("analysis/multilingual_xlmr/flores/method_outputs/variety_vectors.csv",
                 index_col=0)
dialects = ["veneto", "siciliano", "lombardo", "sardo", "ligure", "friulano", "ladino", "italiano"]
df_sub = df.loc[dialects]

run_evaluation(
    df_sub.values,
    list(df_sub.index),
    out_dir="analysis/multilingual_xlmr/flores/evaluation_results_dialects_only",
    method_label="XLM-R — dialects only",
    family_groups={c: FAMILY_GROUPS[c] for c in dialects},
    family_colors=FAMILY_COLORS,
    family_display_names=FAMILY_DISPLAY_NAMES,
    display_names=DISPLAY_NAMES,
    tsne_perplexity=2.5,   # smaller N → smaller perplexity
)
```

---

## 8. Dependencies and optional packages

### Required

All already in `slurm/tools/requirements.txt`:

```
numpy
pandas
scipy
scikit-learn
matplotlib
```

### Optional — UMAP projections

`projection_umap.png` and `sentence_projection_umap.png` are only produced
when `umap-learn` is installed.  If it is absent, a warning is emitted and
those files are skipped without breaking the rest of the evaluation.

```bash
pip install umap-learn
```

### Notes

- The `umap-learn` package name differs from its import name (`umap`).  Do
  not install a package called `umap` (different project).
- On the Bocconi HPC cluster, add `umap-learn` to
  `slurm/tools/requirements.txt` and re-run `slurm/tools/setup_env.sh`.