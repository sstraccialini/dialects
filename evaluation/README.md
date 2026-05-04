
### [evaluation/evaluation.py](evaluation/evaluation.py)

New outputs automatically added to every method's `out_dir`:

| New artefact | What it measures |
|---|---|
| `similarity_matrix.csv` | Raw cosine similarity (not distance) |
| `similarity_heatmap.png` | Annotated heatmap, rows/columns sorted by family |
| `projection_umap.png` | UMAP projection (skipped gracefully if `umap-learn` absent) |
| `per_variety_profiles.csv` | Every dialect's ranked distances to all others |
| `per_variety_plots/<code>.png` | Bar chart per variety — "dialect vs the world" |
| `family_stats.csv` | Intra-family vs inter-family distance + cohesion ratio |
| `clustering_metrics.csv` | DB, CH, ARI, NMI, V-measure, cophenetic r, per-variety silhouette |
| `silhouette_report.txt` | Extended: now includes per-variety silhouette ranked table |

New function `run_sentence_evaluation(sentence_vectors, sentence_labels, out_dir, ...)` for sentence-level analysis (within/between distance, silhouette, t-SNE/UMAP of individual sentences).

### [evaluation/parallel_alignment.py](evaluation/parallel_alignment.py)

For FLORES+ aligned sentences. Call `run_parallel_alignment(sentence_vecs_dict, out_dir)` where `sentence_vecs_dict[code]` is shape `(2009, D)`. Produces:
- `parallel_alignment.csv` + heatmap — mean cosine similarity between aligned sentence pairs
- `parallel_alignment_pairs.csv` — all variety pairs ranked
- `parallel_alignment_dialects.csv` — each dialect vs each reference language
- `parallel_alignment_report.txt`

### [evaluation/compare_methods.py](evaluation/compare_methods.py)

Compares multiple methods' distance matrices. Call `run_cross_method_comparison(method_dirs_dict, out_dir)`. Metrics:
- **Spearman ρ** — do two methods rank pairs the same way?
- **Mantel test** — is that correlation statistically significant?
- **Procrustes disparity** — geometric alignment of two embedding spaces (needs raw vectors)
- **Linear CKA** — rotation-invariant similarity of representation matrices (needs raw vectors)

### [evaluation/cli.py](evaluation/cli.py)

**Command to run on Word2Vec** (simplest implemented method with saved vectors):

```bash
python evaluation/cli.py run \
  --vectors  analysis/word2vec/flores/method_outputs/variety_vectors.npz \
  --method   "Word2Vec (FLORES+)" \
  --out-dir  analysis/word2vec/flores/evaluation_results
```

Other sub-commands:
```bash
# Sentence-level (needs sentence_vectors.npz with keys: matrix, labels)
python evaluation/cli.py sentence \
  --vectors analysis/multilingual_xlmr/flores/method_outputs/sentence_vectors.npz \
  --method  "XLM-R sentences" \
  --out-dir analysis/multilingual_xlmr/flores/evaluation_results/sentence_level

# Cross-method comparison
python evaluation/cli.py compare \
  --method-dirs "Word2Vec:analysis/word2vec/flores/evaluation_results,XLM-R:analysis/multilingual_xlmr/flores/evaluation_results" \
  --vector-paths "Word2Vec:analysis/word2vec/flores/method_outputs/variety_vectors.npz,XLM-R:analysis/multilingual_xlmr/flores/method_outputs/variety_vectors.npz" \
  --out-dir analysis/comparison/flores
```