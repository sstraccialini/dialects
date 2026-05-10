# plots/

Standalone visualisations of the saved variety embeddings.  All scripts read
from the existing `analysis/.../method_outputs/` and `evaluation_results/`
artefacts — nothing is re-trained or re-embedded here.

```
plots/
├── scripts/                 # self-contained Python (pure matplotlib + numpy)
│   ├── _common.py           # taxonomy, geo coords, source registry, loaders
│   ├── 01_method_comparison_panel.py
│   ├── 02_italy_dialect_map.py
│   ├── 03_dialect_radar.py
│   ├── 04_embedding_galaxy.py
│   ├── 05_cross_method_agreement.py
│   ├── 06_dendrogram_panel.py
│   └── 07_silhouette_scoreboard.py
└── outputs/                 # generated PNGs (committed)
```

Run any script standalone:

```bash
python plots/scripts/02_italy_dialect_map.py
```

## What each plot says

### 01 – method comparison panel
3×4 grid of cosine similarity heatmaps, one per evaluated method, sorted
by family.  At a glance: which models keep variety identity (heat
gradient), which collapse everything to high similarity (uniformly green),
and which destroy structure under fine-tuning (XLM-R post-TLM goes red).

### 02 – Italy dialect map
Two-panel geographic map of Italy.  Italo-Romance dialects sit at their
regional capitals; edges connect every pair, thickness/colour encoding
inter-dialect cosine similarity.  Marker size encodes similarity to
standard Italian.  XLM-R (left) sees dense in-dialect similarity;
Word2Vec (right) keeps dialects more distinct lexically.

### 03 – dialect radar
Two-panel polar plot for each Italo-Romance dialect, axes = reference
languages.  **Left** is raw cosine similarity (all dialects look identical
— centroid collapse).  **Right** subtracts the per-reference mean across
dialects, exposing each dialect's actual lean.  This is the visual case
for the new content-subtracted centroid metric in
[evaluation/sentence_relations.py](../evaluation/sentence_relations.py).

### 04 – embedding galaxy
2-D MDS projection on a dark "starfield" background.  Side-by-side
XLM-R vs Word2Vec.  Dashed orange lines = standard Italian's nearest-3
neighbours.  Same data, two completely different topologies.

### 05 – cross-method agreement
Spearman ρ between every pair of methods' similarity matrices, computed on
their shared varieties.  Blocks of agreement reveal which methods see the
data the same way (e.g. all Word2Vec-style methods cluster together;
fine-tuned XLM-R variants drift apart).

### 06 – dendrogram panel
Average-linkage dendrograms for every method.  Colour-coded leaves
(family).  Useful for spotting which methods correctly nest the
Italo-Romance dialects under a Romance super-cluster.

### 07 – silhouette scoreboard
Bar chart of `silhouette_family` (8 families) and `silhouette_romance_vs_rest`
for every evaluated model, parsed from their saved `silhouette_report.txt`
or `clustering_metrics.csv`.  The headline quantitative comparison.

## Adding a new method
Edit `_common.py`'s `SAVED_VECTOR_SOURCES` or `NEW_SIM_SOURCES` lists.
The path can point to a `variety_vectors.npz`, a `variety_vectors.csv`,
or a `similarity_matrix.csv` — the loader auto-detects.  All seven plots
will pick it up automatically on next run.
