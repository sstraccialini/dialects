# Interpretation of Results — Subword / FastText (Person 5)

> **Fill this file after running `run_approach.py`.**
> Use the structure below as a template; replace the placeholder text
> with your actual numbers and observations.

---

## 1. What we did in one paragraph

*(Write after running)*

---

## 2. FastText pipeline results

### `distances.csv` — cosine distance matrix
*(Paste or summarize the key distances: Neapolitan–Sicilian, Italian–Neapolitan, etc.)*

### `nearest_neighbors.csv`
*(For each dialect, list its top-3 neighbors and distances)*

### `variety_vectors.csv`
*(Note the vector dimensionality; describe any obvious clustering in the raw vectors)*

### Silhouette scores
| Label set | Score |
|---|---|
| Family (7 groups) | *fill* |
| Romance vs rest   | *fill* |

### Dendrogram observations
*(Does the dendrogram separate Romance from non-Romance? Are Italo-Romance dialects clustered?)*

### MDS / t-SNE observations
*(Describe the 2D layout; compare to Person 1's baseline plots)*

---

## 3. BPE pipeline results

### `distances.csv`
*(Key pairwise distances)*

### `nearest_neighbors.csv`
*(Top-3 neighbors per dialect)*

### `top_features.csv` — most characteristic BPE pieces
*(List the top-5 BPE pieces for 2–3 dialects and comment on what they reveal
about morphology / orthography)*

### Silhouette scores
| Label set | Score |
|---|---|
| Family (7 groups) | *fill* |
| Romance vs rest   | *fill* |

### Dendrogram / projection observations

---

## 4. Comparison across approaches

| Pipeline | Sil. family | Sil. romance | Notes |
|---|---|---|---|
| TF-IDF word (Person 1) | — | — | Baseline reference |
| TF-IDF char (Person 1) | — | — | Baseline reference |
| FastText (subword) | *fill* | *fill* | |
| BPE + TF-IDF | *fill* | *fill* | |

*(Higher silhouette = varieties from the same family are more tightly grouped
relative to other families.)*

---

## 5. Key findings

*(3–5 bullet points summarizing the most interesting findings)*

- 
- 
- 

---

## 6. Limitations and future work

- FastText variety vectors are averaged over sentences; sentence order
  and syntactic structure are lost.
- BPE vocabulary is trained jointly on all 14 varieties; training a
  per-language BPE might capture language-specific morphology better.
- Increasing `--sample-size` to 50k sentences may improve stability of
  FastText embeddings for high-resource languages (Italian, Spanish, etc.).
