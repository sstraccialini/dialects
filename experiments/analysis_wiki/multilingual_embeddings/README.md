# Person 4: Multilingual / Pretrained Embeddings

This module leverages pre-trained multilingual language models (such as `xlm-roberta-base`, mBERT, etc.) to evaluate whether native cross-lingual alignments implicitly capture historical relationships between standard languages and local dialects (like Neapolitan, Lombard, Sicilian, Venetian, etc.).

## 📌 Rationale
Models like mBERT and XLM-R are exposed to massive multilingual corpora. By mapping dialectal sentences to the high-dimensional space spanned by these models, we expect to see:
1. **Strong Cross-Language Alignment**: Sentences with overlapping meaning should lie closer together.
2. **Historical Relationships**: The distance between language centroids (mean representation of a corpus subset) in the embedding space will often mirror phylogenetic trees of romance languages (e.g. Italian aligning closely with dialects, distinct from Germanic groups).

## 🛠 Prerequisites and Training

We don't need to "train" a new embedding model from scratch. Instead, we perform **feature extraction** using standard pooling mechanisms across pre-trained models.

### Environment Setup
Create a new Virtual Environment or use existing dependencies:
```bash
python -m pip install -r requirements.txt
```

### Configuration
In `src/config.py`, adjust key parameters:
* `MODEL_NAME`: The HuggingFace identifier. Default is `xlm-roberta-base`, but you may use `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` for highly optimized pooling out-of-the-box.
* `SAMPLES_PER_LANG`: Number of text sentences to sample from each `.csv`. Dialects will pull up to their maximum available datasets automatically if under this bound.

### Running the Analysis
To load the datasets, encode all texts into multilingual space, and automatically run the unsupervised grouping checks:
```bash
python run_pipeline.py
```

## 📊 Outputs and Visualizations

All generated outputs will be stored inside `../results/`.

1. **`cross_language_similarity.png`**: Heatmap of the Cosine Similarity mathematically describing how close language subsets are. We expect dialects to peak symmetrically with standard Italian.
2. **`language_dendrogram.png`**: A hierarchically clustered tree showing family groupings. Used directly to test if linguistic categories overlap with topological distances (e.g. Sc/Nap/Lmo/Vec -> Italian -> Spanish/Fr vs German/English).
3. **`texts_umap_projection.png`**: UMAP dimensionality reduction allowing a visual sweep of how "cloudy" or distinct each language rests when restricted to semantic embeddings.
4. **`clustering_stats.txt` & `kmeans_cluster_dist.csv`**: K-Means allocation tables outputting cross-tabulations on true language vs unsupervised cluster assignments. 

## 🚀 Key Takeaways
If valid historical mapping occurs, cross-lingual clustering essentially validates that sub-word tokenization and deep transformer layers inherently capture etymological bridges over highly correlated languages, requiring **zero task-specific dialectal supervision**.