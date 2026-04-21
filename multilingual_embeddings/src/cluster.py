import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import umap
from sklearn.decomposition import PCA

from .config import RESULTS_DIR, DPI

def analyze_and_plot(df, embeddings):
    """
    Run clustering, manifold alignments and plot distributions.
    Saves outputs to RESULTS_DIR.
    """
    if not RESULTS_DIR.exists():
        RESULTS_DIR.mkdir(parents=True)

    # Calculate Language Centroids (Average Embedding per Language)
    lang_centroids = {}
    for lang in df['lang'].unique():
        lang_idx = df[df['lang'] == lang].index
        lang_embeddings = embeddings[lang_idx]
        lang_centroids[lang] = np.mean(lang_embeddings, axis=0)

    # 1. Similarity Matrix & Heatmap
    lang_names = list(lang_centroids.keys())
    centroid_matrix = np.array([lang_centroids[l] for l in lang_names])
    
    sim_matrix = cosine_similarity(centroid_matrix)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(sim_matrix, xticklabels=lang_names, yticklabels=lang_names, 
                annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Cross-Language Cosine Similarity Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "cross_language_similarity.png", dpi=DPI)
    plt.close()
    
    # 2. Hierarchical Clustering (Dendrogram)
    # Using 'ward' linkage on centroid matrix
    plt.figure(figsize=(10, 6))
    linked = linkage(centroid_matrix, method='ward')
    dendrogram(linked, labels=lang_names, leaf_rotation=45, leaf_font_size=12)
    plt.title("Language Clustering Dendrogram")
    plt.ylabel("Ward Distance")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "language_dendrogram.png", dpi=DPI)
    plt.close()

    # 3. UMAP Projection for all Individual Texts
    try:
        import umap.umap_ as umap
        reducer = umap.UMAP(n_components=2, random_state=42)
        projected = reducer.fit_transform(embeddings)

        plt.figure(figsize=(12, 10))
        sns.scatterplot(x=projected[:, 0], y=projected[:, 1], hue=df['lang'], palette='tab20', alpha=0.6)
        plt.title(f"UMAP Projection of Multilingual Texts (Models capturing Dialects)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "texts_umap_projection.png", dpi=DPI)
        plt.close()
    except Exception as e:
        print(f"Skipping UMAP Projection: {e}")

    # 4. K-Means evaluation (Checking structural alignment)
    # Let's say we expect 4 major language families/clusters initially
    k = min(len(lang_names), 4)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clustering_labels = kmeans.fit_predict(embeddings)
    score = silhouette_score(embeddings, clustering_labels)
    
    pd.DataFrame({
        "Text Cluster": clustering_labels,
        "Original Language": df['lang']
    }).groupby(['Text Cluster', 'Original Language']).size().unstack(fill_value=0).to_csv(RESULTS_DIR / "kmeans_cluster_dist.csv")
    
    with open(RESULTS_DIR / "clustering_stats.txt", "w") as f:
        f.write(f"K-Means Clustering applied (k={k})\n")
        f.write(f"Overall Silhouette Score: {score:.4f}\n")
    print("Done generating clustering visualizations.")
