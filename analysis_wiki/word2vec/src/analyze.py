import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


def cosine_similarity_matrix(labels, matrix) -> pd.DataFrame:
    sim = cosine_similarity(matrix)
    return pd.DataFrame(sim, index=labels, columns=labels)


def save_similarity_matrix(sim_df: pd.DataFrame, out_path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sim_df.to_csv(out_path)


def plot_pca(labels, matrix, out_path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(matrix)

    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 0], coords[:, 1])

    for i, label in enumerate(labels):
        plt.annotate(label, (coords[i, 0], coords[i, 1]))

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of Variety Vectors")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def nearest_neighbors(sim_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in sim_df.index:
        neighbors = sim_df.loc[label].drop(label).sort_values(ascending=False)
        top = neighbors.head(3)
        rows.append({
            "variety": label,
            "nn1": top.index[0] if len(top) > 0 else None,
            "nn1_score": top.iloc[0] if len(top) > 0 else None,
            "nn2": top.index[1] if len(top) > 1 else None,
            "nn2_score": top.iloc[1] if len(top) > 1 else None,
            "nn3": top.index[2] if len(top) > 2 else None,
            "nn3_score": top.iloc[2] if len(top) > 2 else None,
        })
    return pd.DataFrame(rows)
