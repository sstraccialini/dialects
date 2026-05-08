"""
URIEL distance matrices for the 13 project varieties.

For each variety we extract two URIEL feature sets that are RELIABLE for
Italo-Romance dialects (as opposed to the typological feature sets, which
are mostly k-NN-imputed for fur/lij/lmo/vec — see Dataset/typology/README.md):

  - `fam` : 3718-dim one-hot encoding of the position of the variety in the
            Glottolog phylogenetic tree. Distance reflects how many tree
            nodes separate two varieties (genealogy).
  - `geo` : 299-dim vector of geographic features (lat/long + others).
            Distance reflects physical proximity.

Output (Dataset/typology/):
  uriel_fam_distances.csv          13 × 13 cosine distance matrix
  uriel_geo_distances.csv          13 × 13 cosine distance matrix
  uriel_fam_dendrogram.png         dendrogram of fam distances
  uriel_geo_dendrogram.png         dendrogram of geo distances
  uriel_distances_heatmap.png      side-by-side heatmaps (fam | geo)
"""
from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

import lang2vec.lang2vec as l2v

from analysis._shared.varieties import (
    REPO_ROOT,
    VARIETY_CODES,
    VARIETY_NAMES,
    GROUP_COLORS,
    VARIETY_GROUP,
)

OUT_DIR = REPO_ROOT / "Dataset" / "typology"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Project code → ISO 639-3 used by URIEL.  Only "sc" needs remapping.
PROJECT_TO_URIEL = {c: ("srd" if c == "sc" else c) for c in VARIETY_CODES}


def get_feature_matrix(feature_set: str) -> tuple[np.ndarray, list[str]]:
    """Return (matrix [n_varieties × n_features], project_codes_kept)."""
    rows: list[np.ndarray] = []
    kept: list[str] = []
    for code in VARIETY_CODES:
        iso = PROJECT_TO_URIEL[code]
        try:
            feats = l2v.get_features([iso], feature_set)
            v = np.asarray(feats[iso], dtype=float)
            rows.append(v)
            kept.append(code)
        except Exception as exc:
            print(f"  ! {code} (ISO {iso}): missing in {feature_set} ({exc})")
    return np.stack(rows, axis=0), kept


def cosine_distance_matrix(M: np.ndarray) -> np.ndarray:
    """Cosine distance, robust to zero-norm rows (returns 1.0 for those)."""
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    safe = np.where(norms > 0, norms, 1.0)
    Mn = M / safe
    sim = Mn @ Mn.T
    sim = np.clip(sim, -1.0, 1.0)
    return 1.0 - sim


def save_matrix(D: np.ndarray, codes: list[str], path: Path) -> pd.DataFrame:
    df = pd.DataFrame(D, index=codes, columns=codes).round(4)
    df.to_csv(path)
    return df


def plot_dendrogram(D: np.ndarray, codes: list[str], title: str, path: Path):
    iu = np.triu_indices_from(D, k=1)
    Z = linkage(D[iu], method="average")
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"{c} · {VARIETY_NAMES[c]}" for c in codes]
    dendrogram(Z, labels=labels, ax=ax, leaf_rotation=45)
    for tick in ax.get_xticklabels():
        code = tick.get_text().split(" · ")[0]
        tick.set_color(GROUP_COLORS[VARIETY_GROUP[code]])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_heatmap_pair(
    D_fam: np.ndarray, D_geo: np.ndarray, codes: list[str], path: Path
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, D, name in zip(axes, [D_fam, D_geo], ["URIEL fam (genealogy)",
                                                  "URIEL geo (geography)"]):
        im = ax.imshow(D, cmap="viridis", vmin=0, vmax=float(D.max()))
        ax.set_xticks(range(len(codes)))
        ax.set_yticks(range(len(codes)))
        ax.set_xticklabels(codes, rotation=45, ha="right")
        ax.set_yticklabels(codes)
        ax.set_title(name)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    print("Loading URIEL fam ...")
    M_fam, codes_fam = get_feature_matrix("fam")
    D_fam = cosine_distance_matrix(M_fam)

    print("Loading URIEL geo ...")
    M_geo, codes_geo = get_feature_matrix("geo")
    D_geo = cosine_distance_matrix(M_geo)

    if codes_fam != codes_geo:
        raise RuntimeError("fam and geo coverage diverge — please inspect.")
    codes = codes_fam

    df_fam = save_matrix(D_fam, codes, OUT_DIR / "uriel_fam_distances.csv")
    df_geo = save_matrix(D_geo, codes, OUT_DIR / "uriel_geo_distances.csv")

    plot_dendrogram(
        D_fam, codes, "URIEL fam — phylogenetic distance",
        OUT_DIR / "uriel_fam_dendrogram.png",
    )
    plot_dendrogram(
        D_geo, codes, "URIEL geo — geographic distance",
        OUT_DIR / "uriel_geo_dendrogram.png",
    )
    plot_heatmap_pair(
        D_fam, D_geo, codes, OUT_DIR / "uriel_distances_heatmap.png",
    )

    print("\nfam (top-left 6×6):")
    print(df_fam.iloc[:6, :6])
    print("\ngeo (top-left 6×6):")
    print(df_geo.iloc[:6, :6])
    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
