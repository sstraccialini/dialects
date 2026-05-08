"""
GRAMBANK typological distance matrix for the project varieties.

GRAMBANK (Skirgård et al. 2023; integrated in URIEL+ via Khan et al. 2025) is
a typological database with 195 grammatical features for ~2400 languages,
built explicitly to cover varieties that WALS underrepresents.

Coverage of OUR 13 project varieties (checked at install time):

  ✓ ita lmo scn fra cat slv eng     (97-100% of features observed)
  ✓ srd via Campidanese (camp1261)  (~93%)
  ✗ fur lij vec spa deu             (NOT in GRAMBANK)

So this script produces a 8 × 8 distance matrix on the observed subset.

Distance metric: pairwise Hamming distance computed only over features
where BOTH languages have an observed (non-"?") value, normalized by the
number of jointly observed features. This is the standard Skirgård-style
treatment of missing data.

Output (Dataset/typology/):
  grambank_distances.csv             8 × 8 distance matrix
  grambank_features_long.csv         long-format observations table
  grambank_dendrogram.png
  grambank_heatmap.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from scipy.cluster.hierarchy import linkage, dendrogram

from analysis._shared.varieties import (
    REPO_ROOT,
    VARIETY_CODES,
    VARIETY_NAMES,
    GROUP_COLORS,
    VARIETY_GROUP,
)

OUT_DIR  = REPO_ROOT / "Dataset" / "typology"
DATA_DIR = OUT_DIR / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

GRAMBANK_BASE = "https://raw.githubusercontent.com/grambank/grambank/master/cldf"
LANGUAGES_URL = f"{GRAMBANK_BASE}/languages.csv"
VALUES_URL    = f"{GRAMBANK_BASE}/values.csv"

# Project code → Glottocode used by GRAMBANK.
# Sardinian: GRAMBANK has Campidanese (camp1261) and Logudorese-Nuorese
# (logu1236), but only Campidanese has features. We map "sc" → camp1261.
# Missing varieties (fur, lij, vec, spa, deu) are deliberately absent below.
PROJECT_TO_GLOTTOCODE = {
    "lmo": "lomb1257",   # Lombard
    "sc":  "camp1261",   # Sardinian (Campidanese)
    "scn": "sici1248",   # Sicilian
    "ita": "ital1282",   # Italian
    "fra": "stan1290",   # French
    "cat": "stan1289",   # Catalan
    "slv": "slov1268",   # Slovenian
    "eng": "stan1293",   # English
}


def download_if_missing(url: str, dest: Path) -> Path:
    if not dest.exists():
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  saved to {dest} ({dest.stat().st_size/1e6:.1f} MB)")
    return dest


def load_grambank() -> pd.DataFrame:
    """Return long-format observations restricted to our 8 varieties."""
    download_if_missing(LANGUAGES_URL, DATA_DIR / "grambank_languages.csv")
    values_path = download_if_missing(VALUES_URL, DATA_DIR / "grambank_values.csv")

    df = pd.read_csv(values_path, low_memory=False)
    df = df[df["Language_ID"].isin(PROJECT_TO_GLOTTOCODE.values())].copy()
    glott_to_code = {g: c for c, g in PROJECT_TO_GLOTTOCODE.items()}
    df["project_code"] = df["Language_ID"].map(glott_to_code)
    return df[["project_code", "Language_ID", "Parameter_ID", "Value"]]


def build_feature_matrix(long_df: pd.DataFrame) -> pd.DataFrame:
    """Wide pivot. Cells are 0/1/2/3 strings or NaN for "?" / unknown."""
    long_df = long_df.copy()
    long_df["Value"] = long_df["Value"].replace("?", np.nan)
    wide = long_df.pivot_table(
        index="project_code",
        columns="Parameter_ID",
        values="Value",
        aggfunc="first",
    )
    return wide


def pairwise_hamming(W: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Hamming distance, normalized over jointly observed features per pair."""
    codes = list(W.index)
    M = W.to_numpy(dtype=object)
    n = len(codes)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            mask = pd.notna(M[i]) & pd.notna(M[j])
            n_joint = int(mask.sum())
            if n_joint == 0:
                D[i, j] = D[j, i] = 1.0
                continue
            mismatches = int(np.sum(M[i, mask] != M[j, mask]))
            D[i, j] = D[j, i] = mismatches / n_joint
    return D, codes


def save_matrix(D: np.ndarray, codes: list[str], path: Path) -> pd.DataFrame:
    df = pd.DataFrame(D, index=codes, columns=codes).round(4)
    df.to_csv(path)
    return df


def plot_dendrogram(D: np.ndarray, codes: list[str], path: Path):
    iu = np.triu_indices_from(D, k=1)
    Z = linkage(D[iu], method="average")
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"{c} · {VARIETY_NAMES[c]}" for c in codes]
    dendrogram(Z, labels=labels, ax=ax, leaf_rotation=45)
    for tick in ax.get_xticklabels():
        code = tick.get_text().split(" · ")[0]
        tick.set_color(GROUP_COLORS[VARIETY_GROUP[code]])
    ax.set_title("GRAMBANK typological distance (Hamming, jointly observed features)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_heatmap(D: np.ndarray, codes: list[str], path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(D, cmap="viridis", vmin=0, vmax=float(D.max()))
    ax.set_xticks(range(len(codes)))
    ax.set_yticks(range(len(codes)))
    ax.set_xticklabels(codes, rotation=45, ha="right")
    ax.set_yticklabels(codes)
    for i in range(len(codes)):
        for j in range(len(codes)):
            ax.text(j, i, f"{D[i,j]:.2f}",
                    ha="center", va="center",
                    color="white" if D[i, j] > D.max() * 0.55 else "black",
                    fontsize=9)
    ax.set_title("GRAMBANK distance")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    long_df = load_grambank()
    long_df.to_csv(OUT_DIR / "grambank_features_long.csv", index=False)
    print(f"Loaded {len(long_df)} observations across "
          f"{long_df['project_code'].nunique()} varieties.")

    W = build_feature_matrix(long_df)
    coverage = W.notna().sum(axis=1)
    print("\nObserved features per variety (out of {}):".format(W.shape[1]))
    for code, n in coverage.items():
        print(f"  {code:<5}  {VARIETY_NAMES[code]:<22}  {n} / {W.shape[1]}")

    D, codes = pairwise_hamming(W)
    df = save_matrix(D, codes, OUT_DIR / "grambank_distances.csv")
    plot_dendrogram(D, codes, OUT_DIR / "grambank_dendrogram.png")
    plot_heatmap(D, codes, OUT_DIR / "grambank_heatmap.png")

    print("\nDistance matrix:")
    print(df)
    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
