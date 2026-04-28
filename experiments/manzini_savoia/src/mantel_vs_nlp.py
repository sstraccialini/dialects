"""
Mantel correlation: ASJP phonetic distance (ground truth) vs NLP method
distances on the same set of varieties.

Subset: only varieties present in both ASJP analysis and FLORES-based
NLP analyses.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASJP_CSV = REPO_ROOT / "manzini_savoia" / "results" / "asjp_phonetic_distance.csv"

# Mapping ASJP labels -> FLORES variety codes used in our pipelines.
# Only varieties that exist in BOTH are kept.
ASJP_TO_FLORES = {
    "italian":   "italiano",
    "sicilian":  "siciliano",
    "lombard":   "lombardo",
    "ligurian":  "ligure",
    "friulian":  "friulano",
    "sardinian": "sardo",
    "spanish":   "spagnolo",
    "catalan":   "catalano",
    "french":    "francese",
    "english":   "inglese",
    "german":    "tedesco",
    "greek":     "greco",
    "arabic":    "arabo",
    "slovenian": "sloveno",
}

# Distance matrix files for each NLP method
NLP_METHODS = {
    "tfidf_word":      "analysis_flores/tfidf/results/word/distances.csv",
    "tfidf_char":      "analysis_flores/tfidf/results/char/distances.csv",
    "word2vec":        "analysis_flores/word2vec/results/distances.csv",
    "fasttext":        "analysis_flores/subword_fasttext/results/fasttext/distances.csv",
    "bpe":             "analysis_flores/subword_fasttext/results/bpe/distances.csv",
    "xlmr":            "analysis_flores/multilingual/results/distances.csv",
    "minilm":          "analysis_flores/sentence_baseline/results/sentence/distances.csv",
    "labse":           "analysis_flores/labse/results/sentence/distances.csv",
    "xlmr_adapted":    "analysis_flores/multilingual_adapted/results/distances.csv",
    "xlmr_contrastive": "contrastive/results/distances.csv",
}


def load_asjp_subset() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load ASJP matrix, return subset for varieties also in FLORES.
    Returns (subset_matrix_df, asjp_labels_kept, flores_labels_kept).
    """
    df = pd.read_csv(ASJP_CSV, index_col=0)
    asjp_labels = [a for a in ASJP_TO_FLORES if a in df.index]
    flores_labels = [ASJP_TO_FLORES[a] for a in asjp_labels]
    sub = df.loc[asjp_labels, asjp_labels].copy()
    return sub, asjp_labels, flores_labels


def load_nlp_distance(path_rel: str) -> pd.DataFrame:
    p = REPO_ROOT / path_rel
    df = pd.read_csv(p, index_col=0)
    return df


def to_condensed(matrix: np.ndarray) -> np.ndarray:
    """Upper triangle (excluding diagonal) flattened."""
    n = matrix.shape[0]
    return matrix[np.triu_indices(n, k=1)]


def mantel_test(d1: np.ndarray, d2: np.ndarray, n_perm: int = 1000, seed: int = 42):
    """Simple Mantel test via Spearman + permutation."""
    rng = np.random.default_rng(seed)
    cond1 = to_condensed(d1)
    cond2 = to_condensed(d2)
    rho, _ = spearmanr(cond1, cond2)
    n = d1.shape[0]
    # Permutation
    null_rhos = []
    for _ in range(n_perm):
        perm = rng.permutation(n)
        d2_perm = d2[perm][:, perm]
        cond_perm = to_condensed(d2_perm)
        nr, _ = spearmanr(cond1, cond_perm)
        null_rhos.append(nr)
    null_rhos = np.array(null_rhos)
    p_value = float(np.mean(np.abs(null_rhos) >= np.abs(rho)))
    return float(rho), p_value


def main():
    print("=== Mantel: ASJP phonetic ground truth vs NLP methods ===")
    print()

    asjp_sub, asjp_labels, flores_labels = load_asjp_subset()
    print(f"Common varieties (ASJP <-> FLORES): {len(asjp_labels)}")
    for a, f in zip(asjp_labels, flores_labels):
        print(f"  {a:12s}  <->  {f}")
    print()

    asjp_mat = asjp_sub.values

    print("\n=== Mantel correlations ===")
    print(f"{'Method':<20s}  {'Spearman r':>12s}  {'p-value':>10s}")
    print("-" * 50)

    results = []
    for method, rel_path in NLP_METHODS.items():
        try:
            nlp_df = load_nlp_distance(rel_path)
        except FileNotFoundError:
            print(f"{method:<20s}  {'<missing>':>12s}")
            continue
        # Reorder rows/cols to match flores_labels
        try:
            nlp_sub = nlp_df.loc[flores_labels, flores_labels]
        except KeyError as e:
            print(f"{method:<20s}  KeyError: {e}")
            continue
        nlp_mat = nlp_sub.values

        rho, p = mantel_test(asjp_mat, nlp_mat, n_perm=500)
        marker = "*" if p < 0.05 else ""
        print(f"{method:<20s}  {rho:>+12.4f}  {p:>10.4f} {marker}")
        results.append({"method": method, "spearman_r": rho, "p_value": p})

    df_res = pd.DataFrame(results)
    out = REPO_ROOT / "manzini_savoia" / "results" / "mantel_asjp_vs_nlp.csv"
    df_res.to_csv(out, index=False, float_format="%.4f")
    print()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
