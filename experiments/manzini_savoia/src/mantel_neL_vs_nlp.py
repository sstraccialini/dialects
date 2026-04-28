"""
Mantel correlation: NorthEuraLex (1016 IPA words/lang) vs NLP method
distances. Subset to 8 languages overlapping between NorthEuraLex and FLORES.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
NEL_CSV = REPO_ROOT / "manzini_savoia" / "results" / "neL_phonetic_distance.csv"

# Mapping NorthEuraLex labels (our internal labels) -> FLORES variety codes.
NEL_TO_FLORES = {
    "italian":   "italiano",
    "spanish":   "spagnolo",
    "catalan":   "catalano",
    "french":    "francese",
    "english":   "inglese",
    "german":    "tedesco",
    "greek":     "greco",
    "slovenian": "sloveno",
}

NLP_METHODS = {
    "tfidf_word":   "analysis_flores/tfidf/results/word/distances.csv",
    "tfidf_char":   "analysis_flores/tfidf/results/char/distances.csv",
    "word2vec":     "analysis_flores/word2vec/results/distances.csv",
    "fasttext":     "analysis_flores/subword_fasttext/results/fasttext/distances.csv",
    "bpe":          "analysis_flores/subword_fasttext/results/bpe/distances.csv",
    "xlmr":         "analysis_flores/multilingual/results/distances.csv",
    "minilm":       "analysis_flores/sentence_baseline/results/sentence/distances.csv",
    "labse":        "analysis_flores/labse/results/sentence/distances.csv",
    "xlmr_adapted": "analysis_flores/multilingual_adapted/results/distances.csv",
    "xlmr_contrastive": "contrastive/results/distances.csv",
}


def to_condensed(matrix):
    n = matrix.shape[0]
    return matrix[np.triu_indices(n, k=1)]


def mantel(d1, d2, n_perm=500, seed=42):
    rng = np.random.default_rng(seed)
    cond1 = to_condensed(d1)
    cond2 = to_condensed(d2)
    rho, _ = spearmanr(cond1, cond2)
    n = d1.shape[0]
    null = []
    for _ in range(n_perm):
        perm = rng.permutation(n)
        d2p = d2[perm][:, perm]
        nr, _ = spearmanr(cond1, to_condensed(d2p))
        null.append(nr)
    p = float(np.mean(np.abs(np.array(null)) >= np.abs(rho)))
    return float(rho), p


def main():
    print("=== Mantel: NorthEuraLex vs NLP methods ===")
    print(f"  ground truth: NorthEuraLex (1016 IPA words/language)")
    print()

    nel = pd.read_csv(NEL_CSV, index_col=0)
    nel_labels = [a for a in NEL_TO_FLORES if a in nel.index]
    flores_labels = [NEL_TO_FLORES[a] for a in nel_labels]
    print(f"Common languages: {len(nel_labels)}")
    for a, f in zip(nel_labels, flores_labels):
        print(f"  {a:12s} <-> {f}")
    print()

    nel_sub = nel.loc[nel_labels, nel_labels]
    nel_mat = nel_sub.values

    print(f"{'Method':<20s}  {'Spearman r':>12s}  {'p-value':>10s}")
    print("-" * 50)
    rows = []
    for method, rel_path in NLP_METHODS.items():
        try:
            nlp_df = pd.read_csv(REPO_ROOT / rel_path, index_col=0)
            nlp_sub = nlp_df.loc[flores_labels, flores_labels]
        except (FileNotFoundError, KeyError) as e:
            print(f"{method:<20s}  <missing/error>: {e}")
            continue
        rho, p = mantel(nel_mat, nlp_sub.values, n_perm=500)
        marker = "*" if p < 0.05 else ""
        print(f"{method:<20s}  {rho:>+12.4f}  {p:>10.4f} {marker}")
        rows.append({"method": method, "spearman_r": rho, "p_value": p})

    out = REPO_ROOT / "manzini_savoia" / "results" / "mantel_neL_vs_nlp.csv"
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.4f")
    print()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
