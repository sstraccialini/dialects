"""
Per-method tally of how many of the 18 historical-influence pairs each method
recovers. A pair (dialect d, contact L) is "recovered" by a method if L appears
in d's top-3 nearest external (Dir1), or d appears in L's top-3 nearest dialect
(Dir2), or both.

Italian is excluded from the Direction-1 candidate set. Direction-2's candidate
set is the six dialects so Italian never enters there.

Run:
    python -m analysis._comparisons.scripts.03_per_method_recovery
"""
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]

TRUSTWORTHY = [
    ("TF-IDF char (norm)",    "analysis/tfidf/experiments/tfidf_wikiOLDI_normalized/evaluation_results/flores/char/distances.csv"),
    ("TF-IDF char (native)",  "analysis/tfidf/experiments/tfidf_wikiOLDI_native/evaluation_results/flores/char/distances.csv"),
    ("TF-IDF word (norm)",    "analysis/tfidf/experiments/tfidf_wikiOLDI_normalized/evaluation_results/flores/word/distances.csv"),
    ("TF-IDF word (native)",  "analysis/tfidf/experiments/tfidf_wikiOLDI_native/evaluation_results/flores/word/distances.csv"),
    ("FastText (norm)",       "analysis/fasttext/experiments/fasttext_wikiOLDI_normalized/evaluation_results/flores/centroid/distances.csv"),
    ("FastText (native)",     "analysis/fasttext/experiments/fasttext_wikiOLDI_native/evaluation_results/flores/centroid/distances.csv"),
    ("Word2Vec (norm)",       "analysis/word2vec/experiments/word2vec_wikiOLDI_normalized/evaluation_results/flores/centroid/distances.csv"),
    ("Word2Vec (native)",     "analysis/word2vec/experiments/word2vec_wikiOLDI_native/evaluation_results/flores/centroid/distances.csv"),
    ("CANINE-MLM",            "analysis/canine/experiments/canine_finetuned_wikiOLDI_dialects_native/evaluation_results/flores/centroid/distances.csv"),
]

DIALECTS = ["fur", "lij", "lmo", "sc", "scn", "vec"]
EXTERNAL_NO_ITA = ["spa", "fra", "cat", "por", "oci", "deu", "eng", "slv", "hrv", "hun"]

GOLD = {
    "fur": ["deu", "slv", "hrv"],
    "lij": ["oci", "spa", "fra"],
    "lmo": ["deu", "oci", "spa"],
    "sc":  ["spa", "cat", "fra"],
    "scn": ["spa", "fra", "cat"],
    "vec": ["hrv", "slv", "deu"],
}


def main() -> None:
    print("=" * 80)
    print("PER-METHOD CONTACT RECOVERY (18 pairs, bidirectional, K=3)")
    print("=" * 80)

    results = []
    for name, rel in TRUSTWORTHY:
        D = pd.read_csv(REPO_ROOT / rel, index_col=0)
        recovered = 0
        dir1_only = 0
        dir2_only = 0
        both_dirs = 0
        for d, contacts in GOLD.items():
            d_top3 = D.loc[d, EXTERNAL_NO_ITA].sort_values().head(3).index.tolist()
            for L in contacts:
                L_top3_dial = D.loc[L, DIALECTS].sort_values().head(3).index.tolist()
                in_dir1 = L in d_top3
                in_dir2 = d in L_top3_dial
                if in_dir1 or in_dir2:
                    recovered += 1
                    if in_dir1 and in_dir2:
                        both_dirs += 1
                    elif in_dir1:
                        dir1_only += 1
                    else:
                        dir2_only += 1
        results.append((name, recovered, dir1_only, dir2_only, both_dirs))

    results.sort(key=lambda x: -x[1])

    print(f"\n  {'Method':<24} {'Recov./18':>10} {'Dir1 only':>10} {'Dir2 only':>10} {'Both':>6}")
    print("  " + "-" * 65)
    for name, r, d1, d2, b in results:
        print(f"  {name:<24} {r:>4}/18    {d1:>6}     {d2:>6}     {b:>4}")

    print("\nNotes:")
    print("  - Dir1: contact L in dialect d's top-3 nearest non-Italian external")
    print("  - Dir2: dialect d in contact L's top-3 nearest dialect")
    print("  - 'Both' = recovered in both directions (most robust)")


if __name__ == "__main__":
    main()
