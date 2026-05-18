"""
Bidirectional check of the historical-influence reference matrix against the
9 trustworthy methods (8 surface + CANINE-MLM).

For each curated (dialect d, contact L) pair we count, for top-K = 3:
  Direction 1 — is L in d's top-K nearest external standards?
  Direction 2 — is d in L's top-K nearest dialect?

Italian is excluded from the external candidate set in Direction 1: it is the
sociolinguistic roof variety and would otherwise dominate the rankings for
every dialect. See the paper's "What embeddings recover" subsection.

Run:
    python -m analysis._comparisons.scripts.02_bidirectional_gold
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
# External candidates for Direction 1 (Italian excluded — see docstring).
EXTERNAL_NO_ITA = ["spa", "fra", "cat", "por", "oci", "deu", "eng", "slv", "hrv", "hun"]

# Historical-influence reference matrix (manually curated qualitative gold).
GOLD = {
    "fur": ["deu", "slv", "hrv"],
    "lij": ["oci", "spa", "fra"],
    "lmo": ["deu", "oci", "spa"],
    "sc":  ["spa", "cat", "fra"],
    "scn": ["spa", "fra", "cat"],
    "vec": ["hrv", "slv", "deu"],
}


def main() -> None:
    Ds = {m: pd.read_csv(REPO_ROOT / p, index_col=0) for m, p in TRUSTWORTHY}

    print("=" * 100)
    print("BIDIRECTIONAL TEST (Italian excluded from Dir1 candidate set)")
    print("=" * 100)

    total_recovered = 0
    total = 0

    for d in DIALECTS:
        print(f"\n[{d}]  expected: {GOLD[d]}")
        print("  Dir1: L in d's top-3 external (no ita) | Dir2: d in L's top-3 dialect")
        print("  " + "-" * 78)
        for L in GOLD[d]:
            total += 1
            dir1, dir2 = 0, 0
            for name, _ in TRUSTWORTHY:
                D = Ds[name]
                d_top3 = D.loc[d, EXTERNAL_NO_ITA].sort_values().head(3).index.tolist()
                L_top3_dial = D.loc[L, DIALECTS].sort_values().head(3).index.tolist()
                if L in d_top3:
                    dir1 += 1
                if d in L_top3_dial:
                    dir2 += 1
            # A pair is "recovered" when a majority (>= 5 / 9) agree in either direction.
            recovered = dir1 >= 5 or dir2 >= 5
            if recovered:
                total_recovered += 1
            mark = "***" if recovered else "   "
            print(f"  {d}<->{L:<5}  Dir1: {dir1}/9  |  Dir2: {dir2}/9   {mark}")

    pct = total_recovered * 100 / total
    print(f"\nTotal recovered (>=5/9 in either direction): {total_recovered}/{total} ({pct:.0f}%)")


if __name__ == "__main__":
    main()
