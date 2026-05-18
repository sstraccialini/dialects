"""
Comprehensive cross-method breakdown of all dialect <-> standard findings.

For every (dialect, external) pair we report:
  Part A: top-1 external standard per dialect (Italian excluded), with method
          breakdown of agreement vs minority votes.
  Part B: position of each curated historical contact in the dialect's external
          ranking (Italian excluded), to see how high it ranks even when not at
          top-1.
  Part C: top-1 dialect per non-Italo-Romance standard (the reverse direction).
  Part D: position of each dialect in its curated contact's dialect ranking
          (the reverse direction of Part B).
  Part E: integration rank — mean distance from each dialect to the 11 external
          standards (Italian included), averaged across the 9 methods.

Run:
    python -m analysis._paper_results.scripts.04_comprehensive_analysis
"""
from collections import Counter
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
EXT_NO_ITA = ["spa", "fra", "cat", "por", "oci", "deu", "eng", "slv", "hrv", "hun"]
ALL_STD = ["ita"] + EXT_NO_ITA

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
    method_names = [m for m, _ in TRUSTWORTHY]

    print("=" * 90)
    print("PART A — TOP-1 EXTERNAL STANDARD per DIALECT (Italian excluded)")
    print("=" * 90)
    for d in DIALECTS:
        print(f"\n[{d}]  gold = {GOLD[d]}")
        top1s = []
        for m in method_names:
            t1 = Ds[m].loc[d, EXT_NO_ITA].sort_values().index[0]
            top1s.append(t1)
        c = Counter(top1s).most_common()
        print(f"  Top-1 votes: {dict(c)}")
        consensus = c[0][0]
        minorities = {m: t1 for m, t1 in zip(method_names, top1s) if t1 != consensus}
        if minorities:
            print(f"  Consensus = {consensus}; minority methods: {minorities}")

    print("\n" + "=" * 90)
    print("PART B — POSITION of each gold contact in DIALECT's ranking (Italian excluded)")
    print("=" * 90)
    for d in DIALECTS:
        print(f"\n[{d}]  gold = {GOLD[d]}")
        for L in GOLD[d]:
            positions = []
            for m in method_names:
                sorted_ext = Ds[m].loc[d, EXT_NO_ITA].sort_values()
                try:
                    pos = list(sorted_ext.index).index(L) + 1
                    positions.append(pos)
                except ValueError:
                    positions.append(None)
            in_top3 = sum(1 for p in positions if p is not None and p <= 3)
            print(f"  {L}: positions {positions} | in top-3: {in_top3}/9")

    print("\n" + "=" * 90)
    print("PART C — TOP-1 DIALECT per non-Italo-Romance STANDARD")
    print("=" * 90)
    for L in EXT_NO_ITA:
        top1_dial = []
        for m in method_names:
            t1 = Ds[m].loc[L, DIALECTS].sort_values().index[0]
            top1_dial.append(t1)
        c = Counter(top1_dial).most_common()
        print(f"  {L}: top-1 dialect votes = {dict(c)}")

    print("\n" + "=" * 90)
    print("PART D — POSITION of dialect d in standard L's ranking, for each (d,L) gold pair")
    print("=" * 90)
    for d in DIALECTS:
        for L in GOLD[d]:
            positions = []
            for m in method_names:
                sorted_dial = Ds[m].loc[L, DIALECTS].sort_values()
                pos = list(sorted_dial.index).index(d) + 1
                positions.append(pos)
            in_top3 = sum(1 for p in positions if p <= 3)
            is_top1 = sum(1 for p in positions if p == 1)
            print(f"  Std {L} -> dial {d}: pos={positions} | top-1:{is_top1}/9 | top-3:{in_top3}/9")

    print("\n" + "=" * 90)
    print("PART E — Integration rank (mean distance to 11 externals incl. ita)")
    print("=" * 90)
    mean_ranks = {d: 0.0 for d in DIALECTS}
    per_method = {d: [] for d in DIALECTS}
    for m in method_names:
        ext_means = {d: Ds[m].loc[d, ALL_STD].mean() for d in DIALECTS}
        ranks = pd.Series(ext_means).rank(method="min").astype(int).to_dict()
        for d in DIALECTS:
            mean_ranks[d] += ranks[d]
            per_method[d].append(ranks[d])
    for d in DIALECTS:
        mean_ranks[d] /= len(method_names)
    for d, r in sorted(mean_ranks.items(), key=lambda x: x[1]):
        print(f"  {d}: mean rank {r:.2f}, per-method ranks: {per_method[d]}")


if __name__ == "__main__":
    main()
