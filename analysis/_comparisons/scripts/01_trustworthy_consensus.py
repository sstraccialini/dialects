"""
Cross-method consensus for the four headline claims discussed in the paper's
"What embeddings recover" subsection. For each claim we report how many of the
"trustworthy" methods (8 surface methods + the two fine-tuned encoders) agree,
and which methods diverge.

Claims tested:
  1. Lombard and Venetian sit at the centre (rank 1-2 among dialects by mean
     distance to the wider standard inventory), Friulian peripheral.
  2. Sardinian's top-3 nearest non-Italian standards are exclusively Iberian
     (spa, por, cat), with spa at top-1.
  3a. Slovenian's nearest dialect = Venetian.
  3b. Croatian's nearest dialect = Venetian.

Run:
    python -m analysis._comparisons.scripts.01_trustworthy_consensus
"""
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]

# (label, distances.csv relative to REPO_ROOT)
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
    ("XLM-R MLM",             "analysis/multilingual_xlmr/experiments/xlmr_finetuned_wikiOLDI_dialects_native/evaluation_results/flores/centroid/distances.csv"),
]

DIALECTS = ["fur", "lij", "lmo", "sc", "scn", "vec"]
STANDARDS_NON_ITALO = ["ita", "spa", "fra", "cat", "por", "oci", "deu", "eng", "slv", "hrv", "hun"]
IBERIAN = {"spa", "por", "cat"}


def main() -> None:
    Ds = {name: pd.read_csv(REPO_ROOT / rel, index_col=0) for name, rel in TRUSTWORTHY}

    # Claim 1 — Lombard/Venetian central, Friulian peripheral.
    print("=" * 80)
    print("CLAIM 1: 'Lombard and Venetian centre' (rank 1-2 by mean distance to externals),")
    print("         Friulian peripheral (rank >= 4)")
    print("=" * 80)
    for name in (n for n, _ in TRUSTWORTHY):
        D = Ds[name]
        ext_means = {d: D.loc[d, STANDARDS_NON_ITALO].mean() for d in DIALECTS}
        ranks = pd.Series(ext_means).rank(method="min").astype(int).to_dict()
        vec_lmo_top2 = ranks["vec"] <= 2 and ranks["lmo"] <= 2
        fur_peripheral = ranks["fur"] >= 4
        print(f"  {name:<22}  vec={ranks['vec']}, lmo={ranks['lmo']}, fur={ranks['fur']}, "
              f"scn={ranks['scn']}, sc={ranks['sc']}, lij={ranks['lij']}  ->  "
              f"vec&lmo top-2: {vec_lmo_top2}, fur>=4: {fur_peripheral}")

    # Claim 2 — Sardinian's top-3 nearest external standards = Iberian.
    print("\n" + "=" * 80)
    print("CLAIM 2: 'Sardinian top-3 nearest external standards exclusively Iberian'")
    print("=" * 80)
    for name in (n for n, _ in TRUSTWORTHY):
        D = Ds[name]
        top3 = D.loc["sc", STANDARDS_NON_ITALO].sort_values().head(3).index.tolist()
        all_iberian = all(t in IBERIAN for t in top3)
        spa_top1 = top3[0] == "spa"
        print(f"  {name:<22}  sc top-3 = {top3}  ->  all Iberian: {all_iberian} | spa top-1: {spa_top1}")

    # Claim 3a — Slovenian's nearest dialect = Venetian.
    print("\n" + "=" * 80)
    print("CLAIM 3a: 'Slovenian's nearest dialect = Venetian'")
    print("=" * 80)
    for name in (n for n, _ in TRUSTWORTHY):
        top1 = Ds[name].loc["slv", DIALECTS].sort_values().index[0]
        print(f"  {name:<22}  slv top-1 dialect = {top1} {'OK' if top1 == 'vec' else 'NO'}")

    # Claim 3b — Croatian's nearest dialect = Venetian.
    print("\n" + "=" * 80)
    print("CLAIM 3b: 'Croatian's nearest dialect = Venetian'")
    print("=" * 80)
    for name in (n for n, _ in TRUSTWORTHY):
        top1 = Ds[name].loc["hrv", DIALECTS].sort_values().index[0]
        print(f"  {name:<22}  hrv top-1 dialect = {top1} {'OK' if top1 == 'vec' else 'NO'}")


if __name__ == "__main__":
    main()
