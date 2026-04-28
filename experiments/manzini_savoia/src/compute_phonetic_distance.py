"""
Compute phonetic-lexical distances between Italian dialect regions using
the Manzini-Savoia corpus (62k records, IPA + Italian gloss + alignments).

Pipeline:
1. Load JSON, normalise region names, drop noise
2. Group by region; build italian_word -> list[IPA_forms] dict per region
3. For each pair of regions:
   - shared_words = set of italian words present in both
   - For each shared word: take majority IPA form per region, Levenshtein
   - distance = mean(normalised Levenshtein) over shared words
4. Output: distance matrix CSV, dendrogram PNG, summary

Run from repo root:
    python experiments/manzini_savoia/src/compute_phonetic_distance.py
"""
from __future__ import annotations

import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
JSON_PATH = REPO_ROOT / "MS_corpus_aligned_tagged_v2.0.json"
OUT_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Region name normalisation: fix typos, drop noise.
REGION_FIX = {
    "Trentino Alto-Adige": "Trentino-Alto Adige",
}
DROP_REGIONS = {None, "", "(none)"}


def normalise_region(reg: str | None) -> str | None:
    if reg is None:
        return None
    reg = reg.strip()
    if reg in DROP_REGIONS:
        return None
    return REGION_FIX.get(reg, reg)


# Filter out grammatical annotations from glosses. Manzini-Savoia uses
# tags like "ClS-3pm" (subject clitic 3rd plural masc), "neg", "ART",
# etc. We keep only italian content words (lowercase + accents only).
GRAMMATICAL_PATTERN = re.compile(r"^(?:ClS|cls|ClO|cl|art|neg|aux|cop|imp|pro|prep|aux)[\-\d]?", re.IGNORECASE)
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def is_content_word(italian: str) -> bool:
    """Return True if `italian` looks like a normal italian word."""
    if not italian:
        return False
    italian = italian.strip()
    if len(italian) < 2:
        return False
    if GRAMMATICAL_PATTERN.match(italian):
        return False
    # has hyphen + digit (e.g., "ClS-3pm")
    if "-" in italian and any(c.isdigit() for c in italian):
        return False
    # mostly letters (allow accented)
    letters = sum(1 for c in italian if c.isalpha())
    if letters / max(len(italian), 1) < 0.6:
        return False
    return True


# Levenshtein distance (no external dep needed)
def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            curr[j] = min(ins, dele, sub)
        prev = curr
    return prev[-1]


def normalised_levenshtein(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    return levenshtein(a, b) / max(len(a), len(b))


def main():
    print(f"Loading {JSON_PATH} ...")
    with open(JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  {len(data)} records")

    # Region -> italian_word -> Counter[ipa_form]
    region_word_forms: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))

    n_pairs_total = 0
    n_pairs_kept = 0

    for rec in data:
        region = normalise_region(rec.get("Regione"))
        if region is None:
            continue
        for pair in rec.get("aligned_pairs_list") or []:
            try:
                ipa, italian = pair
            except Exception:
                continue
            n_pairs_total += 1
            italian_clean = italian.lower().strip().translate(PUNCT_TABLE)
            if not is_content_word(italian_clean):
                continue
            ipa_clean = ipa.strip()
            if not ipa_clean:
                continue
            region_word_forms[region][italian_clean][ipa_clean] += 1
            n_pairs_kept += 1

    print(f"  pairs: {n_pairs_total} total, {n_pairs_kept} kept after filtering")
    print(f"  regions kept: {len(region_word_forms)}")

    # Reduce: per region, italian_word -> majority ipa form
    region_lex: dict[str, dict[str, str]] = {}
    for reg, words in region_word_forms.items():
        region_lex[reg] = {w: forms.most_common(1)[0][0] for w, forms in words.items()}

    # Drop regions with too few words to compute robust distances
    MIN_WORDS = 50
    regions = sorted([r for r in region_lex if len(region_lex[r]) >= MIN_WORDS])
    print(f"  regions with >= {MIN_WORDS} content words: {len(regions)}")
    print(f"  excluded: {sorted(set(region_lex) - set(regions))}")
    print()
    print("  region -> n_unique_words")
    for r in regions:
        print(f"    {r:30s}  {len(region_lex[r]):5d}")
    print()

    # Pairwise distance computation
    n = len(regions)
    dist_matrix = np.zeros((n, n), dtype=float)
    shared_counts = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(i + 1, n):
            ra, rb = regions[i], regions[j]
            shared = set(region_lex[ra]) & set(region_lex[rb])
            if not shared:
                d = float("nan")
            else:
                distances = [
                    normalised_levenshtein(region_lex[ra][w], region_lex[rb][w])
                    for w in shared
                ]
                d = float(np.mean(distances))
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
            shared_counts[i, j] = len(shared)
            shared_counts[j, i] = len(shared)

    df = pd.DataFrame(dist_matrix, index=regions, columns=regions)
    out_csv = OUT_DIR / "ms_phonetic_distance.csv"
    df.to_csv(out_csv, float_format="%.4f")
    print(f"  Saved: {out_csv}")

    shared_df = pd.DataFrame(shared_counts, index=regions, columns=regions)
    shared_csv = OUT_DIR / "ms_shared_word_counts.csv"
    shared_df.to_csv(shared_csv)
    print(f"  Saved: {shared_csv}")

    # Hierarchical clustering + dendrogram
    sq = squareform(dist_matrix, checks=False)
    Z = linkage(sq, method="average")
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(Z, labels=regions, leaf_rotation=90, ax=ax, color_threshold=0.45)
    ax.set_title("Manzini-Savoia phonetic-lexical distance (regions)")
    ax.set_ylabel("normalised Levenshtein")
    plt.tight_layout()
    out_png = OUT_DIR / "ms_dendrogram.png"
    plt.savefig(out_png, dpi=120)
    print(f"  Saved: {out_png}")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"Distance matrix saved with {n} regions.")
    print(f"Average pairwise distance: {np.nanmean(sq):.3f}")
    print(f"Average shared words: {np.mean([shared_counts[i, j] for i in range(n) for j in range(i+1, n)]):.0f}")


if __name__ == "__main__":
    main()
