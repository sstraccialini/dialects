"""
Compute phonetic-lexical distances using NorthEuraLex.

NorthEuraLex provides ~1016 standardized concepts in IPA across ~107
languages of Eurasia, including all the major Romance languages we need
plus Germanic and Slavic comparators. Compared to ASJP (40 words),
NorthEuraLex gives ~25× more concepts per language for more robust
distance estimates.

Pipeline:
1. Download NorthEuraLex 0.9 TSV
2. Filter to target languages (italian standard + foreign)
3. Build language -> concept -> IPA map
4. Pairwise normalised Levenshtein on shared concepts
5. Save distance matrix + dendrogram

Run from repo root:
    python experiments/manzini_savoia/src/northeuralex_distance.py
"""
from __future__ import annotations

import csv
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
NEL_DIR = REPO_ROOT / "manzini_savoia" / "northeuralex_data"
NEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = REPO_ROOT / "manzini_savoia" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NEL_BASE_URL = "https://raw.githubusercontent.com/lexibank/northeuralex/master/cldf"
NEL_FILES = ["forms.csv", "languages.csv", "parameters.csv"]

# NorthEuraLex uses ISO 639-3 / Glottolog codes. Map our project labels
# to the actual codes used in the TSV. We fill these defensively; if a
# code isn't found we skip it.
TARGET_LANGUAGES = {
    "italian":   ["ita"],
    "spanish":   ["spa"],
    "catalan":   ["cat"],
    "french":    ["fra"],
    "portuguese": ["por"],
    "romanian":  ["ron", "rum"],
    "english":   ["eng"],
    "german":    ["deu"],
    "greek":     ["ell", "gre"],
    "slovenian": ["slv"],
    # Italian dialects: NorthEuraLex coverage is limited; we try anyway.
    "sicilian":  ["scn"],
    "sardinian": ["srd", "sro"],
    "venetian":  ["vec"],
    "friulian":  ["fur"],
    "lombard":   ["lmo"],
    "ladin":     ["lld"],
    "ligurian":  ["lij"],
    "neapolitan": ["nap"],
    # Slavic comparators
    "bulgarian": ["bul"],
    "czech":     ["ces", "cze"],
    "polish":    ["pol"],
    # Plus arabic (probably not in NorthEuraLex; it's not Eurasian)
}


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return max(la, lb)
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb))
        prev = curr
    return prev[-1]


def normalised_levenshtein(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    return levenshtein(a, b) / max(len(a), len(b))


def download_neL():
    for fname in NEL_FILES:
        target = NEL_DIR / fname
        if target.exists() and target.stat().st_size > 10_000:
            print(f"  {fname}: already present ({target.stat().st_size/1024/1024:.1f} MB)")
            continue
        url = f"{NEL_BASE_URL}/{fname}"
        print(f"  downloading {url} ...")
        urllib.request.urlretrieve(url, target)
        print(f"    -> {target} ({target.stat().st_size/1024/1024:.1f} MB)")


def load_neL() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    forms = pd.read_csv(NEL_DIR / "forms.csv")
    langs = pd.read_csv(NEL_DIR / "languages.csv")
    params = pd.read_csv(NEL_DIR / "parameters.csv")
    print(f"  forms: {len(forms):,}")
    print(f"  languages: {len(langs):,}")
    print(f"  parameters (concepts): {len(params):,}")
    print(f"  forms columns: {list(forms.columns)}")
    print(f"  languages columns: {list(langs.columns)}")
    return forms, langs, params


# find_language_codes is no longer needed; logic inlined into main() using
# CLDF format (languages.csv has ISO639P3 codes for lookup).


def main():
    print("=== NorthEuraLex phonetic-lexical distance ===")
    print()

    print("Step 1: download")
    download_neL()
    print()

    print("Step 2: load")
    forms, langs, params = load_neL()
    print()

    print("Step 3: find languages (using languages.csv)")
    # CLDF: forms references Language_ID (= ID in languages.csv)
    # Look up by Glottocode or Iso639P3code (= ISO 639-3)
    iso_col = None
    for c in ["ISO639P3code", "Iso_Code", "Glottocode", "Iso639P3code"]:
        if c in langs.columns:
            iso_col = c
            break
    name_col = "Name" if "Name" in langs.columns else "name"
    id_col = "ID" if "ID" in langs.columns else "id"
    print(f"  ISO column: {iso_col}, ID column: {id_col}, Name column: {name_col}")
    print()

    lang_codes: dict[str, str] = {}  # label -> CLDF Language_ID
    for label, codes in TARGET_LANGUAGES.items():
        for code in codes:
            matches = langs[langs[iso_col].astype(str).str.lower() == code.lower()] if iso_col else pd.DataFrame()
            if not matches.empty:
                row = matches.iloc[0]
                lang_codes[label] = str(row[id_col])
                print(f"    {label:12s} -> {row[name_col]} (id={row[id_col]}, iso={code})")
                break
        else:
            print(f"    {label:12s} NOT FOUND (tried {codes})")
    if not lang_codes:
        print("  no languages matched, aborting")
        return
    print()

    # CLDF form column names
    lang_col = "Language_ID" if "Language_ID" in forms.columns else "language_id"
    concept_col = "Parameter_ID" if "Parameter_ID" in forms.columns else "parameter_id"
    form_col = "Form" if "Form" in forms.columns else "form"

    code_to_label = {v: k for k, v in lang_codes.items()}
    sub = forms[forms[lang_col].astype(str).isin(lang_codes.values())].copy()
    sub["label"] = sub[lang_col].astype(str).map(code_to_label)

    # Build label -> concept -> IPA (first form per concept per language)
    lang_word: dict[str, dict[str, str]] = defaultdict(dict)
    for _, row in sub.iterrows():
        label = row["label"]
        concept = str(row[concept_col])
        form = str(row[form_col]).strip()
        if not form or form == "nan":
            continue
        if concept not in lang_word[label]:
            lang_word[label][concept] = form

    print()
    print(f"Step 4: per-language coverage")
    for lab in sorted(lang_word):
        print(f"    {lab:12s}  {len(lang_word[lab])} concepts")

    labels = sorted(lang_word)
    n = len(labels)

    print()
    print("Step 5: pairwise Levenshtein")
    dist = np.zeros((n, n))
    shared = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            la, lb = labels[i], labels[j]
            common = set(lang_word[la]) & set(lang_word[lb])
            if not common:
                d = float("nan")
            else:
                d = float(np.mean([
                    normalised_levenshtein(lang_word[la][c], lang_word[lb][c])
                    for c in common
                ]))
            dist[i, j] = d
            dist[j, i] = d
            shared[i, j] = len(common)
            shared[j, i] = len(common)

    df_d = pd.DataFrame(dist, index=labels, columns=labels)
    out_csv = OUT_DIR / "neL_phonetic_distance.csv"
    df_d.to_csv(out_csv, float_format="%.4f")
    print(f"  Saved: {out_csv}")

    sh_df = pd.DataFrame(shared, index=labels, columns=labels)
    sh_csv = OUT_DIR / "neL_shared_word_counts.csv"
    sh_df.to_csv(sh_csv)
    print(f"  Saved: {sh_csv}")

    sq = squareform(dist, checks=False)
    Z = linkage(sq, method="average")
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(Z, labels=labels, leaf_rotation=90, ax=ax)
    ax.set_title("NorthEuraLex phonetic-lexical distance")
    ax.set_ylabel("normalised Levenshtein")
    plt.tight_layout()
    out_png = OUT_DIR / "neL_dendrogram.png"
    plt.savefig(out_png, dpi=120)
    print(f"  Saved: {out_png}")

    print()
    print("=== SUMMARY ===")
    print(f"Languages: {n}")
    print(f"Average pairwise distance: {np.nanmean(sq):.3f}")
    print(f"Average shared concepts: {np.mean([shared[i,j] for i in range(n) for j in range(i+1,n)]):.0f}")


if __name__ == "__main__":
    main()
