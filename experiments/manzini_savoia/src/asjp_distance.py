"""
Compute phonetic-lexical distances between languages using ASJP database.

ASJP provides ~40 standard Swadesh-like words in simplified IPA for ~5000
languages, including Italian dialects and major Romance/non-Romance
languages. We use this to extend the M-S analysis to non-italian languages.

Pipeline:
1. Download/load ASJP CLDF dataset
2. Filter to varieties of interest (italian dialects + FLORES foreign langs)
3. Build word-form matrices
4. Pairwise Levenshtein distance on shared words
5. Output: distance matrix + dendrogram

Run from repo root:
    python experiments/manzini_savoia/src/asjp_distance.py
"""
from __future__ import annotations

import csv
import io
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASJP_DIR = REPO_ROOT / "manzini_savoia" / "asjp_data"
ASJP_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = REPO_ROOT / "manzini_savoia" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ASJP CLDF data: download individual CSV files from lexibank/asjp on GitHub.
ASJP_BASE_URL = "https://raw.githubusercontent.com/lexibank/asjp/master/cldf"
ASJP_FILES = ["forms.csv", "languages.csv", "parameters.csv"]

# Languages we want from ASJP. ASJP uses its own codes (not always ISO).
# Possible names to look for (case-insensitive, partial match).
# ASJP names (case-sensitive substrings). LADIN and VENETIAN are NOT in ASJP.
TARGET_LANGUAGES = {
    # Italian dialects (only those actually in ASJP)
    "italian":      ["^italian$"],
    "sicilian":     ["^sicilian"],
    "lombard":      ["^lombard"],
    "ligurian":     ["^ligurian"],
    "sardinian":    ["^sardinian$"],
    "gallurese":    ["^gallurese"],          # northern sardinian variety
    "neapolitan":   ["^neapolitan"],
    "friulian":     ["^friulian"],
    "piedmontese":  ["^piemontese", "^piedmontese"],
    "emilian":      ["^emiliano", "^emilian"],
    # Foreign Romance
    "spanish":      ["^spanish$"],
    "catalan":      ["^catalan$"],
    "french":       ["^french$"],
    "portuguese":   ["^portuguese$"],
    "romanian":     ["^romanian$"],
    "occitan":      ["^occitan"],
    # Non-Romance
    "english":      ["^english$"],
    "german":       ["^standard_german$"],
    "greek":        ["^modern_greek$", "^greek$"],
    "arabic":       ["^classical_arabic$"],
    "slovenian":    ["^slovenian$"],
}


def download_asjp() -> Path:
    """Download CLDF CSV files from lexibank/asjp on GitHub."""
    for fname in ASJP_FILES:
        target = ASJP_DIR / fname
        if target.exists() and target.stat().st_size > 100_000:
            print(f"  {fname}: already present ({target.stat().st_size/1024/1024:.1f} MB)")
            continue
        url = f"{ASJP_BASE_URL}/{fname}"
        print(f"  downloading {url} ...")
        urllib.request.urlretrieve(url, target)
        print(f"    -> {target} ({target.stat().st_size/1024/1024:.1f} MB)")
    return ASJP_DIR


def load_asjp(asjp_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load ASJP forms, languages, parameters from individual CSVs."""
    forms = pd.read_csv(asjp_dir / "forms.csv")
    langs = pd.read_csv(asjp_dir / "languages.csv")
    params = pd.read_csv(asjp_dir / "parameters.csv")
    print(f"  forms: {len(forms):,}")
    print(f"  languages: {len(langs):,}")
    print(f"  parameters (concepts): {len(params):,}")
    return forms, langs, params


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


def find_languages(langs_df: pd.DataFrame, targets: dict[str, list[str]]) -> dict[str, str]:
    """Find ASJP language IDs matching our target patterns (regex anchored)."""
    import re as _re
    out: dict[str, str] = {}
    name_col = "Name" if "Name" in langs_df.columns else "name"
    id_col = "ID" if "ID" in langs_df.columns else "id"

    for label, patterns in targets.items():
        best_match = None
        for pat in patterns:
            rx = _re.compile(pat, _re.IGNORECASE)
            matches = langs_df[langs_df[name_col].str.lower().apply(lambda s: bool(rx.search(s)) if isinstance(s, str) else False)]
            if not matches.empty:
                matches = matches.copy()
                matches["score"] = matches[name_col].str.len()
                best_match = matches.sort_values("score").iloc[0]
                break
        if best_match is not None:
            out[label] = str(best_match[id_col])
            print(f"    {label:15s}  -> {best_match[name_col]} (id={best_match[id_col]})")
        else:
            print(f"    {label:15s}  NOT FOUND")
    return out


def build_lang_word_matrix(forms_df, lang_ids: dict[str, str]) -> dict[str, dict[str, str]]:
    """For each language label, return dict[concept_id] -> form_string."""
    lang_id_col = "Language_ID" if "Language_ID" in forms_df.columns else "language_id"
    param_id_col = "Parameter_ID" if "Parameter_ID" in forms_df.columns else "parameter_id"
    form_col = "Form" if "Form" in forms_df.columns else "form"

    asjp_id_to_label = {v: k for k, v in lang_ids.items()}
    out: dict[str, dict[str, str]] = defaultdict(dict)

    sub = forms_df[forms_df[lang_id_col].astype(str).isin(lang_ids.values())]
    print(f"  forms matched: {len(sub):,}")

    for _, row in sub.iterrows():
        label = asjp_id_to_label[str(row[lang_id_col])]
        concept = str(row[param_id_col])
        form = str(row[form_col])
        # ASJP often has multiple variants per concept (synonyms). Take first.
        if concept not in out[label]:
            out[label][concept] = form

    return dict(out)


def main():
    print("=== ASJP-based phonetic-lexical distance ===")
    print()

    print("Step 1: download ASJP")
    zip_path = download_asjp()
    print()

    print("Step 2: parse ASJP")
    forms, langs, params = load_asjp(zip_path)
    print()

    print("Step 3: find target languages")
    lang_ids = find_languages(langs, TARGET_LANGUAGES)
    if not lang_ids:
        print("  no languages matched, aborting")
        return
    print()

    print("Step 4: build word matrices")
    lang_word = build_lang_word_matrix(forms, lang_ids)
    print(f"  languages with data: {len(lang_word)}")
    for label, words in lang_word.items():
        print(f"    {label:15s}  {len(words)} concepts")
    print()

    labels = sorted(lang_word.keys())
    n = len(labels)

    print("Step 5: pairwise Levenshtein")
    dist_matrix = np.zeros((n, n))
    shared_counts = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            la, lb = labels[i], labels[j]
            shared = set(lang_word[la]) & set(lang_word[lb])
            if not shared:
                d = float("nan")
            else:
                d = float(np.mean([
                    normalised_levenshtein(lang_word[la][c], lang_word[lb][c])
                    for c in shared
                ]))
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
            shared_counts[i, j] = len(shared)
            shared_counts[j, i] = len(shared)
    print()

    df = pd.DataFrame(dist_matrix, index=labels, columns=labels)
    out_csv = OUT_DIR / "asjp_phonetic_distance.csv"
    df.to_csv(out_csv, float_format="%.4f")
    print(f"  Saved: {out_csv}")

    sh_df = pd.DataFrame(shared_counts, index=labels, columns=labels)
    sh_csv = OUT_DIR / "asjp_shared_word_counts.csv"
    sh_df.to_csv(sh_csv)
    print(f"  Saved: {sh_csv}")

    sq = squareform(dist_matrix, checks=False)
    Z = linkage(sq, method="average")
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(Z, labels=labels, leaf_rotation=90, ax=ax)
    ax.set_title("ASJP phonetic-lexical distance (italian dialects + foreign langs)")
    ax.set_ylabel("normalised Levenshtein")
    plt.tight_layout()
    out_png = OUT_DIR / "asjp_dendrogram.png"
    plt.savefig(out_png, dpi=120)
    print(f"  Saved: {out_png}")

    print()
    print("=== SUMMARY ===")
    print(f"Languages: {n}")
    print(f"Average pairwise distance: {np.nanmean(sq):.3f}")


if __name__ == "__main__":
    main()
