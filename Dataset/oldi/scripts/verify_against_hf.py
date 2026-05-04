"""
Verify that the local OLDI seed parquets in `Dataset/oldi/not_normalized/`
match the upstream `openlanguagedata/oldi_seed` HuggingFace dataset.

Also verifies FLORES+ local files against `openlanguagedata/flores_plus`.

For each language present locally, downloads a fresh copy from HF and
compares: row count, text-column MD5, first/last sentence. Skips the 3
MT-derived OLDI files (deu/cat/slv) since those aren't on HF.

Output: prints a per-language verdict and a final summary. Read-only —
does NOT touch anything in `Dataset/oldi/` or `Dataset/flores/`.

Run with:
    python Dataset/oldi/scripts/verify_against_hf.py
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import load_dataset
from huggingface_hub import get_token, login


REPO_ROOT = Path(__file__).resolve().parents[3]
OLDI_LOCAL = REPO_ROOT / "Dataset" / "oldi" / "not_normalized"
FLORES_LOCAL = REPO_ROOT / "Dataset" / "flores" / "not_normalized"

OLDI_HF_REPO = "openlanguagedata/oldi_seed"
FLORES_HF_REPO = "openlanguagedata/flores_plus"

# OLDI: ISO-639-3 + script. Skip deu/cat/slv — those were MT-translated
# by us, NOT in the official OLDI release.
OLDI_LANGS = [
    "ace_Arab", "ace_Latn", "ary_Arab",
    "eng_Latn", "ita_Latn", "fra_Latn", "spa_Latn",
    "fur_Latn", "lij_Latn", "lmo_Latn", "scn_Latn", "srd_Latn", "vec_Latn",
]

# FLORES+: ISO + script -> local Italian-name slug.
FLORES_LANGS = [
    ("eng_Latn", "inglese"),
    ("ita_Latn", "italiano"),
    ("fra_Latn", "francese"),
    ("spa_Latn", "spagnolo"),
    ("deu_Latn", "tedesco"),
    ("cat_Latn", "catalano"),
    ("slv_Latn", "sloveno"),
    ("vec_Latn", "veneto"),
    ("scn_Latn", "siciliano"),
    ("lmo_Latn", "lombardo"),
    ("srd_Latn", "sardo"),
    ("lij_Latn", "ligure"),
    ("fur_Latn", "friulano"),
    ("lld_Latn", "ladino"),
]


def md5_of_texts(texts) -> str:
    h = hashlib.md5()
    for s in texts:
        h.update(s.encode("utf-8", errors="replace"))
        h.update(b"\n")
    return h.hexdigest()


def authenticate():
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        sys.exit("No HF token found. Run `hf auth login` or export HF_TOKEN.")
    login(token=token, add_to_git_credential=False)


def verify_oldi_one(lang_code: str) -> dict:
    """Compare local Dataset/oldi/not_normalized/<code>_Latn.parquet to HF."""
    out = {"lang": lang_code, "status": "?", "detail": ""}
    local_path = OLDI_LOCAL / f"{lang_code}.parquet"
    if not local_path.exists():
        out["status"] = "MISSING_LOCAL"
        out["detail"] = f"{local_path} not present"
        return out
    local = pd.read_parquet(local_path)

    try:
        # OLDI on HF is exposed as a dataset with one config per language.
        hf = load_dataset(OLDI_HF_REPO, lang_code, split="train",
                          trust_remote_code=False)
    except Exception as e:
        out["status"] = "HF_ERROR"
        out["detail"] = f"{type(e).__name__}: {e}"
        return out

    hf_texts = [row["text"] for row in hf]
    local_texts = local["text"].tolist()

    if len(hf_texts) != len(local_texts):
        out["status"] = "MISMATCH_LEN"
        out["detail"] = f"local={len(local_texts):,}  HF={len(hf_texts):,}"
        return out

    md5_local = md5_of_texts(local_texts)
    md5_hf = md5_of_texts(hf_texts)
    if md5_local != md5_hf:
        # Find first differing row
        first_diff = None
        for i, (a, b) in enumerate(zip(local_texts, hf_texts)):
            if a != b:
                first_diff = (i, a, b)
                break
        out["status"] = "MISMATCH_TEXT"
        out["detail"] = (f"md5 local={md5_local[:10]}  HF={md5_hf[:10]}"
                         + (f"  first diff at id={first_diff[0]}" if first_diff else ""))
        if first_diff:
            out["first_diff"] = first_diff
        return out

    out["status"] = "OK"
    out["detail"] = f"{len(local_texts):,} rows  md5={md5_local[:10]}"
    return out


def verify_flores_one(hf_code: str, slug: str) -> dict:
    """Compare local Dataset/flores/not_normalized/<slug>.txt to HF."""
    out = {"lang": f"{hf_code} ({slug})", "status": "?", "detail": ""}
    local_path = FLORES_LOCAL / f"{slug}.txt"
    if not local_path.exists():
        out["status"] = "MISSING_LOCAL"
        out["detail"] = f"{local_path} not present"
        return out
    with local_path.open(encoding="utf-8") as f:
        local_texts = [line.rstrip("\n") for line in f if line.strip()]

    try:
        # Local file order is dev (997) + devtest (1012) per download_flores.py.
        dev = load_dataset(FLORES_HF_REPO, hf_code, split="dev")
        dvt = load_dataset(FLORES_HF_REPO, hf_code, split="devtest")
    except Exception as e:
        out["status"] = "HF_ERROR"
        out["detail"] = f"{type(e).__name__}: {e}"
        return out

    # Replicate the cleaning done by download_flores.py: strip + collapse newlines.
    def clean(s):
        return s.strip().replace("\n", " ").replace("\r", " ")

    hf_texts = ([clean(r["text"]) for r in dev]
                + [clean(r["text"]) for r in dvt])

    if len(hf_texts) != len(local_texts):
        out["status"] = "MISMATCH_LEN"
        out["detail"] = f"local={len(local_texts):,}  HF={len(hf_texts):,}"
        return out

    md5_local = md5_of_texts(local_texts)
    md5_hf = md5_of_texts(hf_texts)
    if md5_local != md5_hf:
        first_diff = None
        for i, (a, b) in enumerate(zip(local_texts, hf_texts)):
            if a != b:
                first_diff = (i, a, b)
                break
        out["status"] = "MISMATCH_TEXT"
        out["detail"] = (f"md5 local={md5_local[:10]}  HF={md5_hf[:10]}"
                         + (f"  first diff at line={first_diff[0]}" if first_diff else ""))
        if first_diff:
            out["first_diff"] = first_diff
        return out

    out["status"] = "OK"
    out["detail"] = f"{len(local_texts):,} lines  md5={md5_local[:10]}"
    return out


def print_results(title: str, rows):
    print(f"\n=== {title} ===")
    width = max(len(r["lang"]) for r in rows)
    for r in rows:
        marker = {"OK": "✓", "MISSING_LOCAL": "?", "HF_ERROR": "!",
                  "MISMATCH_LEN": "✗", "MISMATCH_TEXT": "✗"}.get(r["status"], "?")
        print(f"  {marker} {r['lang']:<{width}}  {r['status']:<14}  {r['detail']}")
        if "first_diff" in r:
            i, a, b = r["first_diff"]
            print(f"      [{i}] LOCAL: {a[:100]}")
            print(f"      [{i}] HF:    {b[:100]}")


def main():
    authenticate()

    print("Verifying OLDI seed local against openlanguagedata/oldi_seed ...")
    oldi_rows = []
    for code in OLDI_LANGS:
        print(f"  checking {code} ...")
        oldi_rows.append(verify_oldi_one(code))
    print_results("OLDI seed", oldi_rows)

    print("\nVerifying FLORES+ local against openlanguagedata/flores_plus ...")
    flores_rows = []
    for hf_code, slug in FLORES_LANGS:
        print(f"  checking {hf_code} ({slug}) ...")
        flores_rows.append(verify_flores_one(hf_code, slug))
    print_results("FLORES+", flores_rows)

    print("\n=== Summary ===")
    n_oldi_ok = sum(1 for r in oldi_rows if r["status"] == "OK")
    n_flores_ok = sum(1 for r in flores_rows if r["status"] == "OK")
    print(f"  OLDI:    {n_oldi_ok}/{len(oldi_rows)} match HF")
    print(f"  FLORES+: {n_flores_ok}/{len(flores_rows)} match HF")
    overall = (n_oldi_ok == len(oldi_rows) and n_flores_ok == len(flores_rows))
    print(f"  Overall: {'PASS' if overall else 'FAIL'}")


if __name__ == "__main__":
    main()
