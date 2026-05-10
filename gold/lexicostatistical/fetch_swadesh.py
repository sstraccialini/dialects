"""
Download and merge Swadesh-207 lists from Wiktionary for our 17 varieties.

Sources (all from English Wiktionary):
    - Appendix:Swadesh_lists_for_Italian_languages    italo-romance + ita
        Tables: 0 = main comparative.
        Yields: ita, lmo, fur, vec, scn, lij, sc.
    - Appendix:Swadesh_lists_for_Romance_languages    multiple Romance tables
        Yields:
            fra, spa, cat, por          (table 0)
            oci                         (table 1, "Lengadocian Occitan")
    - Appendix:Swadesh_lists_for_Germanic_languages   modern English + German
        Yields: deu, eng.
    - Appendix:Swadesh_lists_for_Slavic_languages     slovene + serbo-croatian
        Yields: slv, hrv (Serbo-Croatian column).
    - Appendix:Hungarian_Swadesh_list                 single-language appendix
        Yields: hun.

The 207 concepts are aligned by Swadesh row number across all appendices,
since they follow the same canonical Swadesh-207 ordering.

Output: ``wordlist_swadesh207.csv`` with columns
    n, concept_en, ita, fra, spa, cat, por, oci,
                   deu, eng, slv, hrv, hun,
                   fur, lij, lmo, sc, scn, vec

Run:
    python -m gold.lexicostatistical.fetch_swadesh \
        --out gold/lexicostatistical/wordlists/wordlist_swadesh207.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple


URLS = {
    "italian":   "https://en.wiktionary.org/wiki/Appendix:Swadesh_lists_for_Italian_languages",
    "romance":   "https://en.wiktionary.org/wiki/Appendix:Swadesh_lists_for_Romance_languages",
    "germanic":  "https://en.wiktionary.org/wiki/Appendix:Swadesh_lists_for_Germanic_languages",
    "slavic":    "https://en.wiktionary.org/wiki/Appendix:Swadesh_lists_for_Slavic_languages",
    "hungarian": "https://en.wiktionary.org/wiki/Appendix:Hungarian_Swadesh_list",
}

# Final column order for the merged CSV.
COL_ORDER = [
    "n", "concept_en",
    # standards / external
    "ita", "fra", "spa", "cat", "por", "oci",
    "deu", "eng", "slv", "hrv", "hun",
    # italo-romance dialects
    "fur", "lij", "lmo", "sc", "scn", "vec",
]


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "ltp-lex-matrix/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8")


def _clean(text: str) -> str:
    text = re.sub(r"<sup[^>]*>.*?</sup>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&nbsp;", " ").replace("&#160;", " ")
    text = text.replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'")
    return text.strip()


def _parse_table(html: str, table_index: int = 0) -> Tuple[List[str], List[List[str]]]:
    tables = re.findall(r"<table[^>]*>.*?</table>", html, re.DOTALL)
    if table_index >= len(tables):
        return [], []
    tbl = tables[table_index]
    trs = re.findall(r"<tr[^>]*>(.*?)</tr>", tbl, re.DOTALL)
    if not trs:
        return [], []
    headers = [_clean(h) for h in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", trs[0], re.DOTALL)]
    rows = []
    for tr in trs[1:]:
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, re.DOTALL)
        rows.append([_clean(c) for c in cells])
    return headers, rows


def _column_index(headers: List[str], substr: str) -> int:
    s = substr.lower()
    for i, h in enumerate(headers):
        if s in h.lower():
            return i
    return -1


# --------------------------------------------------------------------------- #
# Appendix-specific parsers
# --------------------------------------------------------------------------- #

def parse_italian_languages(html: str) -> Dict[int, Dict[str, str]]:
    """Returns {n: {variety: form}}.  Pulls ita, lmo, fur, vec, scn, lij, sc."""
    headers, rows = _parse_table(html, 0)
    cols = {
        "concept_en": _column_index(headers, "english"),
        "ita": _column_index(headers, "italian"),
        "lmo": _column_index(headers, "lombard"),
        "fur": _column_index(headers, "friulian"),
        "vec": _column_index(headers, "venetan"),
        "scn": _column_index(headers, "sicilian"),
        "lij": _column_index(headers, "ligurian"),
        "sc":  _column_index(headers, "sardinian"),
    }
    print(f"  italian-languages columns: {cols}")
    out: Dict[int, Dict[str, str]] = {}
    for row in rows:
        if len(row) < max(cols.values()) + 1:
            continue
        try:
            n = int(row[0])
        except ValueError:
            continue
        rec = {"concept_en": row[cols["concept_en"]]}
        for code in ("ita", "lmo", "fur", "vec", "scn", "lij", "sc"):
            rec[code] = row[cols[code]] if cols[code] >= 0 else ""
        out[n] = rec
    return out


def parse_romance(html: str) -> Dict[int, Dict[str, str]]:
    """Pulls from multiple Romance tables:
        Table 0: fra, spa, cat, por, ita (reference)
        Table 1: oci   (Lengadocian Occitan)
    """
    out: Dict[int, Dict[str, str]] = {}

    # ----- Table 0: main Romance comparison -----
    headers0, rows0 = _parse_table(html, 0)
    cols0 = {
        "fra": _column_index(headers0, "french"),
        "spa": _column_index(headers0, "spanish"),
        "cat": _column_index(headers0, "catalan"),
        "por": _column_index(headers0, "portuguese"),
    }
    print(f"  romance table-0 columns: {cols0}")
    for n, row in enumerate(rows0, start=1):
        if not row:
            continue
        rec = out.setdefault(n, {})
        for code, idx in cols0.items():
            if 0 <= idx < len(row):
                rec[code] = row[idx]
            else:
                rec.setdefault(code, "")

    # ----- Table 1: Occitan (Lengadocian) -----
    headers1, rows1 = _parse_table(html, 1)
    # Match both 'occitan' and 'lengadocian' to find the column
    oci_idx = _column_index(headers1, "lengadocian")
    if oci_idx < 0:
        oci_idx = _column_index(headers1, "occitan")
    print(f"  romance table-1 occitan column: {oci_idx}")
    for n, row in enumerate(rows1, start=1):
        if not row or oci_idx < 0:
            continue
        rec = out.setdefault(n, {})
        rec["oci"] = row[oci_idx] if oci_idx < len(row) else ""

    return out


def parse_germanic(html: str) -> Dict[int, Dict[str, str]]:
    headers, rows = _parse_table(html, 0)
    cols = {
        "deu": _column_index(headers, "german"),
        "eng": _column_index(headers, "english"),
    }
    if cols["eng"] >= 0 and "old" in headers[cols["eng"]].lower():
        cols["eng"] = -1
    print(f"  germanic columns: {cols}")
    out: Dict[int, Dict[str, str]] = {}
    for row in rows:
        if len(row) < 2:
            continue
        try:
            n = int(row[0])
        except ValueError:
            continue
        rec = {}
        for code in ("deu", "eng"):
            rec[code] = row[cols[code]] if cols[code] >= 0 and cols[code] < len(row) else ""
        out[n] = rec
    return out


def parse_slavic(html: str) -> Dict[int, Dict[str, str]]:
    headers, rows = _parse_table(html, 0)
    cols = {
        "slv": _column_index(headers, "slovene"),
        # Serbo-Croatian (umbrella for hrv/srp/bos) — treat as Croatian for our purposes
        "hrv": _column_index(headers, "serbo-croatian"),
    }
    print(f"  slavic columns: {cols}")
    out: Dict[int, Dict[str, str]] = {}
    for row in rows:
        if len(row) < 2:
            continue
        try:
            n = int(row[0])
        except ValueError:
            continue
        rec = {}
        for code in ("slv", "hrv"):
            rec[code] = row[cols[code]] if cols[code] >= 0 and cols[code] < len(row) else ""
        out[n] = rec
    return out


def parse_single_language_appendix(html: str, code: str, header_substr: str
                                   ) -> Dict[int, Dict[str, str]]:
    """Single-language Swadesh appendices (Albanian, Hungarian, ...).

    Layout: a single table whose first column is № and second column is
    English; subsequent columns are the language form(s).  We grab the
    first language column whose header contains ``header_substr``.
    """
    headers, rows = _parse_table(html, 0)
    if not headers or not rows:
        print(f"  {code}: NO table found")
        return {}
    lang_idx = _column_index(headers, header_substr)
    if lang_idx < 0:
        # fallback: take the third column if exists (after № + English)
        lang_idx = 2 if len(headers) > 2 else -1
    print(f"  {code}: header '{header_substr}' resolved to column {lang_idx}")
    out: Dict[int, Dict[str, str]] = {}
    for row in rows:
        if len(row) < 2:
            continue
        try:
            n = int(row[0])
        except ValueError:
            continue
        if lang_idx < 0 or lang_idx >= len(row):
            continue
        out[n] = {code: row[lang_idx]}
    return out


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #

def merge_all() -> Tuple[List[Dict[str, str]], List[str]]:
    print("[1/5] downloading Italian-languages appendix")
    italian = parse_italian_languages(fetch(URLS["italian"]))
    print(f"   parsed {len(italian)} Swadesh rows")

    print("[2/5] downloading Romance appendix")
    romance = parse_romance(fetch(URLS["romance"]))
    print(f"   parsed {len(romance)} rows")

    print("[3/5] downloading Germanic appendix")
    germanic = parse_germanic(fetch(URLS["germanic"]))
    print(f"   parsed {len(germanic)} rows")

    print("[4/5] downloading Slavic appendix")
    slavic = parse_slavic(fetch(URLS["slavic"]))
    print(f"   parsed {len(slavic)} rows")

    print("[5/5] downloading Hungarian appendix")
    try:
        hungarian = parse_single_language_appendix(
            fetch(URLS["hungarian"]), "hun", "hungarian")
    except Exception as exc:
        print(f"   warning: Hungarian appendix failed ({exc}); hun will be empty")
        hungarian = {}
    print(f"   parsed {len(hungarian)} rows for hun")

    rows_out: List[Dict[str, str]] = []
    all_n = (set(italian) | set(romance) | set(germanic) | set(slavic)
             | set(hungarian))
    for n in sorted(all_n):
        rec: Dict[str, str] = {"n": str(n)}
        rec["concept_en"] = italian.get(n, {}).get("concept_en", "")

        # italo-romance dialects + ita
        for code in ("ita", "lmo", "fur", "vec", "scn", "lij", "sc"):
            rec[code] = italian.get(n, {}).get(code, "")
        # other Romance
        for code in ("fra", "spa", "cat", "por", "oci"):
            rec[code] = romance.get(n, {}).get(code, "")
        # Germanic
        for code in ("deu", "eng"):
            rec[code] = germanic.get(n, {}).get(code, "")
        if not rec.get("eng"):
            rec["eng"] = rec["concept_en"]
        # Slavic
        for code in ("slv", "hrv"):
            rec[code] = slavic.get(n, {}).get(code, "")
        # Hungarian (single-language)
        rec["hun"] = hungarian.get(n, {}).get("hun", "")

        rows_out.append(rec)

    return rows_out, COL_ORDER


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    rows, cols = merge_all()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow({c: r.get(c, "") for c in cols})

    print(f"\nWrote {len(rows)} rows × {len(cols)} columns to {args.out}")
    print("\nNon-empty cells per language:")
    for code in cols[2:]:
        n_filled = sum(1 for r in rows if r.get(code))
        print(f"  {code:<5}  {n_filled}/{len(rows)} rows filled")

    needed = [c for c in cols if c not in ("n", "concept_en")]
    common = [r for r in rows if all(r.get(c) for c in needed)]
    print(f"\nRows where ALL {len(needed)} varieties have a non-empty form: "
          f"{len(common)}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
