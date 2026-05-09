"""
Download and merge Swadesh-207 lists from Wiktionary for our 13 varieties.

Sources (all from English Wiktionary):
    - Appendix:Swadesh_lists_for_Italian_languages    (ita, lmo, fur, vec, scn, lij, sc + nap, pms)
    - Appendix:Swadesh_lists_for_Romance_languages    (table 0: fra, spa, cat + ita reference)
    - Appendix:Swadesh_lists_for_Germanic_languages   (deu, eng + others)
    - Appendix:Swadesh_lists_for_Slavic_languages     (slv + others)

The 207 concepts are aligned by Swadesh row number across all appendices,
since they follow the same canonical Swadesh-207 ordering.

Output: ``wordlist_swadesh207.csv`` with columns
    n, concept_en, ita, fra, spa, cat, deu, slv, eng, fur, lij, lmo, sc, scn, vec
    + raw_<code> = original Wiktionary cell (with all alternate forms)

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
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Tuple


URLS = {
    "italian": "https://en.wiktionary.org/wiki/Appendix:Swadesh_lists_for_Italian_languages",
    "romance": "https://en.wiktionary.org/wiki/Appendix:Swadesh_lists_for_Romance_languages",
    "germanic": "https://en.wiktionary.org/wiki/Appendix:Swadesh_lists_for_Germanic_languages",
    "slavic": "https://en.wiktionary.org/wiki/Appendix:Swadesh_lists_for_Slavic_languages",
}


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "edoardo-lex-matrix/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8")


def _clean(text: str) -> str:
    """Strip HTML, NBSP, surrounding whitespace."""
    text = re.sub(r"<sup[^>]*>.*?</sup>", "", text, flags=re.DOTALL)  # footnotes
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&nbsp;", " ").replace("&#160;", " ")
    text = text.replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'")
    return text.strip()


def _parse_table(html: str, table_index: int = 0) -> List[Tuple[List[str], List[List[str]]]]:
    """Return (headers, rows) for the table_index-th <table> in html.

    The Romance appendix mixes <td>/<th> in its header row (the № column
    is a <td>, the rest are <th>).  We treat any <th>-or-<td> in the first
    <tr> as a header cell so column indices align with data rows.
    """
    tables = re.findall(r"<table[^>]*>.*?</table>", html, re.DOTALL)
    if table_index >= len(tables):
        return [], []
    tbl = tables[table_index]
    trs = re.findall(r"<tr[^>]*>(.*?)</tr>", tbl, re.DOTALL)
    if not trs:
        return [], []
    hdr_html = trs[0]
    headers = [_clean(h) for h in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", hdr_html, re.DOTALL)]
    rows = []
    for tr in trs[1:]:
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, re.DOTALL)
        cells = [_clean(c) for c in cells]
        rows.append(cells)
    return headers, rows


def _strip_concept_label(s: str) -> str:
    """Header cells look like 'Italianitalianoedit (207)' — keep the language code."""
    return s


def _column_index(headers: List[str], substr: str, case_sensitive: bool = False) -> int:
    if not case_sensitive:
        substr_lower = substr.lower()
        for i, h in enumerate(headers):
            if substr_lower in h.lower():
                return i
    else:
        for i, h in enumerate(headers):
            if substr in h:
                return i
    return -1


def parse_italian_languages(html: str) -> Dict[int, Dict[str, str]]:
    """Returns {concept_n: {variety_code: form}}.  Pulls ita, lmo, fur, vec, scn, lij, sc."""
    # Table 0 in this page is the comparative one
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
    """Romance Table 0: English, Latin, Portuguese, Spanish, Catalan, French, Italian, Romanian."""
    headers, rows = _parse_table(html, 0)
    cols = {
        "fra": _column_index(headers, "french"),
        "spa": _column_index(headers, "spanish"),
        "cat": _column_index(headers, "catalan"),
    }
    print(f"  romance columns: {cols}")
    # Romance table doesn't number rows; use 1-based positional index
    out: Dict[int, Dict[str, str]] = {}
    for n, row in enumerate(rows, start=1):
        if not row:
            continue
        rec = {}
        for code in ("fra", "spa", "cat"):
            rec[code] = row[cols[code]] if cols[code] >= 0 and cols[code] < len(row) else ""
        out[n] = rec
    return out


def parse_germanic(html: str) -> Dict[int, Dict[str, str]]:
    headers, rows = _parse_table(html, 0)
    cols = {
        "deu": _column_index(headers, "german"),
        "eng": _column_index(headers, "english"),  # try modern English column
    }
    # If "English" only matches "Old English", drop it
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
    cols = {"slv": _column_index(headers, "slovene")}
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
        rec["slv"] = row[cols["slv"]] if cols["slv"] >= 0 and cols["slv"] < len(row) else ""
        out[n] = rec
    return out


def merge_all() -> Tuple[List[Dict[str, str]], List[str]]:
    print("[1/4] downloading Italian-languages appendix")
    it_html = fetch(URLS["italian"])
    italian = parse_italian_languages(it_html)
    print(f"   parsed {len(italian)} Swadesh rows")

    print("[2/4] downloading Romance appendix")
    rm_html = fetch(URLS["romance"])
    romance = parse_romance(rm_html)
    print(f"   parsed {len(romance)} rows")

    print("[3/4] downloading Germanic appendix")
    ge_html = fetch(URLS["germanic"])
    germanic = parse_germanic(ge_html)
    print(f"   parsed {len(germanic)} rows")

    print("[4/4] downloading Slavic appendix")
    sl_html = fetch(URLS["slavic"])
    slavic = parse_slavic(sl_html)
    print(f"   parsed {len(slavic)} rows")

    # English fallback: use the concept_en column from italian
    # (it's the canonical English Swadesh form)
    rows_out: List[Dict[str, str]] = []
    all_n = sorted(set(italian) | set(romance) | set(germanic) | set(slavic))
    for n in all_n:
        rec: Dict[str, str] = {"n": str(n)}
        rec["concept_en"] = italian.get(n, {}).get("concept_en", "")
        for code in ("ita", "lmo", "fur", "vec", "scn", "lij", "sc"):
            rec[code] = italian.get(n, {}).get(code, "")
        for code in ("fra", "spa", "cat"):
            rec[code] = romance.get(n, {}).get(code, "")
        for code in ("deu", "eng"):
            rec[code] = germanic.get(n, {}).get(code, "")
        # If germanic didn't give us eng, fallback to concept_en
        if not rec.get("eng"):
            rec["eng"] = rec["concept_en"]
        rec["slv"] = slavic.get(n, {}).get("slv", "")
        rows_out.append(rec)

    columns = ["n", "concept_en",
               "ita", "fra", "spa", "cat", "deu", "slv", "eng",
               "fur", "lij", "lmo", "sc", "scn", "vec"]
    return rows_out, columns


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

    # Coverage stats
    print(f"\nWrote {len(rows)} rows × {len(cols)} columns to {args.out}")
    print("\nNon-empty cells per language:")
    for code in cols[2:]:
        n_filled = sum(1 for r in rows if r.get(code))
        print(f"  {code:<5}  {n_filled}/{len(rows)} rows filled")

    # Show common-rows where ALL 13 are non-empty
    common = [r for r in rows
              if all(r.get(c) for c in
                     ("ita","fra","spa","cat","deu","slv","eng",
                      "fur","lij","lmo","sc","scn","vec"))]
    print(f"\nRows where ALL 13 varieties have a non-empty form: {len(common)}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
