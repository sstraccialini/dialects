"""
Expand gender-marked rows in the Swadesh CSV.

Detects cells with explicit "m" / "f" / "n" gender markers (e.g.
``eles m, elas f``) and produces TWO output rows from each such Swadesh
concept: one with masculine forms, one with feminine.  For ungendered
languages within the same row (e.g. English "they") the same form
fills both output rows.  For rows without any gender marker, only the
first form of each cell is kept (single output row).

Examples:
    Input row 3:
        n=3, concept="he, she, it (3sg)",
        ita="egli, lui", fra="il", deu="er, sie, es",
        slv="on, ona, ono", eng="he, she, it (3sg)", ...
    Output rows:
        n=3_m, concept="he", ita="egli", fra="il", deu="er", slv="on", eng="he", ...
        n=3_f, concept="she", ita="lui",  fra="il", deu="sie", slv="ona", eng="she", ...

    Input row 50:
        n=50, concept="worm",
        ita="verme", lmo="giuanìŋ, biótt, cagnon", ...
    Output rows:
        n=50, concept="worm", ita="verme", lmo="giuanìŋ", ...     (only first form)

CLI:
    python -m gold.lexicostatistical.expand_gendered \
        --in  gold/lexicostatistical/wordlists/wordlist_swadesh207.csv \
        --out gold/lexicostatistical/wordlists/wordlist_swadesh207_split.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# Detect a single gender annotation: a word followed by optional " m"/" f"/" n"
# and an optional " sg"/" pl"/" near"/" far" qualifier, ending at comma or end.
GENDER_RE = re.compile(
    r"\s+(?P<gender>m|f|n)\b(?:\s+(?:sg|pl|near|far))*\s*(?=,|$|\)|;|/)"
)

# Swadesh concepts that are SEMANTICALLY gendered even when no language in
# the row has explicit m/f markers.  When matched, we split positionally:
#   first form  → masc
#   second form → fem
# (third forms, if any, are dropped).
SEMANTICALLY_GENDERED_RE = re.compile(
    r"\bhe\b\s*,\s*\bshe\b"     # "he, she" (rows 3, 6 sometimes)
    r"|\bhis\b\s*,\s*\bher\b"   # "his, her" (possessives)
    r"|\bthey\s+\(3pl\)\b"      # "they (3pl)" — collapses m/f in some langs
    , re.IGNORECASE
)


VARIETY_COLS = ["ita", "fra", "spa", "cat", "deu", "slv", "eng",
                "fur", "lij", "lmo", "sc", "scn", "vec"]


def _split_alternates(cell: str) -> List[str]:
    """Split a cell on commas — each part may have its own gender marker."""
    return [p.strip() for p in cell.split(",") if p.strip()]


def _classify_form(form: str) -> Tuple[str, str | None]:
    """Strip the gender marker from a form.

    Returns ``(clean_form, gender)`` where gender is 'm', 'f', 'n' or None.
    Examples:
        "él m sg"        -> ("él", "m")
        "ella f"         -> ("ella", "f")
        "io"             -> ("io", None)
        "this (proximal)"-> ("this", None)
    """
    m = GENDER_RE.search(form + ",")
    if m:
        gender = m.group("gender")
        clean = (form[:m.start()] + form[m.end():]).strip().rstrip(",")
        # also strip trailing parentheticals like "(near)"
        clean = re.sub(r"\s*\([^)]*\)\s*$", "", clean).strip()
        return clean, gender
    # no gender marker: strip parenthetical glosses anyway
    clean = re.sub(r"\s*\([^)]*\)\s*$", "", form).strip()
    return clean, None


def _has_gender_marker(cells: List[str], concept_en: str = "") -> bool:
    """True if at least one cell contains an explicit m/f/n gender marker
    OR if the concept is semantically gendered (e.g., 'he, she, it')."""
    if concept_en and SEMANTICALLY_GENDERED_RE.search(concept_en):
        return True
    for c in cells:
        for part in _split_alternates(c):
            _, g = _classify_form(part)
            if g is not None:
                return True
    return False


def _extract_masc_fem(cell: str) -> Tuple[str, str]:
    """Return (masc_form, fem_form) for a single language cell.

    Strategy:
      1. Parse all alternates with their gender.
      2. Pick the first form tagged 'm' (fallback: first 'n', then first untagged).
      3. Pick the first form tagged 'f' (fallback: first untagged AFTER masc, then masc).
    """
    parts = _split_alternates(cell)
    if not parts:
        return "", ""
    classified = [_classify_form(p) for p in parts]
    masc = None
    fem = None
    untagged = [c for c, g in classified if g is None]
    for c, g in classified:
        if g == "m" and masc is None:
            masc = c
        elif g == "f" and fem is None:
            fem = c
        elif g == "n" and masc is None:
            masc = c
    if masc is None:
        masc = untagged[0] if untagged else (classified[0][0] if classified else "")
    if fem is None:
        # if the language has only one untagged form, use it for both
        # if there are multiple untagged forms and we already used the first for masc,
        # keep the second for fem
        if len(untagged) >= 2 and masc == untagged[0]:
            fem = untagged[1]
        else:
            fem = masc
    return masc, fem


def _take_first(cell: str) -> str:
    parts = _split_alternates(cell)
    if not parts:
        return ""
    clean, _ = _classify_form(parts[0])
    return clean


def expand(in_path: Path, out_path: Path) -> None:
    with in_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    fieldnames = ["n", "concept_en"] + VARIETY_COLS

    out_rows: List[Dict[str, str]] = []
    n_split = 0
    n_single = 0
    for r in rows:
        cells = [r.get(c, "") for c in VARIETY_COLS]
        if _has_gender_marker(cells, r.get("concept_en", "")):
            # Produce two rows
            mr: Dict[str, str] = {"n": f"{r['n']}_m"}
            fr: Dict[str, str] = {"n": f"{r['n']}_f"}
            mr["concept_en"] = (r["concept_en"] + " (m)").strip()
            fr["concept_en"] = (r["concept_en"] + " (f)").strip()
            for col, cell in zip(VARIETY_COLS, cells):
                m_form, f_form = _extract_masc_fem(cell)
                mr[col] = m_form
                fr[col] = f_form
            out_rows.append(mr)
            out_rows.append(fr)
            n_split += 1
        else:
            single: Dict[str, str] = {"n": r["n"], "concept_en": r["concept_en"]}
            for col, cell in zip(VARIETY_COLS, cells):
                single[col] = _take_first(cell)
            out_rows.append(single)
            n_single += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(out_rows)

    print(f"Read   : {len(rows)} input rows from {in_path}")
    print(f"Gendered rows split into masc + fem : {n_split} → {2 * n_split} rows")
    print(f"Single-form rows kept as-is         : {n_single}")
    print(f"Wrote  : {len(out_rows)} output rows to {out_path}")

    # Coverage stats
    print("\nNon-empty cells per language:")
    for c in VARIETY_COLS:
        n_filled = sum(1 for r in out_rows if r.get(c))
        print(f"  {c:<3}  {n_filled}/{len(out_rows)} rows filled")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", type=Path, required=True)
    ap.add_argument("--out", dest="out_path", type=Path, required=True)
    args = ap.parse_args(argv)
    expand(args.in_path, args.out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
