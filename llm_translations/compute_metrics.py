#!/usr/bin/env python3
"""
compute_metrics.py
==================
Compute automatic translation metrics (chrF++, BLEU, BERTScore F1) for a
folder of model-generated translation files, against a multilingual gold
reference table.

Filename pattern (predictions):
    {SOURCE}-to-{DIALECT}.txt
e.g.  ita-to-scn.txt, fra-to-vec.txt, deu-to-lmo.txt

The gold table is a single file (CSV / TSV / XLSX) where every column is a
language/dialect code (e.g. ita, eng, fra, scn, vec, lmo, fur, srd, lij, ...)
and every row is the same sentence across columns. For prediction file
{SOURCE}-to-{DIALECT}.txt the script uses the column {DIALECT} as the
reference.

Run:
    python compute_metrics.py \
        --pred_dir path/to/predictions \
        --gold     path/to/gold_table.csv \
        --out      metrics_results.csv \
        --on_mismatch skip
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

# Third-party metric libs
try:
    from sacrebleu.metrics import BLEU, CHRF
except ImportError as e:
    sys.exit("ERROR: sacrebleu not installed.  pip install sacrebleu")

try:
    from bert_score import score as bert_score_fn
except ImportError as e:
    sys.exit("ERROR: bert-score not installed.  pip install bert-score")

# Torch is only used for device detection
try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    _HAS_CUDA = False


# --------------------------------------------------------------------------- #
# Filename parsing
# --------------------------------------------------------------------------- #
FNAME_RE = re.compile(r"^(?P<src>[A-Za-z]{2,10})-to-(?P<tgt>[A-Za-z]{2,20})\.txt$")

# Map common language/dialect names to canonical codes.
NAME_MAP = {
    "italiano": "ita",
    "italian": "ita",
    "ita": "ita",
    "inglese": "eng",
    "english": "eng",
    "eng": "eng",
    "spagnolo": "spa",
    "spanish": "spa",
    "spn": "spa",
    "spa": "spa",
    "francese": "fra",
    "french": "fra",
    "fra": "fra",
    "tedesco": "deu",
    "german": "deu",
    "deu": "deu",
    "veneto": "vec",
    "venetian": "vec",
    "vec": "vec",
    "siciliano": "scn",
    "sicilian": "scn",
    "scn": "scn",
    "sardo": "srd",
    "sardinian": "srd",
    "srd": "srd",
    "lombardo": "lmo",
    "lombard": "lmo",
    "lmo": "lmo",
    "ligure": "lij",
    "ligurian": "lij",
    "lij": "lij",
    "friulano": "fur",
    "friulian": "fur",
    "fur": "fur",
}



def canonical_name(name: str) -> str:
    """Normalize names/codes to the canonical code used in files and columns."""
    return NAME_MAP.get(str(name).strip().lower(), str(name).strip().lower())


def parse_filename(path: Path) -> Optional[Tuple[str, str]]:
    """Return (source, target) lowercased, or None if pattern doesn't match."""
    m = FNAME_RE.match(path.name)
    if not m:
        return None
    return canonical_name(m.group("src")), canonical_name(m.group("tgt"))


# --------------------------------------------------------------------------- #
# Gold-table loader
# --------------------------------------------------------------------------- #
def load_gold_table(path: Path) -> pd.DataFrame:
    """Load CSV / TSV / XLSX gold table.  Auto-detect delimiter for plaintext."""
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, dtype=str)
    elif suffix == ".tsv":
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    elif suffix == ".csv":
        # Sniff: try comma first, fall back to tab if 1 col only
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    else:
        # Best-effort: try CSV, then TSV
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
        except Exception:
            df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

    # Strip column names + content
    df.columns = [canonical_name(c) for c in df.columns]
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    return df


# --------------------------------------------------------------------------- #
# Prediction-file cleaning
# --------------------------------------------------------------------------- #
_NUM_PREFIX_RE   = re.compile(r"^\s*\d+\s*[\.\)\]:\-]\s+")     # "1. ", "12) ", "3] ", "4: ", "5- "
_BULLET_PREFIX_RE = re.compile(r"^\s*[-*•]\s+")                # "- ", "* ", "• "
_QUOTE_PAIRS = [
    ('"', '"'),
    ("'", "'"),
    ("«", "»"),
    ("“", "”"),
    ("‘", "’"),
    ("‹", "›"),
    ("「", "」"),
]


def clean_prediction_line(line: str) -> str:
    """Remove numbering prefixes, bullets, and surrounding matching quotes."""
    s = line.strip()
    if not s:
        return s
    # Strip leading numbering / bullet markers (apply once)
    s_new = _NUM_PREFIX_RE.sub("", s, count=1)
    if s_new == s:
        s_new = _BULLET_PREFIX_RE.sub("", s, count=1)
    s = s_new.strip()
    # Strip a single layer of surrounding matching quotes if present
    for left, right in _QUOTE_PAIRS:
        if len(s) >= 2 and s.startswith(left) and s.endswith(right):
            s = s[len(left):-len(right)].strip()
            break
    return s


def load_predictions(path: Path) -> List[str]:
    """Read predictions, drop trailing empties, clean each line."""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        raw = fh.read().splitlines()
    # Drop empty trailing lines (but keep internal empty lines as empty
    # predictions — they will still be aligned positionally).
    while raw and not raw[-1].strip():
        raw.pop()
    return [clean_prediction_line(line) for line in raw]


# --------------------------------------------------------------------------- #
# Metric computation
# --------------------------------------------------------------------------- #
def compute_sacrebleu_metrics(preds: List[str], refs: List[str]) -> Tuple[float, float]:
    """Return (chrF++ score, BLEU score) — both 0-100 sacrebleu scale."""
    chrf = CHRF(word_order=2)                 # chrF++ ≡ word_order=2
    bleu = BLEU()
    chrf_score = chrf.corpus_score(preds, [refs]).score
    bleu_score = bleu.corpus_score(preds, [refs]).score
    return chrf_score, bleu_score


def compute_bertscore(
    preds: List[str],
    refs: List[str],
    *,
    model_name: str,
    device: str,
    batch_size: int = 32,
    num_layers: int = 17,
) -> Tuple[float, float, float]:
    """Return (mean P, mean R, mean F1) on bert-score's 0-1 scale."""
    P, R, F = bert_score_fn(
        cands=preds,
        refs=refs,
        model_type=model_name,
        num_layers=num_layers,
        lang=None,                  # let model_type drive tokenization
        device=device,
        batch_size=batch_size,
        verbose=False,
        rescale_with_baseline=False,
    )
    return float(P.mean()), float(R.mean()), float(F.mean())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="Compute chrF++ / BLEU / BERTScore for dialect translations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pred_dir", required=True, type=Path,
                   help="Folder containing {SOURCE}-to-{DIALECT}.txt files.")
    p.add_argument("--gold", required=True, type=Path,
                   help="Reference table (CSV / TSV / XLSX) — columns are codes.")
    p.add_argument("--out", default="metrics_results.csv", type=Path,
                   help="Output CSV file.")
    p.add_argument("--on_mismatch", choices=["skip", "truncate"], default="skip",
                   help="How to handle pred/ref length mismatches.")
    p.add_argument("--bertscore_model", default="xlm-roberta-large",
                   help="HF model name for BERTScore.")
    p.add_argument("--device", default=None,
                   help="cuda / cpu / mps. Default: auto.")
    p.add_argument("--bertscore_batch_size", type=int, default=32)
    p.add_argument("--bertscore_num_layers", type=int, default=17,
                help="Number of transformer layers for BERTScore. "
                        "Use 17 for xlm-roberta-large, especially when using a local snapshot path.")
    p.add_argument("--skip_bertscore", action="store_true",
                help="Skip BERTScore (faster sanity runs).")
    args = p.parse_args()

    # ---- device --------------------------------------------------------- #
    if args.device:
        device = args.device
    else:
        device = "cuda" if _HAS_CUDA else "cpu"
    print(f"[info] device           = {device}")
    print(f"[info] bertscore model  = {args.bertscore_model}")
    print(f"[info] on_mismatch      = {args.on_mismatch}")

    # ---- load gold ------------------------------------------------------ #
    if not args.gold.exists():
        sys.exit(f"ERROR: gold file not found: {args.gold}")
    gold_df = load_gold_table(args.gold)
    print(f"[info] loaded gold      = {args.gold}  shape={gold_df.shape}")
    print(f"[info] gold columns     = {list(gold_df.columns)}")

    # ---- iterate prediction files -------------------------------------- #
    if not args.pred_dir.exists() or not args.pred_dir.is_dir():
        sys.exit(f"ERROR: pred_dir not found or not a dir: {args.pred_dir}")

    pred_files = sorted(args.pred_dir.glob("*.txt"))
    if not pred_files:
        sys.exit(f"ERROR: no .txt files in {args.pred_dir}")

    print(f"[debug] files to analyze = {[p.name for p in pred_files]}")

    rows = []
    skipped_files = []
    for fpath in pred_files:
        parsed = parse_filename(fpath)
        if parsed is None:
            print(f"[warn] skipping {fpath.name}: filename does not match "
                  f"{{SOURCE}}-to-{{DIALECT}}.txt", file=sys.stderr)
            skipped_files.append((fpath.name, "filename pattern"))
            continue
        source, target = parsed

        # ---- gold column check --------------------------------------- #
        if target not in gold_df.columns:
            print(f"[warn] skipping {fpath.name}: target dialect column "
                  f"'{target}' not in gold ({list(gold_df.columns)})",
                  file=sys.stderr)
            skipped_files.append((fpath.name, f"missing gold column: {target}"))
            continue

        # ---- load predictions ---------------------------------------- #
        try:
            preds = load_predictions(fpath)
        except Exception as e:
            print(f"[warn] skipping {fpath.name}: cannot read ({e})",
                  file=sys.stderr)
            skipped_files.append((fpath.name, f"read error: {e}"))
            continue

        refs = [str(r).strip() for r in gold_df[target].tolist()]

        # ---- length reconciliation ----------------------------------- #
        n_preds, n_refs = len(preds), len(refs)
        if n_preds != n_refs:
            msg = (f"[warn] {fpath.name}: predictions ({n_preds}) ≠ "
                   f"references ({n_refs}).")
            if args.on_mismatch == "skip":
                print(msg + "  Skipping (per --on_mismatch=skip).",
                      file=sys.stderr)
                skipped_files.append((fpath.name, "length mismatch"))
                continue
            n = min(n_preds, n_refs)
            print(msg + f"  Truncating to {n} (per --on_mismatch=truncate).",
                  file=sys.stderr)
            preds = preds[:n]
            refs = refs[:n]

        # Drop pairs where either side is empty (avoids divide-by-zero in BLEU)
        clean = [(p, r) for p, r in zip(preds, refs) if p and r]
        if not clean:
            print(f"[warn] {fpath.name}: no non-empty (pred, ref) pairs.",
                  file=sys.stderr)
            preds_clean, refs_clean = [], []
        else:
            preds_clean, refs_clean = zip(*clean)
            preds_clean = list(preds_clean)
            refs_clean  = list(refs_clean)

        n = len(preds_clean)
        print(f"\n=== {fpath.name}  ({source} → {target}, n={n}) ===")

        # ---- sacrebleu metrics --------------------------------------- #
        if n == 0:
            chrfpp = bleu = float("nan")
        else:
            try:
                chrfpp, bleu = compute_sacrebleu_metrics(preds_clean, refs_clean)
            except Exception as e:
                print(f"[warn] sacrebleu failed for {fpath.name}: {e}",
                      file=sys.stderr)
                chrfpp, bleu = float("nan"), float("nan")

        # ---- BERTScore ----------------------------------------------- #
        if n == 0 or args.skip_bertscore:
            bs_p = bs_r = bs_f = float("nan")
        else:
            try:
                bs_p, bs_r, bs_f = compute_bertscore(
                    preds_clean, refs_clean,
                    model_name=args.bertscore_model,
                    device=device,
                    batch_size=args.bertscore_batch_size,
                    num_layers=args.bertscore_num_layers,
                )
            except Exception as e:
                print(f"[warn] bert-score failed for {fpath.name}: {e}",
                      file=sys.stderr)
                bs_p = bs_r = bs_f = float("nan")

        rows.append({
            "file":               fpath.name,
            "source_language":    source,
            "target_dialect":     target,
            "n_sentences":        n,
            "chrfpp":             round(chrfpp, 4),
            "bleu":               round(bleu, 4),
            "bertscore_precision": round(bs_p, 4),
            "bertscore_recall":    round(bs_r, 4),
            "bertscore_f1":        round(bs_f, 4),
        })

        print(f"  chrF++       = {chrfpp:6.2f}")
        print(f"  BLEU         = {bleu:6.2f}")
        print(f"  BERTScore P  = {bs_p:6.4f}")
        print(f"  BERTScore R  = {bs_r:6.4f}")
        print(f"  BERTScore F1 = {bs_f:6.4f}")

    # ---- write output --------------------------------------------------- #
    if not rows:
        sys.exit("\nERROR: no files were scored — nothing written.")

    if skipped_files:
        print(f"[debug] skipped files = {skipped_files}")

    df = pd.DataFrame(rows)
    df = df.sort_values(["target_dialect", "source_language"]).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"\n[ok] wrote {len(df)} rows -> {args.out}")

    # ---- pretty terminal summary --------------------------------------- #
    print("\n" + "=" * 78)
    print("Summary (sorted by target_dialect, then source_language):")
    print("=" * 78)
    with pd.option_context("display.max_rows", None,
                           "display.max_columns", None,
                           "display.width", 200):
        print(df.to_string(index=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()