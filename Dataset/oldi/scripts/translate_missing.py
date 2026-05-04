"""
Fill OLDI gaps for the 3 high-resource comparison languages that the
official OLDI seed does not provide (deu, cat, slv) by translating the
6,193 English seed sentences via Google Translate (`deep-translator`).

Output: 3 new parquet files in the same schema as the existing
OLDI files (id, iso_639_3, iso_15924, glottocode, text, url,
last_updated), saved to `Dataset/oldi/not_normalized/`. The existing
`Dataset/oldi/scripts/normalize.py` will then aggressive-normalize
them in place.

CAVEAT: these 3 languages are MACHINE-TRANSLATED (Google) while the
other 10 are HUMAN-TRANSLATED (OLDI native speakers). This must be
documented in the methodology section of any publication.

Run with:
    python Dataset/oldi/scripts/translate_missing.py
"""
from __future__ import annotations

import threading
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]    # Dataset/oldi/
SRC_DIR = ROOT / "not_normalized"

# Target languages: (ISO 639-3, Google Translate code, glottocode).
TARGETS = [
    ("deu", "de", "stan1295"),   # Standard German
    ("cat", "ca", "stan1289"),   # Standard Catalan
    ("slv", "sl", "slov1268"),   # Slovenian
]

# Concurrency settings — Google's stated limit is 5 req/s per IP (and
# 200k/day). With 8 workers we burst above 5 req/s, but on a fresh IP
# the cumulative-usage threshold (which is what actually trips the
# rate-limiter, far above the per-second number) is not reached during
# one full 3-language run (~18k requests << ~35k threshold observed).
# If you ever exceed the threshold, switch IP (VPN) before re-running.
N_WORKERS = 8
MAX_RETRIES = 3
RETRY_DELAY_S = 5


# Thread-local translator pool. `deep_translator.GoogleTranslator` mutates
# internal state (`self._source_text` etc.) on every `.translate()` call,
# so a single instance shared across N workers causes race conditions
# (one worker overwrites another's source text before the HTTP request
# fires, returning duplicates / wrong-aligned translations). One translator
# per thread fixes this without paying the cost of constructing one per
# sentence.
_TLS = threading.local()


def get_translator(gt_code: str) -> GoogleTranslator:
    cached = getattr(_TLS, "translator", None)
    cached_code = getattr(_TLS, "gt_code", None)
    if cached is None or cached_code != gt_code:
        _TLS.translator = GoogleTranslator(source="en", target=gt_code)
        _TLS.gt_code = gt_code
    return _TLS.translator


def translate_one(gt_code: str, text: str) -> str:
    """Single translation with bounded retries. Uses thread-local translator."""
    if not isinstance(text, str) or not text.strip():
        return ""
    translator = get_translator(gt_code)
    for attempt in range(MAX_RETRIES):
        try:
            return translator.translate(text)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return ""   # give up; will be reported as failed
            time.sleep(RETRY_DELAY_S * (attempt + 1))
    return ""


def translate_concurrent(texts: list[str], gt_code: str,
                         n_workers: int = N_WORKERS,
                         desc: str = "translate") -> list[str]:
    """Translate a list of texts in parallel, preserving order.

    Each worker thread owns its own GoogleTranslator instance via
    threading.local, so the library's per-instance state mutation cannot
    race across workers.
    """
    results: list[str] = [""] * len(texts)

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        # Submit jobs and remember the index of each future for ordered fill.
        futures = {
            ex.submit(translate_one, gt_code, t): i
            for i, t in enumerate(texts)
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception:
                results[idx] = ""

    return results


def main():
    eng_path = SRC_DIR / "eng_Latn.parquet"
    if not eng_path.exists():
        sys.exit(f"OLDI English source not found: {eng_path}")
    src_df = pd.read_parquet(eng_path)
    print(f"Loaded English source: {len(src_df):,} sentences")
    print()

    src_texts = src_df["text"].tolist()

    for iso3, gt_code, glott in TARGETS:
        out_path = SRC_DIR / f"{iso3}_Latn.parquet"
        if out_path.exists():
            print(f"[skip] {iso3}: {out_path.name} already exists")
            continue

        print(f"=== Translating EN → {iso3} (Google Translate code: {gt_code}) ===")
        t0 = time.time()
        translated = translate_concurrent(src_texts, gt_code, desc=f"  {iso3}")
        elapsed = time.time() - t0

        n_failed = sum(1 for t in translated if not t)
        n_ok = len(translated) - n_failed
        print(f"  done in {elapsed:.0f}s  ({n_ok:,} OK, {n_failed} failed)")

        # Build output DataFrame in same schema as eng_Latn.parquet.
        out_df = src_df.copy()
        out_df["text"] = translated
        out_df["iso_639_3"] = iso3
        out_df["glottocode"] = glott

        # Atomic write
        tmp = out_path.with_suffix(".parquet.tmp")
        out_df.to_parquet(tmp, index=False)
        tmp.replace(out_path)
        print(f"  -> {out_path}")
        print()

    print("Done. Now run `python Dataset/oldi/scripts/normalize.py` to "
          "produce normalized versions.")


if __name__ == "__main__":
    main()
