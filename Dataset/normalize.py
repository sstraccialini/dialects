"""
Composable text normalization for the project.

Four monotone levels — each strictly extends the previous:

  none           identity
  hygiene        NFC + curly quote → ASCII + Unicode whitespace → " "
  subword_safe   hygiene + Venetian ł→l (only when lang=="vec")
  tfidf_safe     subword_safe + strip roman numerals + strip digits
                 + strip punctuation/symbols + collapse whitespace

See PIPELINE.md §12 (Normalization & loader API) for rationale and
per-method recommendations.

The functions are deliberately simple and stateless so they can be
applied in pandas via .apply(...) without import-time side effects.

Per-encoder defaults:
  CANINE-c                            → hygiene
  XLM-R / mBERT / LaBSE / mUSE        → subword_safe
  TF-IDF / FastText / Word2Vec / NB   → tfidf_safe
"""
import re
import unicodedata


# Curly quotes / dashes → ASCII equivalents.
# Strict equivalences only (no semantic loss).
_QUOTES = str.maketrans({
    "’": "'", "‘": "'", "‚": "'",
    "”": '"', "“": '"', "„": '"', "«": '"', "»": '"',
    "—": "-", "–": "-",
})

# Non-breaking spaces and other Unicode whitespace → regular space.
_WS_UNI = re.compile(r"[     ]")

# Roman numerals as standalone tokens (length ≥2 to avoid stripping
# a single "I" or "V" from real words).
_ROMAN  = re.compile(r"\b[IVXLCDM]{2,}\b")

# Digit runs.
_DIGIT  = re.compile(r"\d+")

# Anything that's not a letter/digit/underscore/whitespace.
# Note: \w in Python is Unicode-aware → preserves accented letters.
_PUNCT  = re.compile(r"[^\w\s]")

# Multiple whitespace chars → single space.
_SPACES = re.compile(r"\s+")


def none(t: str) -> str:
    """Identity. Returned as-is for symmetry with the other levels."""
    return t


def hygiene(t: str) -> str:
    """Lossless: NFC + ASCII quotes/dashes + standard whitespace."""
    if not isinstance(t, str):
        return t
    t = unicodedata.normalize("NFC", t)
    t = t.translate(_QUOTES)
    t = _WS_UNI.sub(" ", t)
    return t


def subword_safe(t: str, lang: str = None) -> str:
    """hygiene + Venetian ł→l (only when lang=='vec').

    Use this for subword encoders (XLM-R, mBERT, LaBSE) that handle
    the rare `ł` codepoint poorly. Coherent with ITDI which is `ł`-free.
    """
    t = hygiene(t)
    if lang == "vec":
        t = t.replace("ł", "l").replace("Ł", "L")
    return t


def tfidf_safe(t: str, lang: str = None) -> str:
    """subword_safe + strip romans/digits/punctuation. Letters + spaces only.

    Use this for bag-of-features methods (TF-IDF, FastText, Word2Vec,
    Naive Bayes) where digits, roman numerals, and punctuation inflate
    cross-lingual similarity without carrying linguistic signal.
    """
    t = subword_safe(t, lang)
    # Order matters: romans first (they're letters that ALSO match \w),
    # then digits, then everything else not letter/whitespace.
    t = _ROMAN.sub(" ", t)
    t = _DIGIT.sub(" ", t)
    t = _PUNCT.sub(" ", t)
    t = _SPACES.sub(" ", t).strip()
    return t


# Public registry — used by loaders.py to dispatch the `normalize` flag.
NORMALIZERS = {
    "none":         none,
    "hygiene":      hygiene,
    "subword_safe": subword_safe,
    "tfidf_safe":   tfidf_safe,
}
