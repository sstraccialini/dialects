"""
Unified loader API for the Wiki / FLORES / OLDI corpora with on-the-fly
normalization. Use these instead of reading raw files directly so that
all analysis pipelines apply consistent preprocessing.

Usage:
    from Dataset.loaders import load_wiki, load_flores, load_oldi
    from Dataset.loaders import load_flores_parallel, list_supported

    df = load_wiki("vec", normalize="subword_safe")
    df = load_flores("vec", normalize="hygiene")
    df = load_oldi("vec", normalize="tfidf_safe")
    df = load_flores_parallel(normalize="hygiene")  # 16-language aligned
    list_supported()                                 # availability matrix

Conventions:
- `lang` is always the ISO 639-3 code we use internally (vec, fur, lij,
  lmo, sc, scn, lld, nap, pms, roa_tara for dialects; ita, eng, fra,
  spa, cat, deu, ell, ara, slv for comparison languages).
- Source files stay native on disk — no normalization is materialized.
- All loaders return a pandas DataFrame with at least a "text" column.

Coverage matrix (✓ available, — missing):
                       Wiki   FLORES   OLDI
    fur Friulan        ✓      ✓        ✓
    lij Ligurian       ✓      ✓        ✓
    lmo Lombard        ✓      ✓        ✓
    sc  Sardinian      ✓      ✓        ✓ (filename: srd_Latn)
    scn Sicilian       ✓      ✓        ✓
    vec Venetian       ✓      ✓        ✓
    lld Ladin          ✓      ✓        —
    nap Neapolitan     ✓      —        —
    pms Piedmontese    ✓      —        —
    roa_tara Tarantino ✓      —        —
    ita Italian        ✓ (legacy) ✓    ✓
    eng English        ✓ (legacy) ✓    ✓
    fra French         ✓ (legacy) ✓    ✓
    spa Spanish        ✓ (legacy) ✓    ✓
    cat Catalan        ✓ (legacy) ✓    —
    deu German         ✓ (legacy) ✓    —
    ell Greek          ✓ (legacy) ✓    —
    ara Arabic         ✓ (legacy) ✓    —
    slv Slovenian      ✓ (legacy) ✓    —

Note on (legacy): the comparison-language Wiki CSVs in
`wiki/languages/` were produced by the original Camposampiero pipeline
and contain ~30-50% less usable content than a re-extraction with the
current Stage 1-9 pipeline would yield. They are kept as-is because
they cover the comparison languages (es/fr/de/...) for which we don't
need the same surgical cleaning we did on the dialects. To re-extract
them with the current pipeline, add their dump URLs to
`scripts/create.py` and their `_texts` codes to `FOLD_LABEL` in
`scripts/generation.py`.
"""
from pathlib import Path

import pandas as pd

from .normalize import NORMALIZERS


DATASET_ROOT = Path(__file__).resolve().parent

WIKI_GROUP_A   = {"fur", "lij", "lmo", "sc", "scn", "vec"}
WIKI_GROUP_B   = {"lld", "nap", "pms", "roa_tara"}
# Comparison languages — extracted with the legacy Camposampiero pipeline.
# Files live under wiki/languages/<iso>.csv with the same schema as Group A/B
# (text, label, article_id).
WIKI_LANGUAGES = {"ita", "eng", "fra", "spa", "cat", "deu", "ell", "ara", "slv"}
WIKI_LANGS     = WIKI_GROUP_A | WIKI_GROUP_B | WIKI_LANGUAGES

# ISO 639-3 → FLORES filename (the project uses Italian-name files).
_FLORES_NAME = {
    "fur": "friulano", "lij": "ligure", "lmo": "lombardo", "sc": "sardo",
    "scn": "siciliano", "vec": "veneto", "lld": "ladino",
    "ita": "italiano", "eng": "inglese", "fra": "francese", "spa": "spagnolo",
    "cat": "catalano", "deu": "tedesco", "ell": "greco", "ara": "arabo",
    "slv": "sloveno",
}
FLORES_LANGS = set(_FLORES_NAME)

# ISO 639-3 → OLDI parquet filename (BCP47-style "<iso>_<script>").
# Sardinian's OLDI release uses the macrolanguage code "srd".
_OLDI_NAME = {
    "fur": "fur_Latn", "lij": "lij_Latn", "lmo": "lmo_Latn",
    "sc":  "srd_Latn", "scn": "scn_Latn", "vec": "vec_Latn",
    "ita": "ita_Latn", "eng": "eng_Latn",
    "fra": "fra_Latn", "spa": "spa_Latn",
}
OLDI_LANGS = set(_OLDI_NAME)


def _check_normalize(normalize: str) -> None:
    if normalize not in NORMALIZERS:
        raise ValueError(
            f"Unknown normalize={normalize!r}. "
            f"Expected one of {sorted(NORMALIZERS)}."
        )


def _apply(text_series: pd.Series, normalize: str, lang: str) -> pd.Series:
    """Apply the requested normalizer to a text Series, lang-aware."""
    fn = NORMALIZERS[normalize]
    if normalize in ("subword_safe", "tfidf_safe"):
        return text_series.apply(lambda t: fn(t, lang=lang))
    return text_series.apply(fn)


# --------------------------------------------------------------------------- #
# Wiki
# --------------------------------------------------------------------------- #
def _wiki_subdir(lang: str) -> str:
    """Map a wiki lang to its subdirectory under Dataset/wiki/."""
    if lang in WIKI_GROUP_A:
        return "dialects_in_both_OLDI_and_Flores"
    if lang in WIKI_GROUP_B:
        return "others_dialects"
    if lang in WIKI_LANGUAGES:
        return "languages"
    raise ValueError(
        f"Unknown wiki lang {lang!r}. Expected one of {sorted(WIKI_LANGS)}."
    )


def load_wiki(lang: str, normalize: str = "hygiene") -> pd.DataFrame:
    """Load `wiki/<subdir>/<lang>.csv` with normalization applied to text.

    Returns a DataFrame with columns: text, label, article_id, lang.

    Routing:
      Group A dialects → wiki/dialects_in_both_OLDI_and_Flores/
      Group B dialects → wiki/others_dialects/
      Comparison langs → wiki/languages/         (legacy Camposampiero)
    """
    _check_normalize(normalize)
    path = DATASET_ROOT / "wiki" / _wiki_subdir(lang) / f"{lang}.csv"
    df = pd.read_csv(path)
    df["text"] = _apply(df["text"], normalize, lang)
    df["lang"] = lang
    return df


def load_wiki_meta(lang: str) -> pd.DataFrame:
    """Load `wiki/<subdir>/<lang>_meta.csv` (article-level metadata)."""
    path = DATASET_ROOT / "wiki" / _wiki_subdir(lang) / f"{lang}_meta.csv"
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# FLORES+
# --------------------------------------------------------------------------- #
def load_flores(lang: str, normalize: str = "hygiene") -> pd.DataFrame:
    """Load `flores/flores_plus/<flores_name>.txt` as one row per sentence.

    Returns a DataFrame with columns: text, lang, sentence_id.
    """
    _check_normalize(normalize)
    if lang not in FLORES_LANGS:
        raise ValueError(
            f"Unknown FLORES lang {lang!r}. Expected one of {sorted(FLORES_LANGS)}."
        )
    path = DATASET_ROOT / "flores" / "flores_plus" / f"{_FLORES_NAME[lang]}.txt"
    texts = path.read_text(encoding="utf-8").splitlines()
    df = pd.DataFrame({
        "text": texts,
        "lang": lang,
        "sentence_id": range(len(texts)),
    })
    df["text"] = _apply(df["text"], normalize, lang)
    return df


def load_flores_parallel(normalize: str = "hygiene") -> pd.DataFrame:
    """Load `parallel.tsv`: 2009 rows × 16 language columns aligned by id.

    Each language column is normalized with its own `lang` so that the
    Venetian column gets ł→l (when normalize ∈ subword_safe/tfidf_safe).
    """
    _check_normalize(normalize)
    path = DATASET_ROOT / "flores" / "flores_plus" / "parallel.tsv"
    df = pd.read_csv(path, sep="\t")
    inv = {flores_name: iso for iso, flores_name in _FLORES_NAME.items()}
    for col in df.columns:
        if col in inv:
            df[col] = _apply(df[col], normalize, inv[col])
    return df


# --------------------------------------------------------------------------- #
# OLDI
# --------------------------------------------------------------------------- #
def load_oldi(lang: str, normalize: str = "hygiene") -> pd.DataFrame:
    """Load `oldi/<oldi_name>.parquet` and normalize the text column.

    Returns the original schema (id, iso_639_3, iso_15924, glottocode, text,
    url, last_updated) with `text` normalized.
    """
    _check_normalize(normalize)
    if lang not in OLDI_LANGS:
        raise ValueError(
            f"Unknown OLDI lang {lang!r}. Expected one of {sorted(OLDI_LANGS)}."
        )
    path = DATASET_ROOT / "oldi" / f"{_OLDI_NAME[lang]}.parquet"
    df = pd.read_parquet(path)
    df["text"] = _apply(df["text"], normalize, lang)
    return df


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def list_supported() -> pd.DataFrame:
    """Return a DataFrame describing the per-language coverage across sources."""
    all_langs = sorted(WIKI_LANGS | FLORES_LANGS | OLDI_LANGS)
    rows = [
        {
            "lang": lang,
            "wiki":   lang in WIKI_LANGS,
            "flores": lang in FLORES_LANGS,
            "oldi":   lang in OLDI_LANGS,
        }
        for lang in all_langs
    ]
    return pd.DataFrame(rows)
