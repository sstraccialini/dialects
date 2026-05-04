from pathlib import Path

# config.py lives in <repo>/analysis/multilingual_xlmr/wiki/src/, so REPO_ROOT
# is 5 levels up.
REPO_ROOT = Path(__file__).resolve().parents[4]
DATASET_DIR = REPO_ROOT / "Dataset"
WIKI_DIR = DATASET_DIR / "wiki"
GROUP_A_DIR   = WIKI_DIR / "dialects_in_both_OLDI_and_Flores"
LANGUAGES_DIR = WIKI_DIR / "languages"

METHOD_DIR = Path(__file__).resolve().parents[1]
METHOD_OUTPUTS_DIR = METHOD_DIR / "method_outputs"
EVALUATION_RESULTS_DIR = METHOD_DIR / "evaluation_results"
METHOD_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def outputs_subdir(variant: str = "") -> Path:
    p = METHOD_OUTPUTS_DIR if not variant else METHOD_OUTPUTS_DIR / variant
    p.mkdir(parents=True, exist_ok=True)
    return p


def evaluation_subdir(variant: str = "") -> Path:
    p = EVALUATION_RESULTS_DIR if not variant else EVALUATION_RESULTS_DIR / variant
    p.mkdir(parents=True, exist_ok=True)
    return p


RESULTS_DIR = EVALUATION_RESULTS_DIR

# Set Random Seed for reproducibility
RANDOM_SEED = 42

# Variety codes (ISO 639-3). 13 = 6 Group A dialects + 7 comparison languages.
LANGUAGES = [
    # Group A dialects
    "fur", "lij", "lmo", "sc", "scn", "vec",
    # Comparison languages
    "ita", "spa", "fra", "cat", "deu", "slv", "eng",
]

# Mapping variety → which subfolder its CSV lives in.
VARIETY_DIR = {
    "fur": GROUP_A_DIR, "lij": GROUP_A_DIR, "lmo": GROUP_A_DIR,
    "sc":  GROUP_A_DIR, "scn": GROUP_A_DIR, "vec": GROUP_A_DIR,
    "ita": LANGUAGES_DIR, "spa": LANGUAGES_DIR, "fra": LANGUAGES_DIR,
    "cat": LANGUAGES_DIR, "deu": LANGUAGES_DIR, "slv": LANGUAGES_DIR,
    "eng": LANGUAGES_DIR,
}

# Default model. Swap via CLI in run_pipeline.py.
MODEL_NAME = "xlm-roberta-base"

# Sampling settings.
# NOTE: 100k per language × 13 = 1.3M encodings — heavy on CPU.
# Reduce via CLI for development; 100k is the production cap.
SAMPLES_PER_LANG = 100_000
MAX_LENGTH = 128

# Plot resolution.
DPI = 300
