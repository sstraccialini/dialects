from pathlib import Path

# config.py lives in <repo>/analysis/multilingual_xlmr/wiki/src/, so REPO_ROOT
# is 5 levels up.
REPO_ROOT = Path(__file__).resolve().parents[4]
DATASET_DIR = REPO_ROOT / "Dataset"
WIKI_DIR = DATASET_DIR / "wiki"

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


# Aliases kept for legacy imports inside this method's modules.
DATASETS_DIR = WIKI_DIR
RESULTS_DIR = EVALUATION_RESULTS_DIR

# Set Random Seed for reproducibility
RANDOM_SEED = 42

# Variety codes (one CSV per code in WIKI_DIR).
LANGUAGES = [
    "ar", "ca", "de", "el", "en", "es", "fr",
    "ita", "lmo", "nap", "sc", "scn", "sl", "vec",
]

# Default model. Swap via CLI in run_pipeline.py.
MODEL_NAME = "xlm-roberta-base"

# Sampling settings.
SAMPLES_PER_LANG = 1000
MAX_LENGTH = 128

# Plot resolution.
DPI = 300
