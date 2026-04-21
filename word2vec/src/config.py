from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent
WORD2VEC_DIR = CONFIG_DIR.parent
REPO_ROOT = WORD2VEC_DIR.parent

DATA_DIR = REPO_ROOT / "datasets"
RESULTS_DIR = WORD2VEC_DIR / "results"

MODEL_DIR = RESULTS_DIR / "models"
MATRIX_DIR = RESULTS_DIR / "matrices"
FIGURE_DIR = RESULTS_DIR / "figures"

TEXT_COLUMN = "text"

WORD2VEC_CONFIG = {
    "vector_size": 100,
    "window": 5,
    "min_count": 2,
    "sg": 1,
    "workers": 4,
    "epochs": 15,
    "seed": 42,
}
