"""
Contrastive fine-tuning of (already CP-adapted) XLM-R on FLORES+
italian↔dialect parallel pairs.

Loss: MultipleNegativesRankingLoss (sentence-transformers standard).
Goal: explicitly enforce italian sentence_i ≈ dialect_i in embedding
space, addressing the asymmetric-adaptation gap that pure MLM did not
close (see Phase 3).
"""
from __future__ import annotations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FLORES_DIR = REPO_ROOT / "flores_data" / "flores_plus"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Base model: the CP-adapted XLM-R from Phase 3.
BASE_MODEL = str(Path.home() / "xlmr-adapted-italian-dialects")
OUTPUT_MODEL = str(Path.home() / "xlmr-adapted-contrastive")

# Italian standard is the anchor; each dialect is paired with it.
ANCHOR = "italiano"
DIALECTS = ["veneto", "siciliano", "lombardo", "sardo", "ligure", "friulano", "ladino"]

# Train/test split
TRAIN_FRACTION = 0.75
RANDOM_SEED = 42

# Training hyperparameters
NUM_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
WARMUP_STEPS = 100

# Varieties to evaluate against (same as Phase 1-3 for comparability)
EVAL_VARIETIES = [
    "veneto", "siciliano", "lombardo", "sardo", "ligure", "friulano",
    "ladino", "italiano", "spagnolo", "francese", "catalano",
    "tedesco", "inglese", "greco", "arabo", "sloveno",
]
