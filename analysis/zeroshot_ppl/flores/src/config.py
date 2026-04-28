"""
Zero-shot pseudo-perplexity of FLORES+ varieties under several pretrained
masked language models.

Goal: for each (variety, pretrained-LM) pair, measure how "intelligible"
the variety text is to that LM, with no fine-tuning. Lower perplexity =
LM was already closer to the variety in pretraining.

Output: matrix of shape (n_varieties, n_models) of pseudo-perplexities.
"""
from __future__ import annotations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DATASET_DIR = REPO_ROOT / "Dataset"
FLORES_DIR = DATASET_DIR / "flores" / "flores_plus"

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

# Pretrained masked language models (must be MLM-trained, not encoder-only).
# Names must match the FLORES variety codes only metaphorically (the LM is
# not necessarily for the same variety we evaluate).
MODELS = {
    "italian":     "dbmdz/bert-base-italian-cased",
    "spanish":     "dccuchile/bert-base-spanish-wwm-cased",
    "catalan":     "PlanTL-GOB-ES/roberta-base-ca",
    "french":      "camembert-base",
    "german":      "google-bert/bert-base-german-cased",
    "english":     "google-bert/bert-base-uncased",
}

# Varieties from FLORES+ (file stem -> nice display name)
VARIETIES = [
    "veneto", "siciliano", "lombardo", "sardo", "ligure", "friulano",
    "ladino", "italiano", "spagnolo", "francese", "catalano",
    "tedesco", "inglese", "greco", "arabo", "sloveno",
]

# Sub-sampling for speed (FLORES has 2009 sentences per variety).
# 500 sentences x 16 varieties x 7 models ~ 56k inference calls.
N_SENTENCES = 500
RANDOM_SEED = 42

MASK_RATIO = 0.15
MAX_LENGTH = 128
BATCH_SIZE = 16
