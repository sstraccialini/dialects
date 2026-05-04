"""
Fine-tuning–specific configuration for XLM-R.

Lives outside `core/` because these constants are only useful when
training (4 conditions: baseline / mlm_wiki / tlm_oldi / mlm_then_tlm).
Path/variety/embedding constants come from `..core.config`.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Conditions
# --------------------------------------------------------------------------- #
CONDITIONS = ["baseline", "mlm_wiki", "tlm_oldi", "mlm_then_tlm"]

# --------------------------------------------------------------------------- #
# OLDI parallel pairs (Italian ↔ dialect) — used for TLM training
# --------------------------------------------------------------------------- #
OLDI_PAIR_DIALECTS = ["fur", "lij", "lmo", "sc", "scn", "vec"]
OLDI_PAIR_SLUG = {
    "fur": "friulano", "lij": "ligure", "lmo": "lombardo",
    "sc":  "sardo",    "scn": "siciliano", "vec": "veneto",
}

# --------------------------------------------------------------------------- #
# Sequence length / batching for training
# --------------------------------------------------------------------------- #
MAX_LENGTH_TLM = 256                  # sentence-pair input is longer
TRAIN_BATCH_SIZE = 16
GRAD_ACCUMULATION = 4                 # effective batch = 64

# --------------------------------------------------------------------------- #
# Hyperparameters
# --------------------------------------------------------------------------- #
MLM_EPOCHS = 3
TLM_EPOCHS = 5
MLM_LR = 3e-5
TLM_LR = 3e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_WIKI_SAMPLES = 10_000             # per dialect, to keep training tractable

# --------------------------------------------------------------------------- #
# Trained checkpoints — one subfolder per condition.
# Lives at analysis/multilingual_xlmr/models/<condition>/
# --------------------------------------------------------------------------- #
METHOD_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = METHOD_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def model_dir(condition: str) -> Path:
    p = MODELS_DIR / condition
    p.mkdir(parents=True, exist_ok=True)
    return p
