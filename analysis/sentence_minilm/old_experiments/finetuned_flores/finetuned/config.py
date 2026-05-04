"""
Fine-tuning–specific configuration for the MiniLM backbone.

Lives outside `core/` because these constants are only used when
training (4 conditions: baseline / tsdae_wiki / mnrl_oldi /
tsdae_then_mnrl). Path/variety/embedding constants come from
`..core.config`.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Conditions
# --------------------------------------------------------------------------- #
CONDITIONS = ["baseline", "tsdae_wiki", "mnrl_oldi", "tsdae_then_mnrl"]

# --------------------------------------------------------------------------- #
# OLDI parallel pairs (Italian ↔ dialect) — used for MNRL training
# --------------------------------------------------------------------------- #
OLDI_PAIR_DIALECTS = ["fur", "lij", "lmo", "sc", "scn", "vec"]
OLDI_PAIR_SLUG = {
    "fur": "friulano", "lij": "ligure", "lmo": "lombardo",
    "sc":  "sardo",    "scn": "siciliano", "vec": "veneto",
}

# --------------------------------------------------------------------------- #
# Training-batch size (inference MAX_LENGTH and BATCH_SIZE come from core.config)
# --------------------------------------------------------------------------- #
TRAIN_BATCH_SIZE = 16

# --------------------------------------------------------------------------- #
# Hyperparameters
# --------------------------------------------------------------------------- #
TSDAE_EPOCHS = 3
MNRL_EPOCHS = 5
TSDAE_LR = 2e-5
MNRL_LR = 2e-5
WARMUP_RATIO = 0.1
MAX_WIKI_SAMPLES = 10_000

# --------------------------------------------------------------------------- #
# Trained checkpoints — one subfolder per condition.
# Lives at analysis/sentence_minilm/models/<condition>/
# --------------------------------------------------------------------------- #
METHOD_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = METHOD_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def model_dir(condition: str) -> Path:
    p = MODELS_DIR / condition
    p.mkdir(parents=True, exist_ok=True)
    return p
