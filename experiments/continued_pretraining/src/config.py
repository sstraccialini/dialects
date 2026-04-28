"""
Continued pretraining of XLM-R on Italian dialect Wikipedia.

Goal: address the OOD bias of XLM-R for Italian dialects by exposing
it to dialect text via continued masked language modeling. After
adaptation we re-run the similarity analysis on FLORES+ and compare
with the out-of-the-box XLM-R results.
"""
from __future__ import annotations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WIKI_DIR = REPO_ROOT / "wiki_data"
OUTPUT_DIR = Path.home() / "xlmr-adapted-italian-dialects"

# Base model: standard multilingual XLM-R
BASE_MODEL = "xlm-roberta-base"

# Italo-Romance dialects + Italian standard from Wiki.
# Skip non-Romance / non-Italian (greco/arabo/ecc.): we want adaptation
# to italo-romance dialect distribution, not generic multilingual data.
DIALECT_CSVS = ["nap.csv", "scn.csv", "vec.csv", "lmo.csv", "sc.csv", "ita.csv"]

# Some bigger Wiki dialects to use a maximum number of sentences per file
# (limited by what we trimmed to 16k for storage). Full dataset = ~96k
# sentences = ~2-3M tokens.
N_PER_FILE = 16000

# Training hyperparameters (light-touch domain adaptation)
NUM_EPOCHS = 2
PER_DEVICE_BATCH_SIZE = 16
LEARNING_RATE = 5e-5
MAX_LENGTH = 128
MLM_PROBABILITY = 0.15
WARMUP_STEPS = 100
LOGGING_STEPS = 100

RANDOM_SEED = 42
