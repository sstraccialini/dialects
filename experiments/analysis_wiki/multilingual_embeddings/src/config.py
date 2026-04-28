import os
from pathlib import Path

# Base Paths
# This file lives in <repo>/analysis_wiki/multilingual_embeddings/src/config.py
PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
DATASETS_DIR = PROJECT_ROOT / "wiki_data"
RESULTS_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results')))

# Set Random Seed for reproducibility
RANDOM_SEED = 42

# List of all target languages / dialect abbreviations from datasets
# Based on files like ar.csv, it.csv, etc.
LANGUAGES = [
    "ar", "ca", "de", "el", "en", "es", "fr", 
    "ita", "lmo", "nap", "sc", "scn", "sl", "vec"
]

# Model settings
# 'xlm-roberta-base' is a standard raw pretrained model
# 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' gives high-quality aligned sentence embeddings
MODEL_NAME = "xlm-roberta-base" # Default to XLM-R as requested

# Data settings
SAMPLES_PER_LANG = 1000 # Number of sentences to sample per language to speed up testing and ensure balance
MAX_LENGTH = 128 # Truncation length for tokenizer

# Visualization
DPI = 300