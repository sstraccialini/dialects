import logging
import time
import os
import sys

# Ensure src module is in path
sys.path.append(os.path.dirname(__file__))

from src.config import RESULTS_DIR, MODEL_NAME
from src.data_loader import load_data
from src.embedder import Embedder
from src.cluster import analyze_and_plot
import pandas as pd
import numpy as np

def main():
    start_time = time.time()
    print(f"Loading data using MODEL: {MODEL_NAME}")
    
    # Check if results folder exist
    if not RESULTS_DIR.exists():
        RESULTS_DIR.mkdir(parents=True)
    
    # 1. Load Texts
    df = load_data()
    print(f"Loaded {len(df)} total sentences across {df['lang'].nunique()} languages.")
    
    # 2. Extract Embeddings
    print("Initializing Multi-lingual Embedder (e.g. mBERT / XLM-R)")
    embedder = Embedder(model_name=MODEL_NAME)
    
    texts = df['text'].tolist()
    embeddings = embedder.encode(texts, batch_size=32)
    
    print(f"Extraction successful. Encoding matrix shape: {embeddings.shape}")
    
    # 3. Analyze, cluster and plot
    print("Performing cross-language structural alignment & grouping computations...")
    analyze_and_plot(df, embeddings)

    print("Pipeline completed successfully in {:.2f}s.".format(time.time() - start_time))

if __name__ == '__main__':
    main()
