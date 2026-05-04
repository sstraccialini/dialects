"""
Embedder utilizing `sentence-transformers`.
"""

from __future__ import annotations

import gc
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import MAX_LENGTH

def embed_sentences(
    sents: List[str],
    model_name_or_path: str,
    batch_size: int = 64
) -> np.ndarray:
    """
    Generate normalized sentence embeddings using the given SentenceTransformer.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[embedder] Loading Model '{model_name_or_path}' on {device}...")
    
    model = SentenceTransformer(model_name_or_path)
    model.to(device)
    model.max_seq_length = MAX_LENGTH
    
    # Sentence Transformer .encode automatically handles batching and can normalize
    print(f"[embedder] Encoding {len(sents)} sentences in batches of {batch_size}...")
    embeddings = model.encode(
        sents,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Free memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return embeddings
