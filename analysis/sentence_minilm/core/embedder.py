"""Sentence-Transformer embedder (encodes both pretrained models and fine-tuned checkpoints)."""
from __future__ import annotations

import gc
from typing import Dict, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .config import MAX_LENGTH


def _select_device(device: str = None) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def embed_sentences(
    sents: List[str],
    model_name_or_path: str,
    batch_size: int = 64,
    device: str = None,
    max_length: int = MAX_LENGTH,
) -> np.ndarray:
    """Encode sentences with a SentenceTransformer; returns L2-normalised embeddings."""
    dev = _select_device(device)
    print(f"[embedder] Loading '{model_name_or_path}' on {dev}...")
    model = SentenceTransformer(model_name_or_path)
    model.to(dev)
    model.max_seq_length = max_length

    embeddings = model.encode(
        sents,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return embeddings.astype(np.float32)


def embed_per_variety(
    data: Dict[str, List[str]],
    codes: List[str],
    model_name_or_path: str,
    batch_size: int = 64,
    device: str = None,
    max_length: int = MAX_LENGTH,
) -> Dict[str, np.ndarray]:
    """One model load, encode every variety in turn."""
    dev = _select_device(device)
    print(f"[embedder] Loading '{model_name_or_path}' on {dev}...")
    model = SentenceTransformer(model_name_or_path)
    model.to(dev)
    model.max_seq_length = max_length

    out: Dict[str, np.ndarray] = {}
    for code in codes:
        if code not in data:
            continue
        emb = model.encode(
            data[code], batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        out[code] = emb.astype(np.float32)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return out


def aggregate_variety_vectors(
    sentence_vectors: np.ndarray,
    sentence_codes: List[str],
    codes: List[str],
):
    arr_codes = np.asarray(sentence_codes)
    rows = []
    out_codes = []
    for code in codes:
        mask = (arr_codes == code)
        if mask.sum() == 0:
            continue
        rows.append(sentence_vectors[mask].mean(axis=0))
        out_codes.append(code)
    X = np.vstack(rows).astype(np.float32)
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms, out_codes


def aggregate_from_per_variety(
    per_variety_vectors: Dict[str, np.ndarray],
    codes: List[str],
):
    rows = []
    out_codes = []
    for code in codes:
        if code not in per_variety_vectors:
            continue
        mat = per_variety_vectors[code]
        if mat.shape[0] == 0:
            continue
        rows.append(mat.mean(axis=0))
        out_codes.append(code)
    X = np.vstack(rows).astype(np.float32)
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms, out_codes
