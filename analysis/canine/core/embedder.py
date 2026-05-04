"""
CANINE (character-level) sentence embedder. Mean-pooled, L2-normalised.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .config import DEFAULT_MODEL_NAME, MAX_LENGTH, BATCH_SIZE


class CanineEmbedder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        max_length: int = MAX_LENGTH,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.model_name = model_name
        self.max_length = max_length
        print(f"Loading '{model_name}' onto {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _mean_pool(model_output, attention_mask):
        token_embeddings = model_output[0]
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / denom

    def encode(
        self,
        texts: List[str],
        batch_size: int = BATCH_SIZE,
        verbose: bool = True,
    ) -> np.ndarray:
        if not texts:
            raise ValueError("encode() received an empty list of texts")
        chunks: List[np.ndarray] = []
        iterator = range(0, len(texts), batch_size)
        if verbose:
            iterator = tqdm(iterator, desc="Embedding", unit="batch")
        for i in iterator:
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc)
            pooled = self._mean_pool(out, enc["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            chunks.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(chunks)

    def encode_per_variety(
        self,
        data: Dict[str, List[str]],
        codes: List[str],
        batch_size: int = BATCH_SIZE,
    ) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for code in codes:
            if code not in data:
                continue
            out[code] = self.encode(data[code], batch_size=batch_size)
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
