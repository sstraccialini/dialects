"""
Multilingual transformer encoder (mean-pooled).

Loads any HuggingFace `AutoModel` (default xlm-roberta-base) and produces
L2-normalised mean-pooled sentence embeddings.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from .config import MODEL_NAME, MAX_LENGTH


class MultilingualEmbedder:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
        max_length: int = MAX_LENGTH,
    ):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.max_length = max_length
        print(f"Loading {model_name} onto {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def mean_pooling(model_output, attention_mask) -> torch.Tensor:
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, texts: List[str], batch_size: int = 32, centre: bool = True) -> np.ndarray:
        """Encode a list of sentences into L2-normalised vectors.

        ``centre=True`` subtracts the corpus mean from the raw pooled vectors
        before L2-normalisation. This corrects the anisotropy of BERT-style
        models (all token embeddings tend to occupy a narrow cone), spreading
        representations so that cosine distances become meaningful.
        """
        raw_chunks: List[np.ndarray] = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc)
            emb = self.mean_pooling(out, enc["attention_mask"])
            raw_chunks.append(emb.cpu().numpy().astype(np.float32))

        raw = np.vstack(raw_chunks)
        if centre:
            raw = raw - raw.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        return (raw / np.clip(norms, 1e-9, None)).astype(np.float32)

    def encode_per_variety(
        self,
        data: Dict[str, List[str]],
        codes: List[str],
        batch_size: int = 32,
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
    """L2-normalised mean of sentence vectors per variety."""
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
