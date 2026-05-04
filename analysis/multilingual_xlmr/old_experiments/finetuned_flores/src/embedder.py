"""
Mean-pooled sentence embedder for XLM-R (baseline or fine-tuned checkpoint).

Identical in behaviour to multilingual_xlmr/embedder.py — reproduced here so
this sub-project is self-contained. Accepts both HuggingFace Hub model names
("xlm-roberta-base") and local directory paths to fine-tuned models.

When loading a model saved via AutoModelForMaskedLM.save_pretrained(), the LM
head weights are present in the checkpoint but ignored by AutoModel; only the
encoder is used for mean pooling.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from config import BATCH_SIZE, MAX_LENGTH


class MultilingualEmbedder:
    def __init__(
        self,
        model_name: str,
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
    def _mean_pool(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
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
        centre: bool = True,
    ) -> np.ndarray:
        """
        ``centre=True`` subtracts the corpus mean before L2-normalisation to
        correct the anisotropy of BERT-style token embeddings.
        """
        if not texts:
            raise ValueError("encode() received an empty list")
        raw_chunks: List[np.ndarray] = []
        iterator = range(0, len(texts), batch_size)
        if verbose:
            iterator = tqdm(iterator, desc="Embedding batches", unit="batch")
        for i in iterator:
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc)
            pooled = self._mean_pool(out, enc["attention_mask"])
            raw_chunks.append(pooled.cpu().numpy().astype(np.float32))
        raw = np.vstack(raw_chunks)
        if centre:
            raw = raw - raw.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        return (raw / np.clip(norms, 1e-9, None)).astype(np.float32)
