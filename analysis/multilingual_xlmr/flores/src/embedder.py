"""
Contextual sentence embedder for the multilingual approach.

Given a HuggingFace model name (default: xlm-roberta-base), this module
produces L2-normalised sentence embeddings via attention-masked mean
pooling of the last hidden layer.

The extracted embeddings are unit vectors in R^D (D=768 for XLM-R base
and mBERT base). Downstream code aggregates them per variety and
computes cosine distances.

Notes:
- Model inference runs on GPU if available, CPU otherwise.
- We batch the input and show a progress bar (tqdm) so long runs are
  visible.
- `max_length` caps each sentence; FLORES+ sentences are short (~30-80
  tokens), so MAX_LENGTH=128 is plenty and avoids rare outliers.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from config import DEFAULT_MODEL_NAME, MAX_LENGTH, BATCH_SIZE


class MultilingualEmbedder:
    """Mean-pooled sentence embedder for multilingual transformers."""

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

    # ---------- Pooling ----------

    @staticmethod
    def _mean_pool(model_output, attention_mask):
        """
        Attention-masked mean pooling of the last hidden state.

        - Pads are excluded via the mask.
        - Division is clamped at a tiny epsilon to avoid NaNs on
          hypothetically all-pad inputs.
        """
        token_embeddings = model_output[0]
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / denom

    # ---------- Encoding ----------

    def encode(
        self,
        texts: List[str],
        batch_size: int = BATCH_SIZE,
        verbose: bool = True,
        centre: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of sentences into L2-normalised vectors.

        ``centre=True`` subtracts the corpus mean from the raw pooled vectors
        before L2-normalisation.  This corrects the anisotropy of BERT-style
        models (all token embeddings tend to occupy a narrow cone), spreading
        representations so that cosine distances become meaningful.

        Returns a float32 numpy array of shape (len(texts), D).
        """
        if not texts:
            raise ValueError("encode() received an empty list of texts")

        raw_chunks: List[np.ndarray] = []
        iterator = range(0, len(texts), batch_size)
        if verbose:
            iterator = tqdm(iterator, desc="Embedding batches", unit="batch")

        for i in iterator:
            batch = texts[i:i + batch_size]
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
