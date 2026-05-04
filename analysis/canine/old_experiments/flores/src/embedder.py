"""
Sentence embedder for CANINE (character-level encoder).

CANINE has the same Hugging Face interface as XLM-R / mBERT —
``AutoModel.from_pretrained`` returns an encoder whose ``last_hidden_state``
has shape ``(batch, seq_len, hidden_dim)``. Internally CANINE downsamples
the character sequence by 4x, runs the transformer at that coarser
granularity, and upsamples back to one vector per character — so the
output sequence length matches the input character length and the
attention mask we already have works without modification.

Mean pooling over characters (masked) gives a sentence vector. We
L2-normalise so cosine distance between variety centroids is well-behaved.

Notes:
- CANINE is a tokenizer-free model: dialectal orthography (xè, ł, ç, ...)
  is preserved at the input — no SentencePiece fallback to noisy subwords.
- max_length is in *characters*, not tokens. FLORES+ sentences are
  ~150-300 chars; 512 is a safe cap.
- CANINE is heavier per-sample than subword models because of the longer
  effective sequence; default BATCH_SIZE is therefore smaller.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from config import DEFAULT_MODEL_NAME, MAX_LENGTH, BATCH_SIZE


class CanineEmbedder:
    """Mean-pooled sentence embedder for CANINE."""

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
        """Attention-masked mean pool of last_hidden_state."""
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
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            chunks.append(pooled.cpu().numpy().astype(np.float32))

        return np.vstack(chunks)
