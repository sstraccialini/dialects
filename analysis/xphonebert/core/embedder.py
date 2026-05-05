"""
XPhoneBERT phoneme-level embedder (no GPU needed).

Two modes:
- "lookup":  read each phoneme's row from the input embedding matrix
             (no contextualisation; pure static phoneme representation).
- "single": run the full model on each phoneme as a 1-token "sentence"
             and mean-pool — phoneme is contextualised in isolation.

Both produce a dict {code: vector} where the vector is the variety's
phoneme-inventory centroid in the model's embedding space.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .config import MODEL_NAME


class PhonemeEmbedder:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
        mode: str = "lookup",
    ):
        if mode not in ("lookup", "single"):
            raise ValueError(f"mode must be 'lookup' or 'single', got {mode!r}")
        self.mode = mode

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        print(f"Loading {model_name} onto {self.device} (mode={mode}) ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Cache per-phoneme vectors so we don't re-embed when a phoneme
        # appears in several languages.
        self._cache: Dict[str, np.ndarray] = {}

    # --------------------------------------------------------------------- #
    # Phoneme → vector
    # --------------------------------------------------------------------- #
    def phoneme_vector(self, phoneme: str) -> np.ndarray:
        if phoneme in self._cache:
            return self._cache[phoneme]
        if self.mode == "lookup":
            vec = self._lookup(phoneme)
        else:
            vec = self._single(phoneme)
        self._cache[phoneme] = vec
        return vec

    def _lookup(self, phoneme: str) -> np.ndarray:
        """Average input-embeddings of every subtoken of `phoneme`."""
        # add_special_tokens=False so we don't average in CLS/SEP
        ids = self.tokenizer(
            phoneme, add_special_tokens=False, return_tensors="pt",
        ).input_ids.to(self.device)
        if ids.numel() == 0:
            # tokenizer dropped it — return zero vector with model hidden size
            return np.zeros(self.model.config.hidden_size, dtype=np.float32)
        emb = self.model.get_input_embeddings()(ids)   # (1, n_subtokens, d)
        vec = emb.mean(dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)
        return vec

    @torch.no_grad()
    def _single(self, phoneme: str) -> np.ndarray:
        """Run the full encoder on `phoneme` as a 1-phoneme input."""
        enc = self.tokenizer(
            phoneme, return_tensors="pt", add_special_tokens=True,
        ).to(self.device)
        out = self.model(**enc)
        h = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return pooled.squeeze(0).cpu().numpy().astype(np.float32)

    # --------------------------------------------------------------------- #
    # Variety centroid
    # --------------------------------------------------------------------- #
    def variety_centroid(self, phonemes: List[str]) -> np.ndarray:
        """Mean of phoneme vectors → L2-normalised."""
        if not phonemes:
            raise ValueError("variety_centroid received empty inventory")
        vecs = np.vstack([self.phoneme_vector(p) for p in phonemes])
        c = vecs.mean(axis=0)
        n = np.linalg.norm(c)
        return (c / n).astype(np.float32) if n > 0 else c.astype(np.float32)

    def all_centroids(self, inventories: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        return {code: self.variety_centroid(phs) for code, phs in inventories.items()}


# --------------------------------------------------------------------------- #
# Helpers re-used in evaluation pipeline
# --------------------------------------------------------------------------- #
def stack_centroids(
    centroids: Dict[str, np.ndarray], codes: List[str]
) -> tuple[np.ndarray, List[str]]:
    """Stack per-variety centroids into a (n_var, d) matrix in `codes` order.
    Returns (matrix, used_codes); skips any code missing from `centroids`.
    """
    rows, used = [], []
    for c in codes:
        if c in centroids:
            rows.append(centroids[c])
            used.append(c)
    X = np.vstack(rows).astype(np.float32)
    return X, used
