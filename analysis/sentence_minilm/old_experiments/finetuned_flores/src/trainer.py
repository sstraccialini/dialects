"""
Sentence Transformers fine-tuning routines.

tsdae_wiki (Unsupervised) — TSDAE (Transformer-based Sequential Denoising Auto-Encoder)
  on monolingual dialect text from Wikipedia.

mnrl_oldi (Contrastive) — MultipleNegativesRankingLoss (MNRL) on 
  Italian ↔ dialect sentence pairs from OLDI.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

from config import (
    MAX_LENGTH,
    TSDAE_EPOCHS,
    TSDAE_LR,
    MNRL_EPOCHS,
    MNRL_LR,
    TRAIN_BATCH_SIZE,
    WARMUP_RATIO,
)

def run_tsdae_training(
    base_model: str,
    texts: List[str],
    output_dir: Path,
    epochs: int = TSDAE_EPOCHS,
    batch_size: int = TRAIN_BATCH_SIZE,
    lr: float = TSDAE_LR,
) -> Path:
    """Unsupervised TSDAE pretraining on monolingual texts. Returns output_dir."""
    print(f"\n[TSDAE / MLM] base='{base_model}' texts={len(texts):,} epochs={epochs} lr={lr}")

    model = SentenceTransformer(base_model)
    model.max_seq_length = MAX_LENGTH

    try:
        from sentence_transformers.datasets import DenoisingAutoEncoderDataset
        from sentence_transformers.losses import DenoisingAutoEncoderLoss
    except ImportError:
        raise ImportError("TSDAE requires DenoisingAutoEncoderDataset/Loss from sentence_transformers.")

    train_data = [InputExample(texts=[t]) for t in texts]
    train_dataset = DenoisingAutoEncoderDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_loss = DenoisingAutoEncoderLoss(model, tie_encoder_decoder=True)

    warmup_steps = math.ceil(len(train_dataloader) * epochs * WARMUP_RATIO)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        weight_decay=0,
        scheduler="WarmupLinear",
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        show_progress_bar=True,
    )

    model.save(str(output_dir))
    print(f"[TSDAE / MLM] saved → {output_dir}")
    return output_dir


def run_mnrl_training(
    base_model: str,
    pairs: List[Tuple[str, str]],
    output_dir: Path,
    epochs: int = MNRL_EPOCHS,
    batch_size: int = TRAIN_BATCH_SIZE,
    lr: float = MNRL_LR,
) -> Path:
    """
    Contrastive training (MNRL) on parallel (Italian, dialect) pairs.
    Returns output_dir.
    """
    print(f"\n[MNRL / TLM] base='{base_model}' pairs={len(pairs):,} epochs={epochs} lr={lr}")

    model = SentenceTransformer(base_model)
    model.max_seq_length = MAX_LENGTH
    
    train_data = [InputExample(texts=[ita, dial]) for ita, dial in pairs]
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = math.ceil(len(train_dataloader) * epochs * WARMUP_RATIO)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        weight_decay=0,
        scheduler="WarmupLinear",
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        show_progress_bar=True,
    )

    model.save(str(output_dir))
    print(f"[MNRL / TLM] saved → {output_dir}")
    return output_dir
