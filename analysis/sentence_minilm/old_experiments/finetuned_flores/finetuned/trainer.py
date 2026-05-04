"""
TSDAE (unsupervised denoising auto-encoder) and MNRL (contrastive)
training routines for SentenceTransformer.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

from ..core.config import MAX_LENGTH
from .config import (
    TSDAE_EPOCHS, TSDAE_LR,
    MNRL_EPOCHS, MNRL_LR,
    TRAIN_BATCH_SIZE, WARMUP_RATIO,
)


def run_tsdae_training(
    base_model: str,
    texts: List[str],
    output_dir: Path,
    epochs: int = TSDAE_EPOCHS,
    batch_size: int = TRAIN_BATCH_SIZE,
    lr: float = TSDAE_LR,
) -> Path:
    """Unsupervised TSDAE pretraining on monolingual texts."""
    print(f"\n[TSDAE] base='{base_model}'  texts={len(texts):,}  epochs={epochs}  lr={lr}")

    model = SentenceTransformer(base_model)
    model.max_seq_length = MAX_LENGTH

    from sentence_transformers.datasets import DenoisingAutoEncoderDataset
    from sentence_transformers.losses import DenoisingAutoEncoderLoss

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
    print(f"[TSDAE] saved → {output_dir}")
    return output_dir


def run_mnrl_training(
    base_model: str,
    pairs: List[Tuple[str, str]],
    output_dir: Path,
    epochs: int = MNRL_EPOCHS,
    batch_size: int = TRAIN_BATCH_SIZE,
    lr: float = MNRL_LR,
) -> Path:
    """Contrastive MNRL training on (anchor, positive) pairs."""
    print(f"\n[MNRL] base='{base_model}'  pairs={len(pairs):,}  epochs={epochs}  lr={lr}")

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
    print(f"[MNRL] saved → {output_dir}")
    return output_dir
