"""
TSDAE (Transformer-based Sequential Denoising Auto-Encoder) continued
pretraining of paraphrase-multilingual-MiniLM on Wiki text.

Used by `experiments/tsdae_wiki_to_flores_oldi/` to adapt the sentence
encoder to the 6 Italo-Romance dialects + 7 standard languages before
evaluating on FLORES+ and OLDI parallel corpora.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer, InputExample
from torch.utils.data import DataLoader


def run_tsdae_training(
    base_model: str,
    texts: List[str],
    output_dir: Path,
    *,
    epochs: int = 1,
    batch_size: int = 16,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
    max_length: int = 128,
) -> Path:
    """Unsupervised TSDAE pretraining on monolingual sentences.

    TSDAE works by corrupting the input (random word deletion) and
    teaching the encoder to reconstruct the original from the embedding.
    Spreads representations across the variety distribution we feed it
    — well-suited to mixing dialects + standard languages.
    """
    print(f"\n[TSDAE] base='{base_model}'  texts={len(texts):,}  epochs={epochs}  lr={lr}")

    model = SentenceTransformer(base_model)
    model.max_seq_length = max_length

    from sentence_transformers.datasets import DenoisingAutoEncoderDataset
    from sentence_transformers.losses import DenoisingAutoEncoderLoss

    train_data = [InputExample(texts=[t]) for t in texts]
    train_dataset = DenoisingAutoEncoderDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loss = DenoisingAutoEncoderLoss(model, tie_encoder_decoder=True)
    warmup_steps = math.ceil(len(train_dataloader) * epochs * warmup_ratio)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        weight_decay=0,
        scheduler="WarmupLinear",
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        show_progress_bar=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir))
    print(f"[TSDAE] saved → {output_dir}")
    return output_dir


def run_mnrl_training(
    base_model: str,
    pairs: list[tuple[str, str]],
    output_dir: Path,
    *,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
    max_length: int = 128,
) -> Path:
    """MultipleNegativesRankingLoss (MNRL) — symmetric contrastive learning.

    For each batch of (italian, dialect) pairs, the in-batch negatives
    pull together the matched pair and push apart all other (italian, *)
    pairs in the batch. This is the standard contrastive objective in
    sentence-transformers; works very well even with modest data sizes.
    """
    print(f"\n[MNRL] base='{base_model}'  pairs={len(pairs):,}  epochs={epochs}  lr={lr}")

    from sentence_transformers import SentenceTransformer, losses, InputExample
    from torch.utils.data import DataLoader
    import math

    model = SentenceTransformer(base_model)
    model.max_seq_length = max_length

    train_data = [InputExample(texts=[ita, dial]) for ita, dial in pairs]
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    warmup_steps = math.ceil(len(train_dataloader) * epochs * warmup_ratio)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        weight_decay=0,
        scheduler="WarmupLinear",
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        show_progress_bar=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir))
    print(f"[MNRL] saved → {output_dir}")
    return output_dir
