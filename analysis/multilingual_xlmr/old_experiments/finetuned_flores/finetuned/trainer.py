"""
MLM and TLM continued-pretraining routines for XLM-R.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from datasets import Dataset as HFDataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from ..core.config import MAX_LENGTH
from .config import (
    GRAD_ACCUMULATION, MAX_LENGTH_TLM,
    MLM_EPOCHS, MLM_LR, TLM_EPOCHS, TLM_LR,
    TRAIN_BATCH_SIZE, WARMUP_RATIO, WEIGHT_DECAY,
)


def _mixed_precision() -> dict:
    if not torch.cuda.is_available():
        return {}
    cap = torch.cuda.get_device_capability()[0]
    if cap >= 8:
        return {"bf16": True}
    if cap >= 7:
        return {"fp16": True}
    return {}


def run_mlm_training(
    base_model: str,
    texts: List[str],
    output_dir: Path,
    epochs: int = MLM_EPOCHS,
    batch_size: int = TRAIN_BATCH_SIZE,
    lr: float = MLM_LR,
) -> Path:
    print(f"\n[MLM] base='{base_model}'  texts={len(texts):,}  epochs={epochs}  lr={lr}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForMaskedLM.from_pretrained(base_model)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True,
                         max_length=MAX_LENGTH, padding=False)

    ds = HFDataset.from_dict({"text": texts}).map(
        tokenize_fn, batched=True, remove_columns=["text"]
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=lr,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        **_mixed_precision(),
        save_strategy="no",
        logging_steps=200,
        report_to="none",
        dataloader_num_workers=2,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[MLM] saved → {output_dir}")
    return output_dir


def run_tlm_training(
    base_model: str,
    pairs: List[Tuple[str, str]],
    output_dir: Path,
    epochs: int = TLM_EPOCHS,
    batch_size: int = TRAIN_BATCH_SIZE,
    lr: float = TLM_LR,
) -> Path:
    print(f"\n[TLM] base='{base_model}'  pairs={len(pairs):,}  epochs={epochs}  lr={lr}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForMaskedLM.from_pretrained(base_model)

    def tokenize_fn(batch):
        return tokenizer(
            batch["italian"], batch["dialect"],
            truncation=True, max_length=MAX_LENGTH_TLM, padding=False,
        )

    ds = HFDataset.from_dict({
        "italian": [p[0] for p in pairs],
        "dialect": [p[1] for p in pairs],
    }).map(tokenize_fn, batched=True, remove_columns=["italian", "dialect"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=lr,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        **_mixed_precision(),
        save_strategy="no",
        logging_steps=100,
        report_to="none",
        dataloader_num_workers=2,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[TLM] saved → {output_dir}")
    return output_dir
