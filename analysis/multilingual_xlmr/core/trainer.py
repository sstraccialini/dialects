"""
Continued MLM and TLM pretraining of XLM-R.

  - `run_mlm_training`: standard MLM on flat monolingual sentences (Wiki).
  - `run_tlm_training`: Translation LM on (italian, dialect) sentence pairs
    from OLDI — concatenates each pair with [SEP], runs MLM masking on the
    concatenation. Cross-lingual masking encourages alignment between
    Italian and dialect representations.
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


def _mixed_precision() -> dict:
    """fp16/bf16 kwargs for HuggingFace Trainer based on GPU capability."""
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
    *,
    epochs: int = 3,
    batch_size: int = 16,
    grad_accumulation: int = 4,
    lr: float = 3e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    max_length: int = 128,
    mlm_probability: float = 0.15,
) -> Path:
    """Continued MLM pretraining on a flat list of monolingual sentences."""
    print(f"\n[MLM] base='{base_model}'  texts={len(texts):,}  epochs={epochs}  lr={lr}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForMaskedLM.from_pretrained(base_model)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True,
                         max_length=max_length, padding=False)

    ds = HFDataset.from_dict({"text": texts}).map(
        tokenize_fn, batched=True, remove_columns=["text"]
    )
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability,
    )

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accumulation,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
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
    *,
    epochs: int = 5,
    batch_size: int = 16,
    grad_accumulation: int = 4,
    lr: float = 3e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    max_length: int = 256,
    mlm_probability: float = 0.15,
) -> Path:
    """Translation Language Model training on (italian, dialect) pairs.

    Each pair is fed to the tokenizer as a sentence pair, producing
    [CLS] italian [SEP] dialect [SEP]. Standard MLM masking is applied
    across the whole concatenation, so the model must use cross-lingual
    context to reconstruct masked tokens.
    """
    print(f"\n[TLM] base='{base_model}'  pairs={len(pairs):,}  epochs={epochs}  lr={lr}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForMaskedLM.from_pretrained(base_model)

    def tokenize_fn(batch):
        return tokenizer(
            batch["italian"], batch["dialect"],
            truncation=True, max_length=max_length, padding=False,
        )

    ds = HFDataset.from_dict({
        "italian": [p[0] for p in pairs],
        "dialect": [p[1] for p in pairs],
    }).map(tokenize_fn, batched=True, remove_columns=["italian", "dialect"])

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability,
    )

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accumulation,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
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
