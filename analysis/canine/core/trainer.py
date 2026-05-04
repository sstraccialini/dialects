"""
Continued MLM pretraining of CANINE on Wiki text.

CANINE is a tokenizer-free, character-level encoder. Continued MLM
training adapts its character-level representations to the 6
Italo-Romance dialects + 7 standard languages before evaluating on
FLORES+ and OLDI parallel corpora.

Note: `google/canine-c` was originally trained with a character-level
autoregressive objective; running MLM-style training over it works but
re-initialises the masked LM head. `google/canine-s` (subword loss) is
slightly closer to a vanilla MLM setup; both are valid bases.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

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
    batch_size: int = 8,
    grad_accumulation: int = 8,
    lr: float = 3e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    max_length: int = 512,
    mlm_probability: float = 0.15,
) -> Path:
    """Continued MLM pretraining of CANINE on a flat list of sentences.

    `max_length` is in **characters** for CANINE (not tokens). The default
    of 512 covers ~99% of Wiki sentences; per-batch/grad-accumulation are
    set lower than for XLM-R because CANINE has heavier sequences.
    """
    print(f"\n[CANINE-MLM] base='{base_model}'  texts={len(texts):,}  "
          f"epochs={epochs}  lr={lr}")

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
    print(f"[CANINE-MLM] saved → {output_dir}")
    return output_dir


def run_tlm_training(
    base_model: str,
    pairs: list[tuple[str, str]],
    output_dir: Path,
    *,
    epochs: int = 5,
    batch_size: int = 4,
    grad_accumulation: int = 16,
    lr: float = 3e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    max_length: int = 1024,
    mlm_probability: float = 0.15,
) -> Path:
    """Sentence-pair MLM on CANINE: italian + dialect concatenated, with
    standard MLM masking applied to the joint character sequence.

    `max_length` is in characters; concatenated pairs can be long, so
    1024 is a reasonable cap. Batch is smaller than vanilla MLM because
    sequences are roughly 2x longer.
    """
    print(f"\n[CANINE-TLM] base='{base_model}'  pairs={len(pairs):,}  "
          f"epochs={epochs}  lr={lr}")

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
    print(f"[CANINE-TLM] saved → {output_dir}")
    return output_dir
