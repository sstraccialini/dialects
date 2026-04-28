"""
End-to-end orchestrator for continued pretraining of XLM-R on Italian
dialect Wikipedia. Saves the adapted model to OUTPUT_DIR.

Run:
    python experiments/continued_pretraining/src/run_cp.py
"""
from __future__ import annotations

import os
import sys
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset

from config import (
    REPO_ROOT, WIKI_DIR, OUTPUT_DIR,
    BASE_MODEL, DIALECT_CSVS, N_PER_FILE,
    NUM_EPOCHS, PER_DEVICE_BATCH_SIZE, LEARNING_RATE, MAX_LENGTH,
    MLM_PROBABILITY, WARMUP_STEPS, LOGGING_STEPS, RANDOM_SEED,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dialect_text() -> list[str]:
    """Load and combine Italo-Romance dialect Wiki text."""
    all_text: list[str] = []
    print(f"Loading dialect text from {WIKI_DIR} ...")
    for fname in DIALECT_CSVS:
        path = WIKI_DIR / fname
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping")
            continue
        df = pd.read_csv(path)
        if len(df) > N_PER_FILE:
            df = df.sample(n=N_PER_FILE, random_state=RANDOM_SEED).reset_index(drop=True)
        sentences = df["text"].astype(str).tolist()
        sentences = [s.strip() for s in sentences if s.strip()]
        print(f"  {fname:10s}  {len(sentences):6d} sentences")
        all_text.extend(sentences)
    print(f"Total: {len(all_text)} sentences\n")
    return all_text


class TextDataset(Dataset):
    """Pre-tokenised dataset: stores input_ids / attention_mask per row."""

    def __init__(self, texts: list[str], tokenizer, max_length: int):
        print(f"Tokenising {len(texts)} texts ...")
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_special_tokens_mask=True,
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}


def main():
    set_seed(RANDOM_SEED)

    print(f"Continued pretraining of {BASE_MODEL}")
    print("=" * 60)
    print(f"  output_dir         = {OUTPUT_DIR}")
    print(f"  num_epochs         = {NUM_EPOCHS}")
    print(f"  batch_size         = {PER_DEVICE_BATCH_SIZE}")
    print(f"  learning_rate      = {LEARNING_RATE}")
    print(f"  max_length         = {MAX_LENGTH}")
    print(f"  mlm_probability    = {MLM_PROBABILITY}")
    print()

    # Data
    texts = load_dialect_text()

    # Tokeniser + model
    print(f"Loading tokenizer + model ({BASE_MODEL}) ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL, use_safetensors=True)

    dataset = TextDataset(texts, tokenizer, MAX_LENGTH)
    print(f"Dataset size: {len(dataset)} examples\n")

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROBABILITY,
    )

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "_train_tmp"),
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        save_strategy="no",       # no intermediate checkpoints
        report_to=[],
        fp16=torch.cuda.is_available(),
        seed=RANDOM_SEED,
        dataloader_drop_last=True,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print("Starting training ...\n")
    trainer.train()

    print(f"\nSaving adapted model to {OUTPUT_DIR} ...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Cleanup intermediate train_tmp directory (saves disk)
    import shutil
    tmp = OUTPUT_DIR / "_train_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)

    print(f"\nDone. Adapted model at {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
