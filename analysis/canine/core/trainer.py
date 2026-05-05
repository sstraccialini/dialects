"""
Continued MLM pretraining of CANINE on Wiki text (and TLM on OLDI pairs).

CANINE has no native `CanineForMaskedLM` head in HuggingFace
(`AutoModelForMaskedLM` cannot load it). We add a small char-level
prediction head over a fixed Unicode codepoint vocabulary covering
Latin (basic, supplement, ext-A, ext-B), IPA Extensions and combining
diacritics — enough for all 6 Italo-Romance dialects + 7 standard
languages.

Masked positions in the input are replaced with CANINE's MASK
codepoint (0xE003); the head predicts the original codepoint's vocab
index. After training, only the underlying `CanineModel` encoder is
saved so that the embedder can reload it via `AutoModel.from_pretrained`.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    CanineConfig,
    CanineModel,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import MaskedLMOutput


# --------------------------------------------------------------------------- #
# CANINE special codepoints (mirrors transformers.tokenization_canine)
# --------------------------------------------------------------------------- #
CANINE_PAD = 0x0000
CANINE_CLS = 0xE000
CANINE_SEP = 0xE001
CANINE_BOS = 0xE002
CANINE_MASK = 0xE003


# --------------------------------------------------------------------------- #
# Char-level MLM vocabulary
# --------------------------------------------------------------------------- #
# A fixed small vocab that covers all printable codepoints we expect to see
# in Italo-Romance + 7 standard languages. Anything outside falls to UNK.
_CHAR_RANGES: List[Tuple[int, int]] = [
    (0x20, 0x7F),    # ASCII printable + DEL
    (0x80, 0xFF),    # Latin-1 Supplement (à, è, ñ, ß, …)
    (0x100, 0x17F),  # Latin Extended-A   (ā, č, ě, ł, …)
    (0x180, 0x24F),  # Latin Extended-B   (ƒ, ǎ, ə, …)
    (0x250, 0x2AF),  # IPA Extensions
    (0x300, 0x36F),  # Combining Diacritical Marks
    (0x2000, 0x206F),  # General Punctuation (en/em dash, “smart” quotes)
    (0x20A0, 0x20CF),  # Currency
]


def _build_char_vocab() -> Tuple[dict, int]:
    """codepoint → small vocab index. 0=PAD, 1=UNK, 2..N = codepoints."""
    vocab: dict = {}
    idx = 2
    for lo, hi in _CHAR_RANGES:
        for cp in range(lo, hi + 1):
            vocab[cp] = idx
            idx += 1
    return vocab, idx


_CHAR_VOCAB, _VOCAB_SIZE = _build_char_vocab()
_VOCAB_KEYS = list(_CHAR_VOCAB.keys())
PAD_IDX = 0
UNK_IDX = 1


def _cp_to_label(cp: int) -> int:
    return _CHAR_VOCAB.get(cp, UNK_IDX)


# --------------------------------------------------------------------------- #
# CANINE wrapper with a small char-level MLM head
# --------------------------------------------------------------------------- #
class CanineForCharMaskedLM(PreTrainedModel):
    """`CanineModel` + linear head over a fixed Unicode-codepoint vocab."""

    config_class = CanineConfig

    def __init__(self, config: CanineConfig, mlm_vocab_size: int = _VOCAB_SIZE):
        super().__init__(config)
        self.canine = CanineModel(config)
        self.mlm_head = nn.Linear(config.hidden_size, mlm_vocab_size)
        self.mlm_vocab_size = mlm_vocab_size

    @classmethod
    def from_canine_pretrained(cls, base_model_name: str) -> "CanineForCharMaskedLM":
        canine = CanineModel.from_pretrained(base_model_name)
        wrapper = cls(canine.config)
        wrapper.canine = canine
        return wrapper

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs,
    ):
        outputs = self.canine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs.last_hidden_state
        logits = self.mlm_head(sequence_output)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.mlm_vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# --------------------------------------------------------------------------- #
# Custom data collator for char-level MLM
# --------------------------------------------------------------------------- #
class CharLevelMLMCollator:
    """Standard 80/10/10 MLM masking on character codepoints.

    - 80% of selected positions are replaced with CANINE_MASK.
    - 10% are replaced with a random codepoint drawn from our vocab.
    - 10% are kept unchanged.
    Special codepoints (PAD, CLS, SEP, BOS) are never masked.
    Labels at non-masked positions are -100 (ignored by cross-entropy).
    """

    def __init__(self, mlm_probability: float = 0.15, mask_id: int = CANINE_MASK):
        self.mlm_probability = mlm_probability
        self.mask_id = mask_id
        self._special = {CANINE_PAD, CANINE_CLS, CANINE_SEP, CANINE_BOS, CANINE_MASK}

    def __call__(self, examples):
        # Pad to longest example in this batch.
        max_len = max(len(ex["input_ids"]) for ex in examples)

        out_ids: List[List[int]] = []
        out_attn: List[List[int]] = []
        out_lbls: List[List[int]] = []

        if "token_type_ids" in examples[0]:
            out_ttids: List[List[int]] = []
        else:
            out_ttids = None

        for ex in examples:
            ids = list(ex["input_ids"])
            attn = list(ex["attention_mask"])
            ttids = list(ex["token_type_ids"]) if out_ttids is not None else None

            pad = max_len - len(ids)
            ids = ids + [CANINE_PAD] * pad
            attn = attn + [0] * pad
            if ttids is not None:
                ttids = ttids + [0] * pad

            new_ids = list(ids)
            labels = [-100] * len(ids)

            for i, cp in enumerate(ids):
                if attn[i] == 0 or cp in self._special:
                    continue
                if random.random() < self.mlm_probability:
                    labels[i] = _cp_to_label(cp)
                    r = random.random()
                    if r < 0.8:
                        new_ids[i] = self.mask_id
                    elif r < 0.9:
                        new_ids[i] = random.choice(_VOCAB_KEYS)
                    # else: keep original

            out_ids.append(new_ids)
            out_attn.append(attn)
            out_lbls.append(labels)
            if out_ttids is not None:
                out_ttids.append(ttids)

        batch = {
            "input_ids": torch.tensor(out_ids, dtype=torch.long),
            "attention_mask": torch.tensor(out_attn, dtype=torch.long),
            "labels": torch.tensor(out_lbls, dtype=torch.long),
        }
        if out_ttids is not None:
            batch["token_type_ids"] = torch.tensor(out_ttids, dtype=torch.long)
        return batch


# --------------------------------------------------------------------------- #
# Training entrypoints
# --------------------------------------------------------------------------- #
def _mixed_precision() -> dict:
    if not torch.cuda.is_available():
        return {}
    cap = torch.cuda.get_device_capability()[0]
    if cap >= 8:
        return {"bf16": True}
    if cap >= 7:
        return {"fp16": True}
    return {}


def _save_encoder(model: CanineForCharMaskedLM, tokenizer, output_dir: Path) -> None:
    """Save *only* the `CanineModel` so the embedder can load it with
    `AutoModel.from_pretrained`. The MLM head is discarded."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.canine.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


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

    `max_length` is in **characters** (not tokens). The default of 512
    covers ~99% of Wiki sentences; per-batch and grad-accumulation are
    set lower than for XLM-R because CANINE's char sequences are heavy.
    """
    print(f"\n[CANINE-MLM] base='{base_model}'  texts={len(texts):,}  "
          f"epochs={epochs}  lr={lr}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = CanineForCharMaskedLM.from_canine_pretrained(base_model)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"], truncation=True,
            max_length=max_length, padding=False,
        )

    ds = HFDataset.from_dict({"text": texts}).map(
        tokenize_fn, batched=True, remove_columns=["text"]
    )
    collator = CharLevelMLMCollator(mlm_probability=mlm_probability)

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
    _save_encoder(model, tokenizer, output_dir)
    print(f"[CANINE-MLM] saved encoder → {output_dir}")
    return output_dir


def run_tlm_training(
    base_model: str,
    pairs: List[Tuple[str, str]],
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
    char-level MLM masking applied to the joint character sequence.

    `max_length` is in characters; concatenated pairs can be long, so
    1024 is a reasonable cap. Batch is smaller than vanilla MLM because
    sequences are roughly 2x longer.
    """
    print(f"\n[CANINE-TLM] base='{base_model}'  pairs={len(pairs):,}  "
          f"epochs={epochs}  lr={lr}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = CanineForCharMaskedLM.from_canine_pretrained(base_model)

    def tokenize_fn(batch):
        return tokenizer(
            batch["italian"], batch["dialect"],
            truncation=True, max_length=max_length, padding=False,
        )

    ds = HFDataset.from_dict({
        "italian": [p[0] for p in pairs],
        "dialect": [p[1] for p in pairs],
    }).map(tokenize_fn, batched=True, remove_columns=["italian", "dialect"])

    collator = CharLevelMLMCollator(mlm_probability=mlm_probability)

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
    _save_encoder(model, tokenizer, output_dir)
    print(f"[CANINE-TLM] saved encoder → {output_dir}")
    return output_dir
