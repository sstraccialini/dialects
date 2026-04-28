"""
Pseudo-perplexity computation for masked language models.

Pseudo-perplexity is a tractable approximation of true LM perplexity for
encoder-only MLM models (BERT, RoBERTa, etc.). For each sentence, we
randomly mask MASK_RATIO of non-special tokens, predict them with the
LM, and accumulate the cross-entropy loss. Perplexity = exp(mean loss).

Lower = the LM finds the text more "predictable" = closer to its
pretraining distribution.
"""
from __future__ import annotations

import math
import random
from typing import Iterable

import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_name: str, device: torch.device):
    print(f"  loading {model_name} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Force safetensors: torch 2.4 (in flux_env) refuses .bin loading due to CVE.
    model = AutoModelForMaskedLM.from_pretrained(model_name, use_safetensors=True)
    model.to(device).eval()
    return tokenizer, model


def compute_pseudo_ppl(
    model,
    tokenizer,
    sentences: Iterable[str],
    mask_ratio: float = 0.15,
    max_length: int = 128,
    seed: int = 42,
    device: torch.device | None = None,
) -> float:
    """
    Compute pseudo-perplexity by masking ~mask_ratio of non-special
    tokens per sentence and accumulating cross-entropy loss on the
    masked positions.
    """
    if device is None:
        device = select_device()

    rng = random.Random(seed)
    losses = []

    # Excluding UNK from masking eliminates the "UNK collapse" artifact:
    # a tokenizer that doesn't know the script (e.g., bert-italian on
    # Greek/Arabic) maps everything to [UNK]. Predicting UNK->UNK is
    # trivial and yields artificially low perplexity.
    special_ids = set(tokenizer.all_special_ids)
    unk_id = tokenizer.unk_token_id
    if unk_id is not None:
        skip_ids = special_ids | {unk_id}
    else:
        skip_ids = special_ids

    n_unk_total = 0
    n_tokens_total = 0

    for sentence in sentences:
        if not sentence.strip():
            continue

        enc = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(device)

        labels = input_ids.clone()

        # Identify maskable token positions (non-special, non-UNK)
        ids_list = input_ids[0].tolist()
        n_tokens_total += len(ids_list)
        if unk_id is not None:
            n_unk_total += sum(1 for tid in ids_list if tid == unk_id)
        maskable_pos = [i for i, tid in enumerate(ids_list) if tid not in skip_ids]

        if not maskable_pos:
            continue

        n_mask = max(1, int(round(len(maskable_pos) * mask_ratio)))
        mask_idx = rng.sample(maskable_pos, min(n_mask, len(maskable_pos)))

        # Build labels: ignore index for non-masked positions
        full_labels = torch.full_like(input_ids, -100)
        for pos in mask_idx:
            full_labels[0, pos] = labels[0, pos]
            input_ids[0, pos] = tokenizer.mask_token_id

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn, labels=full_labels)

        if torch.isfinite(out.loss):
            losses.append(out.loss.item())

    compute_pseudo_ppl.last_unk_rate = (n_unk_total / n_tokens_total) if n_tokens_total else 0.0

    if not losses:
        return float("nan")

    return math.exp(sum(losses) / len(losses))


def evaluate_model_on_varieties(
    model_name: str,
    variety_sentences: dict[str, list[str]],
    mask_ratio: float = 0.15,
    max_length: int = 128,
    seed: int = 42,
    skip_existing: dict[str, float] | None = None,
    save_callback=None,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Load one model, compute pseudo-PPL on each variety, free memory.
    Returns (ppl_per_variety, unk_rate_per_variety).

    skip_existing: dict variety -> ppl already computed; those are skipped.
    save_callback(var, ppl, unk): called after each variety so progress is
                                  saved incrementally.
    """
    device = select_device()
    tokenizer, model = load_model(model_name, device)

    ppl_out: dict[str, float] = {}
    unk_out: dict[str, float] = {}
    if skip_existing:
        ppl_out.update(skip_existing)

    for var_name, sents in tqdm(variety_sentences.items(), desc=f"  {model_name}", leave=False):
        if var_name in ppl_out:
            continue
        try:
            ppl = compute_pseudo_ppl(
                model, tokenizer, sents,
                mask_ratio=mask_ratio,
                max_length=max_length,
                seed=seed,
                device=device,
            )
            unk = getattr(compute_pseudo_ppl, "last_unk_rate", 0.0)
        except Exception as e:
            print(f"  ERROR on {var_name}: {e}", flush=True)
            ppl = float("nan")
            unk = float("nan")

        ppl_out[var_name] = ppl
        unk_out[var_name] = unk

        if save_callback is not None:
            save_callback(var_name, ppl, unk)

    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return ppl_out, unk_out
