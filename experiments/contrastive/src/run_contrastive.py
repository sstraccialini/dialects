"""
Contrastive fine-tuning + evaluation pipeline.

Steps:
1. Load FLORES italian + dialects, build (italian_i, dialect_i) train pairs
2. Fine-tune sentence-transformers wrapper around adapted XLM-R using
   MultipleNegativesRankingLoss
3. Save the contrastively-tuned model
4. Re-embed all FLORES varieties, compute distance matrix
5. Save distances + silhouette report

Run from repo root:
    python experiments/contrastive/src/run_contrastive.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader

from config import (
    REPO_ROOT, FLORES_DIR, RESULTS_DIR,
    BASE_MODEL, OUTPUT_MODEL,
    ANCHOR, DIALECTS, EVAL_VARIETIES,
    TRAIN_FRACTION, RANDOM_SEED,
    NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, MAX_LENGTH, WARMUP_STEPS,
)


def load_variety(name: str) -> list[str]:
    with open(FLORES_DIR / f"{name}.txt", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_pairs() -> tuple[list[InputExample], dict[str, list[int]]]:
    """Build training pairs and remember held-out indices per dialect."""
    rng = random.Random(RANDOM_SEED)

    italian = load_variety(ANCHOR)
    n_total = len(italian)
    print(f"FLORES italiano: {n_total} sentences")

    indices = list(range(n_total))
    rng.shuffle(indices)
    n_train = int(round(n_total * TRAIN_FRACTION))
    train_idx = sorted(indices[:n_train])
    test_idx = sorted(indices[n_train:])
    print(f"Train idx: {len(train_idx)}, Test idx: {len(test_idx)}")

    train_examples: list[InputExample] = []
    test_split: dict[str, list[int]] = {}

    for dialect in DIALECTS:
        dialect_sents = load_variety(dialect)
        n = min(len(italian), len(dialect_sents))
        for i in train_idx:
            if i >= n:
                continue
            train_examples.append(InputExample(texts=[italian[i], dialect_sents[i]]))
        test_split[dialect] = [i for i in test_idx if i < n]

    print(f"Total train pairs: {len(train_examples)}")
    return train_examples, test_split


def build_st_model(base_path: str) -> SentenceTransformer:
    """Wrap the (HuggingFace masked-LM) base model into a SentenceTransformer."""
    word_embedding = models.Transformer(base_path, max_seq_length=MAX_LENGTH)
    pooling = models.Pooling(
        word_embedding.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
    )
    return SentenceTransformer(modules=[word_embedding, pooling])


def evaluate_on_flores(model: SentenceTransformer) -> pd.DataFrame:
    """Embed every FLORES variety, return per-variety centroid + distance matrix."""
    centroids = {}
    for var in EVAL_VARIETIES:
        sents = load_variety(var)
        emb = model.encode(sents, batch_size=BATCH_SIZE, show_progress_bar=False,
                           convert_to_numpy=True)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-9)
        centroid = emb.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid).clip(min=1e-9)
        centroids[var] = centroid

    labels = list(centroids.keys())
    n = len(labels)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim = float(np.dot(centroids[labels[i]], centroids[labels[j]]))
            dist[i, j] = 1.0 - sim
    df = pd.DataFrame(dist, index=labels, columns=labels)
    return df


def main():
    print("=" * 60)
    print("Phase 5: Contrastive fine-tuning of adapted XLM-R")
    print("=" * 60)
    print(f"Base model:    {BASE_MODEL}")
    print(f"Output model:  {OUTPUT_MODEL}")
    print(f"Anchor:        {ANCHOR}")
    print(f"Dialects:      {DIALECTS}")
    print(f"Train frac:    {TRAIN_FRACTION}")
    print(f"Epochs:        {NUM_EPOCHS}")
    print(f"Batch size:    {BATCH_SIZE}")
    print(f"LR:            {LEARNING_RATE}")
    print()

    print("Step 1: build training pairs")
    train_examples, test_split = build_pairs()
    print()

    print("Step 2: load base model (Sentence-Transformer wrapper)")
    model = build_st_model(BASE_MODEL)
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
    print(f"  device: {model.device}")
    print()

    print("Step 3: contrastive training (MultipleNegativesRankingLoss)")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=NUM_EPOCHS,
        warmup_steps=WARMUP_STEPS,
        optimizer_params={"lr": LEARNING_RATE},
        show_progress_bar=True,
        use_amp=torch.cuda.is_available(),
    )
    print()

    print("Step 4: save model")
    Path(OUTPUT_MODEL).mkdir(parents=True, exist_ok=True)
    model.save(OUTPUT_MODEL)
    print(f"  saved to {OUTPUT_MODEL}")
    print()

    print("Step 5: evaluate — embed FLORES varieties + distance matrix")
    dist_df = evaluate_on_flores(model)
    out_csv = RESULTS_DIR / "distances.csv"
    dist_df.to_csv(out_csv, float_format="%.6f")
    print(f"  saved {out_csv}")
    print()

    # Save train/test split for reproducibility
    with open(RESULTS_DIR / "train_test_split.json", "w") as f:
        json.dump({
            "train_fraction": TRAIN_FRACTION,
            "seed": RANDOM_SEED,
            "test_indices_per_dialect": test_split,
            "n_train_pairs": len(train_examples),
        }, f, indent=2)
    print(f"  saved train/test split metadata")

    # Quick NN summary
    print()
    print("=== top-3 nearest neighbours (excluding self) ===")
    for var in dist_df.index:
        row = dist_df[var].drop(var).sort_values()
        top3 = row.head(3)
        formatted = ", ".join(f"{idx}({val:.4f})" for idx, val in top3.items())
        print(f"  {var:12s} -> {formatted}")


if __name__ == "__main__":
    main()
