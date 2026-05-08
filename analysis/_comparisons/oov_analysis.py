"""
OOV analysis: how much of FLORES / OLDI vocabulary is NOT covered by the
corresponding Wikipedia training corpus, per variety.

For each variety v and each evaluation corpus C in {FLORES, OLDI}:
  vocab_wiki[v]  = set of unique whitespace-tokens in Wiki[v] (normalized)
  vocab_C[v]     = set of unique whitespace-tokens in C[v]    (normalized)
  type_oov[v,C]  = |vocab_C[v] - vocab_wiki[v]| / |vocab_C[v]|
  token_oov[v,C] = sum(count(t in C[v]) for t not in vocab_wiki[v])
                   / sum(count(t in C[v]) for all t)

Tokenization is just .split() because all three corpora are already
normalized to lowercase ASCII + single spaces by Dataset/{wiki,flores,oldi}/
scripts/{generation,normalize}.py (see Dataset/wiki/PIPELINE.md §6).

Outputs (under analysis/_comparisons/results/oov_analysis/):
  oov_summary.csv             one row per (variety, corpus) with type/token OOV
                              and vocab/token counts.
  oov_heatmap.png             heatmap of token-OOV % across varieties × corpora.
  top_oov/<variety>_<corpus>.csv   top-50 OOV tokens by frequency in C[v]
                              (helps see WHAT is missing: domain words,
                              rare dialectal forms, foreign borrowings).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis._shared.varieties import (
    FLORES_DIR,
    FLORES_SLUG,
    OLDI_DIR,
    OLDI_PARQUET,
    REPO_ROOT,
    VARIETY_CODES,
    VARIETY_NAMES,
    WIKI_VARIETY_DIR,
)

OUT_DIR = REPO_ROOT / "analysis" / "_comparisons" / "results" / "oov_analysis"
TOP_OOV_DIR = OUT_DIR / "top_oov"
TOP_K_OOV = 50


# --------------------------------------------------------------------------- #
# Loaders — every loader returns a list of sentence strings (no further clean).
# --------------------------------------------------------------------------- #
def load_wiki(code: str) -> list[str]:
    path = WIKI_VARIETY_DIR[code] / f"{code}.csv"
    df = pd.read_csv(path)
    return df["text"].dropna().astype(str).tolist()


def load_flores(code: str) -> list[str]:
    path = FLORES_DIR / f"{FLORES_SLUG[code]}.txt"
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_oldi(code: str) -> list[str]:
    path = OLDI_DIR / f"{OLDI_PARQUET[code]}.parquet"
    df = pd.read_parquet(path, columns=["text"])
    return df["text"].dropna().astype(str).tolist()


# --------------------------------------------------------------------------- #
# Counting
# --------------------------------------------------------------------------- #
def token_counter(sentences: list[str]) -> Counter:
    """Whitespace tokenization on already-normalized text."""
    c: Counter = Counter()
    for s in sentences:
        c.update(s.split())
    return c


def oov_stats(
    eval_counter: Counter, train_vocab: set[str]
) -> tuple[float, float, int, int, list[tuple[str, int]]]:
    """
    Returns (type_oov, token_oov, eval_vocab_size, eval_token_count, top_oov_list).
    type_oov / token_oov in [0, 1].
    """
    eval_vocab = set(eval_counter)
    oov_types = eval_vocab - train_vocab
    eval_tokens = sum(eval_counter.values())
    oov_tokens = sum(eval_counter[t] for t in oov_types)

    type_oov = (len(oov_types) / len(eval_vocab)) if eval_vocab else 0.0
    token_oov = (oov_tokens / eval_tokens) if eval_tokens else 0.0

    top_oov = sorted(
        ((t, eval_counter[t]) for t in oov_types),
        key=lambda x: -x[1],
    )[:TOP_K_OOV]
    return type_oov, token_oov, len(eval_vocab), eval_tokens, top_oov


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def plot_heatmap(summary: pd.DataFrame, out_path: Path) -> None:
    pivot = summary.pivot(index="variety", columns="corpus", values="token_oov_pct")
    pivot = pivot.loc[[c for c in VARIETY_CODES if c in pivot.index]]

    fig, ax = plt.subplots(figsize=(6, 0.45 * len(pivot) + 1.5))
    im = ax.imshow(pivot.values, cmap="Reds", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{c}  ({VARIETY_NAMES[c]})" for c in pivot.index])

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            color = "white" if v > 50 else "black"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    color=color, fontsize=9)

    ax.set_title("Token-OOV % vs Wiki training vocab")
    fig.colorbar(im, ax=ax, label="token-OOV %")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TOP_OOV_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for code in VARIETY_CODES:
        print(f"[{code}] loading Wiki / FLORES / OLDI ...", flush=True)
        wiki_counter = token_counter(load_wiki(code))
        wiki_vocab = set(wiki_counter)
        wiki_tokens = sum(wiki_counter.values())

        for corpus_name, loader in (("flores", load_flores), ("oldi", load_oldi)):
            eval_counter = token_counter(loader(code))
            type_oov, token_oov, eval_vocab_size, eval_tokens, top_oov = oov_stats(
                eval_counter, wiki_vocab
            )

            rows.append({
                "variety":            code,
                "variety_name":       VARIETY_NAMES[code],
                "corpus":             corpus_name,
                "wiki_vocab_size":    len(wiki_vocab),
                "wiki_n_tokens":      wiki_tokens,
                "eval_vocab_size":    eval_vocab_size,
                "eval_n_tokens":      eval_tokens,
                "type_oov_pct":       round(100 * type_oov, 2),
                "token_oov_pct":      round(100 * token_oov, 2),
            })

            top_df = pd.DataFrame(top_oov, columns=["token", "count_in_eval"])
            top_df.to_csv(
                TOP_OOV_DIR / f"{code}_{corpus_name}.csv", index=False
            )

            print(
                f"  {corpus_name:6s} "
                f"type-OOV {100*type_oov:5.1f}%  "
                f"token-OOV {100*token_oov:5.1f}%  "
                f"({eval_vocab_size:,} types, {eval_tokens:,} tokens)",
                flush=True,
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "oov_summary.csv", index=False)
    plot_heatmap(summary, OUT_DIR / "oov_heatmap.png")

    print(f"\nWrote {OUT_DIR / 'oov_summary.csv'}")
    print(f"Wrote {OUT_DIR / 'oov_heatmap.png'}")
    print(f"Wrote {len(VARIETY_CODES) * 2} top-OOV CSVs under {TOP_OOV_DIR}")


if __name__ == "__main__":
    main()
