"""
07_silhouette_scoreboard.py — bar-chart scoreboard of cluster-quality
metrics across every method we have evaluation_results for.

Reads each method's silhouette_report.txt and clustering_metrics.csv,
then plots two grouped bars per method: silhouette_family and
silhouette_romance_vs_rest.

Output: plots/outputs/07_silhouette_scoreboard.png
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import PLOTS_DIR, REPO_ROOT


# ---- where to look ---------------------------------------------------------
EVAL_DIRS = [
    ("CANINE old/FLORES",         "analysis/canine/old_experiments/flores/evaluation_results"),
    ("XLM-R old/FLORES",          "analysis/multilingual_xlmr/old_experiments/flores/evaluation_results"),
    ("Word2Vec old/FLORES",       "analysis/word2vec/old_experiments/flores/evaluation_results"),
    ("Word2Vec old/Wiki",         "analysis/word2vec/old_experiments/wiki/evaluation_results"),
    ("Sentence-finetuned old",    "analysis/sentence_finetuned/flores/evaluation_results/baseline"),
    ("CANINE TLM-OLDI",           "analysis/canine/experiments/tlm_oldi_to_flores/evaluation_results/flores/centroid"),
    ("CANINE MLM-Wiki",           "analysis/canine/experiments/mlm_wiki_to_flores_oldi/evaluation_results/flores/centroid"),
    ("CANINE MLM-Wiki + TLM",     "analysis/canine/experiments/mlm_wiki_then_tlm_oldi_to_flores/evaluation_results/flores/centroid"),
    ("XLM-R TLM-OLDI",            "analysis/multilingual_xlmr/experiments/tlm_oldi_to_flores/evaluation_results/flores/centroid"),
    ("XLM-R MLM-Wiki",            "analysis/multilingual_xlmr/experiments/mlm_wiki_to_flores_oldi/evaluation_results/flores/centroid"),
    ("XLM-R MLM-Wiki + TLM",      "analysis/multilingual_xlmr/experiments/mlm_wiki_then_tlm_oldi_to_flores/evaluation_results/flores/centroid"),
    ("Sentence-MiniLM MNRL-OLDI", "analysis/sentence_minilm/experiments/mnrl_oldi_to_flores/evaluation_results/flores/centroid"),
    ("FastText (Wiki)",           "analysis/fasttext/experiments/wiki_to_flores_oldi/evaluation_results/flores/centroid"),
    ("Word2Vec new (Wiki)",       "analysis/word2vec/experiments/wiki_to_flores_oldi/evaluation_results/flores/centroid"),
]

PAT_FAMILY  = re.compile(r"silhouette\s*\(family\)\s*[:=]\s*([+\-\d.]+)", re.I)
PAT_ROMANCE = re.compile(r"silhouette\s*\(romance.*?\)\s*[:=]\s*([+\-\d.]+)", re.I)


def _parse(report: Path) -> tuple[Optional[float], Optional[float]]:
    if not report.exists():
        return None, None
    txt = report.read_text(encoding="utf-8", errors="replace")
    fam = PAT_FAMILY.search(txt)
    rom = PAT_ROMANCE.search(txt)
    return (float(fam.group(1)) if fam else None,
            float(rom.group(1)) if rom else None)


def _from_clustering(p: Path) -> tuple[Optional[float], Optional[float]]:
    if not p.exists():
        return None, None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None, None
    fam = rom = None
    if "metric" in df.columns and "value" in df.columns:
        d = dict(zip(df["metric"].astype(str), df["value"]))
        fam = d.get("silhouette_family")
        rom = d.get("silhouette_romance_vs_rest")
        if fam is not None: fam = float(fam)
        if rom is not None: rom = float(rom)
    return fam, rom


def main():
    rows = []
    for label, rel in EVAL_DIRS:
        d = REPO_ROOT / rel
        report = d / "silhouette_report.txt"
        fam, rom = _parse(report)
        if fam is None or rom is None:
            f2, r2 = _from_clustering(d / "clustering_metrics.csv")
            fam = fam if fam is not None else f2
            rom = rom if rom is not None else r2
        rows.append({"method": label, "sil_family": fam, "sil_romance": rom})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df = df.dropna(subset=["sil_family", "sil_romance"], how="all")
    df = df.sort_values("sil_romance", ascending=True, na_position="first").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13, max(6, len(df) * 0.5)))
    y = np.arange(len(df))
    h = 0.36

    fam_vals = df["sil_family"].fillna(0).values
    rom_vals = df["sil_romance"].fillna(0).values

    bars_f = ax.barh(y - h/2, fam_vals, h, color="#1f77b4",
                     edgecolor="white", label="silhouette (8 families)")
    bars_r = ax.barh(y + h/2, rom_vals, h, color="#d62728",
                     edgecolor="white", label="silhouette (Romance vs rest)")

    for bar, v in zip(bars_f, fam_vals):
        ax.text(v + (0.005 if v >= 0 else -0.005), bar.get_y() + bar.get_height()/2,
                f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right",
                fontsize=8, color="#1f77b4", fontweight="bold")
    for bar, v in zip(bars_r, rom_vals):
        ax.text(v + (0.005 if v >= 0 else -0.005), bar.get_y() + bar.get_height()/2,
                f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right",
                fontsize=8, color="#d62728", fontweight="bold")

    ax.axvline(0, color="black", linewidth=0.7, alpha=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(df["method"], fontsize=10)
    ax.set_xlabel("Silhouette score (higher = better cluster separation)", fontsize=11)
    ax.set_title(
        "Cluster quality scoreboard across all evaluated models\n"
        "Family-silhouette = 8-family separation; Romance-silhouette = Romance vs non-Romance",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=10, frameon=False)
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    out = PLOTS_DIR / "07_silhouette_scoreboard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
