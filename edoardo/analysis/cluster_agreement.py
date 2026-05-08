"""
Cluster agreement (ARI / NMI / V-measure) between each model's clustering
and a gold genealogical labelling.

Two gold partitions are evaluated:

    GENEALOGICAL_LABELS  — 4-way: italo_romance, other_romance, germanic, slavic
    ROMANCE_BINARY_LABELS — 2-way: romance vs non_romance

For each model:
    1. take its ``distances.csv``
    2. run hierarchical clustering (average linkage, scipy)
    3. cut at k = number of unique gold classes
    4. compute ARI, NMI, V-measure vs the gold labels

Output: ``edoardo/results/cluster_agreement.csv`` with columns
    method, experiment, variant_path, model_id,
    gold_label_set, n_classes, ari, nmi, v_measure

Run:
    python -m edoardo.analysis.cluster_agreement
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
)

from edoardo.analysis.load_models import discover_models, restrict_to_codes
from edoardo.varieties_extra import GENEALOGICAL_LABELS, ROMANCE_BINARY_LABELS


RESULTS_DIR_DEFAULT = Path(__file__).resolve().parents[1] / "results"

GOLD_PARTITIONS = {
    "genealogical_4way":   GENEALOGICAL_LABELS,
    "romance_binary":      ROMANCE_BINARY_LABELS,
}


def _agreement_one(model, gold_name: str, gold_labels: dict[str, str]
                   ) -> dict:
    target = list(gold_labels.keys())
    mat, model_labels = model.load_distances()
    shared = [c for c in target if c in model_labels]
    if len(shared) < 4:
        return {"gold_label_set": gold_name, "n_classes": 0,
                "n_shared": len(shared),
                "ari": float("nan"), "nmi": float("nan"), "v_measure": float("nan")}
    md, _ = restrict_to_codes(mat, model_labels, shared)

    # Hierarchical clustering on cosine-distance condensed form
    cond = squareform(md, checks=False)
    Z = linkage(cond, method="average")
    classes = sorted(set(gold_labels[c] for c in shared))
    k = len(classes)
    pred = fcluster(Z, k, criterion="maxclust").tolist()
    truth = [gold_labels[c] for c in shared]

    return {
        "gold_label_set": gold_name,
        "n_classes": k,
        "n_shared": len(shared),
        "ari":       float(adjusted_rand_score(truth, pred)),
        "nmi":       float(normalized_mutual_info_score(truth, pred)),
        "v_measure": float(v_measure_score(truth, pred)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR_DEFAULT)
    ap.add_argument("--include-old", action="store_true")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    models = discover_models(include_old=args.include_old)
    if not models:
        print("No models found.", file=sys.stderr)
        return 1

    rows = []
    for m in models:
        for gname, glabels in GOLD_PARTITIONS.items():
            res = _agreement_one(m, gname, glabels)
            rows.append({
                "method": m.method,
                "experiment": m.experiment,
                "variant_path": m.variant_path,
                "model_id": m.short_id,
                **res,
            })

    df = pd.DataFrame(rows)
    out_path = args.out_dir / "cluster_agreement.csv"
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"  → {out_path}")

    print("\nTop ARI (genealogical 4-way):")
    g4 = df[df["gold_label_set"] == "genealogical_4way"].sort_values("ari", ascending=False)
    print(g4[["model_id", "ari", "nmi", "v_measure"]].head(15).to_string(index=False))

    print("\nTop ARI (romance binary):")
    rb = df[df["gold_label_set"] == "romance_binary"].sort_values("ari", ascending=False)
    print(rb[["model_id", "ari", "nmi", "v_measure"]].head(15).to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
