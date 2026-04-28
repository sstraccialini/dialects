"""
Compare Mantel correlations: contrastive model evaluated on FULL FLORES
vs HOLDOUT FLORES, against both ASJP and NorthEuraLex ground truths.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASJP_CSV = REPO_ROOT / "manzini_savoia" / "results" / "asjp_phonetic_distance.csv"
NEL_CSV = REPO_ROOT / "manzini_savoia" / "results" / "neL_phonetic_distance.csv"
FULL_CSV = REPO_ROOT / "contrastive" / "results" / "distances.csv"
HOLDOUT_CSV = REPO_ROOT / "contrastive" / "results" / "distances_holdout.csv"

ASJP_TO_FLORES = {
    "italian":"italiano", "sicilian":"siciliano", "lombard":"lombardo",
    "ligurian":"ligure", "friulian":"friulano", "sardinian":"sardo",
    "spanish":"spagnolo", "catalan":"catalano", "french":"francese",
    "english":"inglese", "german":"tedesco", "greek":"greco",
    "arabic":"arabo", "slovenian":"sloveno",
}
NEL_TO_FLORES = {
    "italian":"italiano", "spanish":"spagnolo", "catalan":"catalano",
    "french":"francese", "english":"inglese", "german":"tedesco",
    "greek":"greco", "slovenian":"sloveno",
}


def to_cond(m): n = m.shape[0]; return m[np.triu_indices(n, k=1)]
def mantel(d1, d2, n_perm=500, seed=42):
    rng = np.random.default_rng(seed)
    cond1, cond2 = to_cond(d1), to_cond(d2)
    rho, _ = spearmanr(cond1, cond2)
    n = d1.shape[0]
    null = []
    for _ in range(n_perm):
        perm = rng.permutation(n)
        d2p = d2[perm][:, perm]
        nr, _ = spearmanr(cond1, to_cond(d2p))
        null.append(nr)
    p = float(np.mean(np.abs(np.array(null)) >= np.abs(rho)))
    return float(rho), p


def compare(gt_csv, mapping, label):
    gt = pd.read_csv(gt_csv, index_col=0)
    gt_labels = [a for a in mapping if a in gt.index]
    flores_labels = [mapping[a] for a in gt_labels]
    gt_mat = gt.loc[gt_labels, gt_labels].values

    print(f"\n=== Mantel against {label} ({len(gt_labels)} common langs) ===")
    print(f"{'Eval':<25s}  {'r':>10s}  {'p':>8s}")
    for name, csv in [("contrastive (full FLORES)", FULL_CSV),
                      ("contrastive (holdout 502)", HOLDOUT_CSV)]:
        df = pd.read_csv(csv, index_col=0)
        sub = df.loc[flores_labels, flores_labels].values
        rho, p = mantel(gt_mat, sub, n_perm=500)
        marker = "*" if p < 0.05 else ""
        print(f"{name:<25s}  {rho:>+10.4f}  {p:>8.4f} {marker}")


print("Compare contrastive model: full FLORES eval vs holdout-only eval")
compare(ASJP_CSV, ASJP_TO_FLORES, "ASJP (40 IPA words, 14 langs)")
compare(NEL_CSV, NEL_TO_FLORES, "NorthEuraLex (1016 IPA words, 8 langs)")
