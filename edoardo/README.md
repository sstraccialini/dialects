# `edoardo/` — does fine-tuning shift the *kind* of linguistic similarity captured?

This folder reframes the project's central question.  Instead of asking
"is fine-tuning good or bad?" — which is ill-posed without a reference —
we ask:

> **Does each embedding method align with a particular *dimension* of
> linguistic similarity (genealogy, typology, geography, lexicon), and
> does fine-tuning cause a *shift* between dimensions?**

The hypothesis is that pre-trained multilingual models (XLM-R, CANINE,
MiniLM) sit closer to typological / genealogical proxies, and that
fine-tuning on dialectal data *moves* them toward surface-lexical
proxies — not "ruining" the space but rotating it onto a different
axis of variation.

---

## Layout

```
edoardo/
├── README.md
├── varieties_extra.py         # ISO mappings (lang2vec, Glottolog) + gold cluster labels
├── requirements.txt           # extra deps for this folder (lang2vec)
├── inventory_hpc.sh           # quick shell wrapper to run inventory on HPC
│
├── gold_references/
│   ├── build_uriel.py         # URIEL → 6 distance matrices (genetic, syntactic, ...)
│   ├── build_glottolog.py     # hand-coded Glottolog tree distance
│   ├── build_expert.py        # hand-curated Italian dialectology matrix
│   ├── build_asjp.py          # ASJP LDND (lexical, optional)
│   ├── build_all.py           # convenience runner
│   └── matrices/              # output: <gold_name>.npz with keys matrix, labels, meta
│
├── analysis/
│   ├── inventory_models.py    # walks analysis/ and lists every (method, exp, variant)
│   ├── load_models.py         # ModelOutput dataclass + discovery
│   ├── load_gold.py           # GoldRef discovery + loader
│   ├── correlate_with_gold.py # Spearman ρ + Mantel test for every (model, gold)
│   ├── cluster_agreement.py   # ARI / NMI / V-measure vs gold cluster labels
│   ├── cka_baseline_vs_finetuned.py  # linear CKA + Procrustes between model pairs
│   ├── shift_analysis.py      # Δρ per gold for every (A, B) experiment pair
│   └── run_all.py             # the orchestrator
│
├── results/                   # everything ends up here
└── slurm/run_edoardo.slurm    # one-shot HPC job
```

We DO NOT touch existing folders.  `analysis._shared.varieties` and
`evaluation.compare_methods` are imported, never modified.

---

## Research questions and the metrics that answer them

| Question | Metric | Where |
|---|---|---|
| Which dimension does each model align with? | Spearman ρ between model distance matrix and each gold | `correlate_with_gold.csv` |
| Is the alignment significantly above chance? | Mantel test p-value | same |
| Does the model recover gold cluster boundaries? | ARI / NMI / V-measure | `cluster_agreement.csv` |
| How much did fine-tuning change the *space*? | Linear CKA, Procrustes disparity | `cka_baseline_vs_finetuned.csv` |
| What is the shift signature of fine-tuning? | Δρ per gold (B − A) | `shift_analysis.csv` |
| One-glance summary | Best gold per model + ARI cols | `summary_leaderboard.csv` |

Spearman ρ tells you *whether* two distance orderings agree.  Mantel adds
a permutation-based p.  CKA is the only one that uses the raw embedding
vectors (so it requires `variety_vectors.npz`); the rest work from
`distances.csv` alone.

---

## Gold reference matrices

Each gold is a 13×13 distance matrix saved as `.npz` with keys
`matrix`, `labels`, `meta`.  No single matrix is "the truth" — each
captures a different dimension.

| File | Dimension | Source / method | Notes |
|---|---|---|---|
| `uriel_genetic.npz` | genealogical | URIEL `fam` features (Glottolog one-hot) → cosine | tree-derived, no contact |
| `uriel_syntactic.npz` | syntactic typology | URIEL `syntax_knn` → cosine | sparse, k-NN imputed |
| `uriel_phonological.npz` | phonology | URIEL `phonology_knn` → cosine | small feature space |
| `uriel_inventory.npz` | phoneme inventory | URIEL `inventory_knn` → cosine | from PHOIBLE |
| `uriel_geographic.npz` | geography | URIEL `geo` → cosine | proximity to fixed reference points |
| `uriel_featural.npz` | combined typology | concat syntactic + phonological + inventory | broad typology proxy |
| `glottolog_tree.npz` | genealogical (tree) | hand-coded Glottolog 5.x paths → LCA distance | no imputation, sanity check vs `uriel_genetic` |
| `expert_dialectology.npz` | curated dialectology | hand-built from Pellegrini & Maiden-Parry | encodes contact + areal as well as genealogy |
| `asjp_ldnd.npz` | lexical surface | ASJP 40-Swadesh Levenshtein, LDND | partial coverage; some rows may be NaN |

The expert matrix is meant to be *edited* by the team.  Open
`gold_references/build_expert.py`, change values in `EXPERT_PAIRWISE`,
re-run.  The builder fails loudly if any pair is missing — no silent
imputation.

### Limitations to cite in the paper

- URIEL features for low-resource dialects are mostly k-NN-imputed from
  Glottolog neighbours, which can artificially pull dialects toward the
  closest documented language (often Italian).
- `uriel_genetic` and `glottolog_tree` are both tree-based and do NOT
  capture contact / areal effects.
- The expert matrix is subjective; we publish all values explicitly so
  reviewers can disagree cell by cell.
- ASJP coverage of Italian dialects is partial; affected pairs become
  NaN and are dropped from Mantel/Spearman calculations.

References (must be cited):

- Littell, Mortensen, Lin, Kairis, Turner, Levin (2017).
  *URIEL and lang2vec: Representing Languages as Typological,
  Geographical, and Phylogenetic Vectors.* EACL.
- Hammarström, Forkel, Haspelmath, Bank (2024). *Glottolog 5.x.*
- Wichmann, Holman, Brown (eds.). *The ASJP Database.*
- Mantel (1967). *The detection of disease clustering and a generalized
  regression approach.* Cancer Research 27, 209–220.
- Kornblith, Norouzi, Lee, Hinton (2019). *Similarity of Neural Network
  Representations Revisited.* ICML. (CKA)
- Pellegrini (1977). *Carta dei dialetti d'Italia.*
- Maiden & Parry, eds. (1997). *The Dialects of Italy.* Routledge.
- Pires, Schlinger, Garrette (2019). *How multilingual is multilingual
  BERT?* ACL.  (Methodological precursor.)
- Choenni, Shutova (2022). *Investigating Language Relationships in
  Multilingual Sentence Encoders Through the Lens of Linguistic
  Typology.* CL.

---

## How to run

### A. Quick HPC inventory (do this first)

```bash
bash edoardo/inventory_hpc.sh
# or to include old_experiments/:
bash edoardo/inventory_hpc.sh --include-old
```

Produces a printed summary and a CSV at
`edoardo/results/hpc_inventory.csv` listing every available output
(distance matrix, variety vectors, checkpoint size).  Run this on HPC
to confirm what's already trained — no GPU needed.

### B. Full pipeline on HPC (one job)

```bash
sbatch edoardo/slurm/run_edoardo.slurm
```

Or, if `lang2vec` isn't yet installed in your venv:

```bash
pip install -r edoardo/requirements.txt
sbatch edoardo/slurm/run_edoardo.slurm
```

Optional environment variables:

```bash
EDOARDO_INCLUDE_OLD=1                   sbatch edoardo/slurm/run_edoardo.slurm
EDOARDO_BASELINE_EXP=mlm_wiki_to_flores sbatch edoardo/slurm/run_edoardo.slurm
EDOARDO_PERMUTATIONS=9999                sbatch edoardo/slurm/run_edoardo.slurm
EDOARDO_SKIP_ASJP=1                      sbatch edoardo/slurm/run_edoardo.slurm
```

### C. Local reproduction

```bash
pip install lang2vec
python -m edoardo.gold_references.build_all
python -m edoardo.analysis.run_all
```

CKA / Procrustes are skipped when `variety_vectors.npz` files are not
present locally; everything else works from `distances.csv` alone.

### D. Edit the expert matrix

```bash
$EDITOR edoardo/gold_references/build_expert.py    # change EXPERT_PAIRWISE
python -m edoardo.gold_references.build_expert     # re-build that one matrix
python -m edoardo.analysis.run_all                 # re-correlate
```

### E. Pin a baseline experiment for shift / CKA

By default we report every (A, B) pair within a method.  If you want
ALL fine-tuned experiments compared back to one specific baseline:

```bash
python -m edoardo.analysis.run_all \
    --baseline-experiment mlm_wiki_to_flores_oldi
```

---

## Output files (everything lands in `results/`)

| File | What it shows |
|---|---|
| `hpc_inventory.csv` | Every (method, experiment, variant) tuple available, with `variety_vectors.npz` flag and checkpoint size. |
| `correlation_with_gold.csv` | One row per (model × gold) — Spearman ρ, Mantel r, Mantel p, n_shared. |
| `correlation_with_gold_pivot.csv` | Wide form, one column per gold, value = Spearman ρ.  Drop-in for paper Table 1. |
| `cluster_agreement.csv` | ARI / NMI / V-measure of each model's hierarchical cut vs gold cluster labels. |
| `cka_baseline_vs_finetuned.csv` | Linear CKA + Procrustes between every comparable pair (needs vectors). |
| `shift_analysis.csv` | Δρ per gold for every pair — the dimensional shift signature. |
| `summary_leaderboard.csv` | Single combined view: best gold + Δρ + ARI per model. |

---

## How to read the results (paper-style)

Open `summary_leaderboard.csv`.  Each row is one model.  Two columns
matter most:

- **`best_gold` + `best_rho`** — which dimension of similarity that
  model captures best.  A model with `best_gold=uriel_genetic` and
  `best_rho=0.79` is doing genealogy.  A model with `best_gold=asjp_ldnd`
  is doing surface lexicon.

- **`ari_genealogical_4way`** — does the *clustering* match a
  Romance/Germanic/Slavic split?  >0.5 is good.

Then go to `shift_analysis.csv` and read off the Δρ for the (A, B) pair
you care about (e.g. `mlm_wiki_to_flores_oldi` → `tlm_oldi_to_flores`
in `multilingual_xlmr`).  A *negative* Δρ_genetic with a *positive*
Δρ_lexical is the classic "shift toward surface" pattern.

`cka_baseline_vs_finetuned.csv` answers a related but distinct question:
*how much* did fine-tuning move the space (regardless of direction)?
Low CKA + low Δρ on every gold = catastrophic forgetting.  Low CKA +
positive Δρ on one gold = useful re-orientation.

---

## What this folder is and isn't

- It IS a scaffold for one specific reframing of the project's evaluation.
- It IS designed to leave the rest of the repo untouched.  When the
  approach proves itself, individual scripts here can be promoted into
  `evaluation/` and `analysis/_shared/`.
- It is NOT a replacement for the existing per-method evaluation —
  silhouettes, dendrograms etc. still live where they always lived.
- It is NOT meant to settle "which method is best".  It is meant to say
  "method X captures dimension Y of similarity; fine-tuning Z shifts it
  toward dimension W."
