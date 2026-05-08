# New experiments under `edoardo/` — exp1 + exp2

Two new isolated experiments live under `edoardo/`:

```
edoardo/
├── (existing analysis pipeline for the original 6 dialects, normalized Wiki)
├── _shared_gold_builders/                  # generic CLDF-aware gold builders
├── exp1_uriel_native/                      # NEW dialect set, all in URIEL natively
└── exp2_native_text_original_dialects/     # original 6 dialects, NOT-normalized text + richer gold
```

The shared gold builders parametrically read codes + ISO mappings from a
``varieties.py`` module, so the same code works for any 13-variety setup.

---

## Gold reference set (8 sources)

For both experiments the same 8 sources are built:

| Gold | Captures | Source | Coverage notes |
|---|---|---|---|
| `uriel_*` (×6) | typology, genealogy | URIEL via lang2vec | k-NN imputed for low-resource varieties |
| `glottolog_tree` | genealogy | hand-coded Glottolog 5.x | distinct leaves per language |
| `grambank` | typology (195 features) | Grambank v1.0.3 | Italian dialects partial (scn, lmo, srd) |
| `phoible` | phoneme inventory | PHOIBLE v2.0.1 | Jaccard on phonemes; Italian dialects partial |
| `lexibank` | lexical features | lexibank-analysed v2.1 | broadest dialect coverage |
| `asjp_lexical` | lexical surface (LDND) | ASJP v20 | sc, vec missing |
| `asjp_genealogy` | genealogy (auto-classified) | ASJP v20 classification | independent from Glottolog |
| `geographic_glottolog` | geography | Glottolog 5.x lat/lon | full coverage; Haversine km |

Run all 8 with:

```bash
python -m edoardo._shared_gold_builders.build_all \
    --varieties-module edoardo.exp1_uriel_native.varieties \
    --out-dir edoardo/exp1_uriel_native/gold_references/matrices
```

---

## Experiment 1 — `exp1_uriel_native`

**Variety set**: nap, scn, lij, vec, pms, eml + ita, spa, fra, cat, deu, slv, eng (13).

**Why**: every dialect has a native URIEL entry (no k-NN imputation), so URIEL is a
trustworthy gold for this set.  The trade-off: OLDI and FLORES do not cover
nap/pms/eml, so we must change the test protocol.

**Test protocol**: 80 / 20 hold-out split of Wikipedia per variety.  Centroid
is computed on the held-out 20%.  Methods that depend on parallel pairs
(TLM, MNRL) are not runnable for this experiment because no OLDI exists for
nap/pms/eml.

**Train / Wiki normalization**: two sub-runs (separate output folders):
* `results_normalized/`     — uses `Dataset/wiki/normalized/`
* `results_not_normalized/` — uses `Dataset/wiki/not_normalized/`

**HPC commands** (in order):

1. Add `eml` to the Wiki pipeline (one-time):
   ```bash
   # Verify actual emlwiki page-range at https://dumps.wikimedia.org/emlwiki/2026-04-01/
   # (the URL in Dataset/wiki/scripts/create.py uses a placeholder p1p20000 range)
   $EDITOR Dataset/wiki/scripts/create.py     # confirm or fix eml URL
   sbatch edoardo/exp1_uriel_native/slurm/download_eml_wiki.slurm
   ```

2. Build all gold matrices:
   ```bash
   sbatch edoardo/exp1_uriel_native/slurm/build_golds.slurm
   ```
   ~5 minutes; populates
   `edoardo/exp1_uriel_native/gold_references/matrices/`.

3. Train models on the new dialect set: **NOT YET WRITTEN AS SLURM** — see
   "Training new models" section below.

4. Run the analysis pipeline:
   ```bash
   python -m edoardo.analysis.run_all \
       --gold-dir edoardo/exp1_uriel_native/gold_references/matrices \
       --out-dir  edoardo/exp1_uriel_native/results_normalized
   ```

---

## Experiment 2 — `exp2_native_text_original_dialects`

**Variety set**: fur, lij, lmo, sc, scn, vec + 7 standards (the **same** 13 as the
parent `edoardo/` analysis).

**Why**: same dialects as your already-trained models, but with a richer gold
set (8 sources instead of 2-3) AND a re-evaluation on the *not-normalized*
Wikipedia variant.  Pretrained encoders (XLM-R, CANINE, Sentence-MiniLM) want
native cased+accented text — feeding them lowercase-ASCII destroys their
sub-word vocabulary.  This experiment re-runs them on the appropriate text.

**No re-training needed** for the gold-correlation step: existing model
outputs under `analysis/<method>/experiments/<exp>/method_outputs/` are
re-correlated against the new gold matrices.  If you want to also re-train
the pretrained encoders on the *not-normalized* Wiki, that's an additional
step (see Training section).

**HPC command** (one-shot, ~10 minutes):

```bash
sbatch edoardo/exp2_native_text_original_dialects/slurm/run_exp2.slurm
```

Outputs land in `edoardo/exp2_native_text_original_dialects/results/`.

---

## Training new models for experiment 1

Methods that **don't need OLDI** (so they work for nap/pms/eml):

| Method | Trainable on Wiki only? |
|---|---|
| TF-IDF (char/word) | YES |
| Word2Vec | YES |
| FastText | YES |
| canine MLM | YES |
| xlmr MLM | YES |
| sentence_minilm baseline (no fine-tune) | YES |
| canine TLM / MLM-then-TLM | NO (no OLDI for nap/pms/eml) |
| xlmr TLM / MLM-then-TLM | NO |
| sentence_minilm MNRL | NO |

To retrain for exp1 you need to:

1. Update `analysis/_shared/varieties.py::VARIETY_CODES` to the exp1 list,
   OR set up an experiment-specific override.
2. Run each method's existing `run.py` with that override.
3. Make sure `WIKI_VARIETY_DIR` points to either `wiki/normalized/` or
   `wiki/not_normalized/` for the relevant sub-run.

The cleanest path is to write **per-experiment run wrappers** under
`analysis/<method>/experiments/exp1_uriel_native_normalized/run.py`
(and `..._not_normalized/run.py`).  Each just imports its training logic
and runs it with the exp1 variety set.

These wrappers are **not yet generated** — let me know when you want them
and I'll produce one per (method × normalization variant).

---

## Training re-runs for experiment 2 (optional)

For pretrained encoders only (XLM-R, CANINE, Sentence-MiniLM, LaBSE).  Each
re-run is the same as the existing `analysis/<method>/experiments/<exp>/run.py`
but reading from `Dataset/wiki/not_normalized/` instead of normalized.

The existing code already supports this via `WIKI_VARIETY_DIR_NATIVE`
(see `analysis._shared.varieties`) — there's a one-line switch in each
method's `core/data_loader.py`.

If you want me to produce per-method "not_normalized" SLURM jobs, ask.

---

## Reading the results

Once `results/` is populated for an experiment, the contents are the same
shape as the parent `edoardo/results/`:

| File | Content |
|---|---|
| `correlation_with_gold.csv` | per-(model, gold) Spearman + Mantel |
| `correlation_with_gold_pivot.csv` | wide table — paper Table 1 |
| `cluster_agreement.csv` | ARI / NMI vs genealogical labels |
| `cka_baseline_vs_finetuned.csv` | linear CKA between model pairs |
| `shift_analysis.csv` | Δρ per gold for every (A, B) pair |
| `summary_leaderboard.csv` | one-glance summary |

For exp1 the *current* coverage is:
- gold matrices: built ✓
- model outputs: NOT YET (training step pending)
- analysis: will work once models exist

For exp2:
- gold matrices: build via SLURM
- model outputs: ALREADY exist (same as parent edoardo)
- analysis: runs immediately
