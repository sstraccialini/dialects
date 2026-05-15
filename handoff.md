# Session handoff — 2026-05-11

## Goal we're working toward

Reframe the project from "does fine-tuning help/hurt?" to **"what dimension of
linguistic similarity does each model capture, and how does fine-tuning shift
it?"** for 6 Italo-Romance dialects (fur, lij, lmo, sc, scn, vec) plus 11
standards (ita, fra, spa, cat, por, oci, deu, eng, slv, hrv, hun).

The mechanism is a 3-way **gold-matrix triangulation**:

1. **Lexicostatistical LDND** — Wiktionary Swadesh-207 lists, ASJPcode-normalised
   Levenshtein. Captures surface lexical similarity / cognates.
2. **Geographic Haversine** — great-circle distance between Glottolog languoid
   centroids (or capital cities for standards), normalised by max off-diagonal.
3. **Historical-influence top-3 sets** — hand-curated per-dialect documented
   influences, scored with set-based **Mean Precision@3** (baseline 0.300).

Every model produces a 17×17 cosine-distance matrix from FLORES centroids;
we correlate it (Spearman ρ, full matrix + dialect↔external block) against
golds 1 and 2, and compute Mean P@3 against gold 3.

## Current state of the code

- Variety set is **17** (final, after dropping `ron`/`glg`/`sqi` from logic;
  their Wiki CSVs stay on disk, gitignored intentionally).
- 3 gold matrices regenerated on the 17-variety set; CSV mirror of the
  geographic matrix added at `gold/geographic/matrices/geographic_haversine.csv`.
- Per-experiment auto-write hook in `evaluation/evaluation.py::run_evaluation`
  writes `gold_correlations.csv` (Spearman vs the 3 matrices) **and**
  `historical_influence.csv` (per-dialect P@3 + MEAN row) into every
  `evaluation_results/<test>/centroid/`.
- Cross-method aggregators (`evaluation/correlate_against_gold.py`,
  `evaluation/check_historical_influence.py`) now scan **only**
  `analysis/<method>/experiments/`, ignoring `old_experiments/`.
- Aggregated rankings in:
  - `gold/_correlations/correlation_lexicostat_ldnd.csv` (13 rows)
  - `gold/_correlations/correlation_lexicostat_lev_mean.csv` (13 rows)
  - `gold/_correlations/correlation_geographic_haversine.csv` (13 rows)
  - `gold/historical_influence/results/historical_influence_summary.csv` (13 rows)
  - `gold/historical_influence/results/historical_influence_detail.csv` (72 rows)
- **14 active model_id × variant** combinations covered:
  TF-IDF (char + word) × {native, normalized} = 4; fastText × {native, normalized} = 2;
  word2vec × {native, normalized} = 2; CANINE × {zero-shot, fine-tuned} = 2;
  XLM-R × {zero-shot, fine-tuned} = 2; LaBSE × {zero-shot, fine-tuned} = 2.
- Wiki 100k sub-sampled CSVs live in the repo (Federico pushed them after
  relaxing `.gitignore`). The full original Wiki CSVs are NOT recoverable from
  this repo state — check `Dataset_archive/` locally for backups.

## Files actively edited this session

- `evaluation/evaluation.py` — added the historical-influence block in the
  per-experiment hook (right after the gold-matrix block, ~line 1040). Adds
  `historical_influence_path` to the return dict.
- `evaluation/correlate_against_gold.py` — `_iter_distance_csvs` now scans
  only `experiments/`.
- `evaluation/check_historical_influence.py` — same `_iter_distance_csvs`
  fix; default `--out-dir` stays at `gold/historical_influence/results/`.
- `gold/historical_influence/influences.csv` — fixed `hr` → `hrv` for `fur`
  and `vec` (typo: our variety code is `hrv`, never `hr`).
- `gold/geographic/matrices/geographic_haversine.csv` — created (CSV export
  of the `.npz`, for human readability).
- (Earlier in same session, pre-compaction)
  `analysis/_shared/varieties.py`, `gold/lexicostatistical/varieties.py`,
  `gold/geographic/varieties.py`, `gold/lexicostatistical/fetch_swadesh.py`,
  `gold/lexicostatistical/build_ldnd.py`,
  `gold/lexicostatistical/expand_gendered.py` — all reduced to the 17-variety
  set.

## Things tried that failed (avoid repeating)

1. **`git restore Dataset/wiki/`** to resolve a pull conflict — destroyed the
   user's locally-extracted full Wiki CSVs (~7 GB). Never use `git restore`
   on data folders without verifying they aren't the only copy. Federico's
   100k versions are the current source of truth.
2. Centralising historical-influence outputs into `gold/_correlations/` —
   user reverted, prefers per-experiment auto-write alongside
   `gold_correlations.csv`. Don't repeat.
3. `python -c "snapshot_download(...)"` for HF cache pre-download without
   `source slurm/tools/env.sh` first — the env sets
   `HF_HOME=$HOME/ltp_hf_cache`. Without sourcing, files land in
   `~/.cache/huggingface` and SLURM jobs can't see them.
4. Submitting CANINE fine-tune (`12_canine_finetuned_*`) with stray
   `--time=00:14:00` from a debug run — TIMEOUT'd at 14:20. Use a real
   wall-time (`--time=02:30:00` on `medium_gpunew` is enough; the H100 finishes
   the 18h-declared job in ~1h15min real).
5. SSH to `login.unibocconi.it` from outside Bocconi network — timeout.
   Correct host is `login.hpc.unibocconi.it` (alias `hpc` in `~/.ssh/config`).
6. Cyberduck download of the LaBSE model dir — silently truncated at 1.2 GB
   of 1.9 GB. Use `rsync -avzP --partial` from the Mac terminal instead.
7. `git add --dry-run` mistakenly used in place of real `git add` —
   commit silently did nothing, push got rejected for unrelated reason
   (origin had advanced), and the duplicate `git pull` brought in Federico's
   commits without ever staging the user's. Verify with `git status` before
   committing.

## Instructions / preferences user gave me

- **User runs all git commits and pushes themselves**. I print the commands;
  I never execute them. (Stored in memory under `feedback_commits`.)
- Per-experiment outputs (one CSV per gold) > one giant aggregate CSV.
- For `historical_influence`, prefer auto-written per-experiment file
  alongside `gold_correlations.csv`, **not** a centralised location.
- For the influence gold (`influences.csv`): only specific edits — `oc → oci`
  everywhere it appeared, `ar → cat` for `scn`. Everything else stays.
- Final variety set: 17. Romanian / Galician / Albanian removed from logic.
  Their Wiki CSVs stay on disk untouched.
- Cross-method aggregators should consider only `experiments/`, not
  `old_experiments/`. The legacy folders exist for archival reference.
- Model weights (gigabyte-scale) live under `analysis/**/method_outputs/models/`
  which is gitignored. They are shared with teammates out-of-band (OneDrive,
  rsync), never via git.

## Paper: starting point for the gold-matrix comparison

A short results subsection, before any deep analysis, can look like this:

> **Triangulating model representations with three reference golds.**
> We score each model's FLORES-centroid 17×17 cosine-distance matrix against
> three independent gold references: a lexicostatistical LDND matrix derived
> from Swadesh-207 Wiktionary wordlists (lexical similarity), a Haversine
> matrix over Glottolog languoid centroids (geographic proximity), and a
> hand-curated set of historical influences scored as Mean Precision@3.
> Two complementary Spearman scores are reported for the matrix golds:
> ρ on the full upper triangle, and ρ restricted to the dialect↔external
> block (excluding standard Italian and intra-dialect pairs), which isolates
> genealogy-crossing relationships from the trivial "Italian-anchored"
> baseline.

### Skeleton of the results table (14 model_id × variant)

| Model | ρ LDND (full / d↔e) | ρ LEV (full / d↔e) | ρ Geo (full / d↔e) | P@3 |
|---|---|---|---|---|
| TF-IDF char wikiOLDI native | 0.905 / 0.851 | 0.921 / 0.879 | 0.272 / −0.150 | 0.389 |
| TF-IDF char wikiOLDI normalized | 0.881 / 0.813 | 0.889 / 0.822 | 0.323 / −0.087 | 0.333 |
| TF-IDF word wikiOLDI native | 0.794 / 0.720 | 0.813 / 0.754 | 0.369 / 0.137 | 0.500 |
| TF-IDF word wikiOLDI normalized | 0.788 / 0.734 | 0.805 / 0.770 | 0.363 / 0.126 | 0.444 |
| fastText wikiOLDI native | 0.867 / 0.776 | 0.870 / 0.779 | 0.254 / −0.220 | 0.389 |
| fastText wikiOLDI normalized | 0.874 / 0.812 | 0.872 / 0.806 | 0.244 / −0.230 | 0.389 |
| word2vec wikiOLDI native | 0.841 / 0.703 | 0.844 / 0.711 | 0.383 / −0.012 | 0.389 |
| word2vec wikiOLDI normalized | 0.851 / 0.778 | 0.853 / 0.786 | 0.372 / −0.112 | 0.444 |
| CANINE zero-shot | 0.665 / 0.700 | 0.686 / 0.751 | 0.164 / −0.027 | 0.389 |
| CANINE fine-tuned (MLM wikiOLDI) | 0.725 / 0.803 | 0.738 / 0.826 | 0.140 / −0.206 | 0.333 |
| XLM-R zero-shot | 0.244 / 0.263 | 0.274 / 0.313 | 0.095 / −0.166 | 0.333 |
| XLM-R fine-tuned (MLM wikiOLDI) | 0.055 / 0.382 | 0.089 / 0.427 | 0.012 / 0.286 | 0.278 |
| LaBSE zero-shot | −0.070 / 0.266 | −0.049 / 0.296 | −0.110 / −0.106 | 0.333 |
| LaBSE fine-tuned (MNRL OLDI) | −0.021 / 0.327 | −0.012 / 0.295 | 0.006 / 0.139 | 0.222 |

(numbers from `gold/_correlations/correlation_*.csv` and
`gold/historical_influence/results/historical_influence_summary.csv` at HEAD.)

### Three observations to make the paper draft

1. **Shallow lexical models dominate the lexicostatistical gold.** TF-IDF
   character-n-grams reach ρ ≈ 0.90 on LDND, with fastText a close second.
   This is expected — both reduce to surface character/word statistics, which
   is precisely what LDND measures.

2. **Geographic gold separates language-aware from geography-aware models.**
   A *positive* geography score on the full matrix combined with a
   *near-zero or negative* score on the dialect↔external block is the
   "good" signal: it means the model groups by genealogy, not by physical
   proximity. Slovenian is geographically near Veneto but linguistically
   Slavic — a language-aware model should put it far from `vec`. All shallow
   models and CANINE behave this way; LaBSE/XLM-R cluster less by either.

3. **Fine-tuning shifts the captured dimension.** CANINE's LDND ρ climbs
   from 0.665 (zero-shot) to 0.725 (fine-tuned on Wiki+OLDI dialects) —
   continued character-MLM does push representations toward lexical
   similarity. XLM-R goes the *other* way (0.244 → 0.055), suggesting that
   3 epochs of dialect-only MLM degrades general-purpose lexical
   alignment rather than reinforcing it. LaBSE is a counter-example: MNRL
   on OLDI ita↔dialect pairs collapses representations to "Italian-like",
   driving LDND further from useful. The paper's narrative is **fine-tuning
   moves models along these three axes in predictable but not always
   beneficial directions** — the gold triangulation is what reveals the
   direction of motion.

### What's missing before the section is publishable

- **Bootstrap CIs / Mantel p-values** on the Spearman scores. Bootstrap is
  already wired into the LDND build (`rebuild_lex_ldnd.slurm`,
  `--bootstrap 1000`); the same can be wrapped around the per-model
  correlation step but is not yet implemented.
- A short **methodology box** describing why each gold is independent and
  what a high/low score *means* for each (already drafted as text in
  `gold/_correlations/README.md`).
