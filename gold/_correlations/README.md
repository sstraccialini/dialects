# Correlations against gold reference matrices

One CSV per gold reference matrix.  Columns:

| Column | Meaning |
|---|---|
| `Spearman ρ (full matrix)` | Spearman on the full upper triangle of the shared distance matrix. |
| `Spearman ρ (dialect ↔ external)` | Spearman on the cross-block (dialect × external-non-Italian) only. |

Range −1 … +1.  Per-experiment versions of the same metrics are written
inside every method's `evaluation_results/.../gold_correlations.csv`.

## Reading per gold type

* **Lexicostatistical (LDND on Swadesh-207).**  High ρ = the model
  recovers lexical-cognate similarity.  Central metric for language
  similarity.

* **Geographic (Haversine).**  A *language-aware* model is expected to
  score MODERATELY on the full matrix and NEAR-ZERO or NEGATIVELY on the
  dialect↔external block — Slovenian is geographically near Veneto but
  linguistically Slavic.  A negative ρ on dialect↔external for the geo
  gold is a positive signal that the model captures language rather
  than geography.

## Sub-variety roles

Defined in ``gold/lexicostatistical/varieties.py``:
``dialect`` ∈ {fur, lij, lmo, sc, scn, vec};
``italian`` = {ita} (excluded from the dialect↔external column);
``external`` ∈ {fra, spa, cat, deu, slv, eng}.

## Files

* `correlation_<gold>.csv` — point estimates only (Spearman ρ + d↔e).
* `correlation_<gold>_with_pvalue.csv` — same + Mantel two-sided p-values
  from `evaluation/mantel_pvalues.py` (B = 10 000 permutations).
* `correlation_<gold>_with_bootstrap.csv` — ρ + 95 % bootstrap CI + Mantel p,
  built by `evaluation/aggregate_bootstrap.py` after every method's
  per-experiment `bootstrap_results.csv` is present.

## Reproducing the bootstrap layer

Each method has a `bootstrap.py` under `analysis/<method>/core/`.  It
**loads the already-trained model** (no retraining) from
`method_outputs/models/`, embeds every FLORES sentence, and runs B = 1000
stratified resamples to write
`<exp>/evaluation_results/flores/centroid/bootstrap_results.csv`.

```
# Shallow (run wherever the trained model exists):
python -m analysis.word2vec.core.bootstrap  --experiment word2vec_wikiOLDI_native
python -m analysis.word2vec.core.bootstrap  --experiment word2vec_wikiOLDI_normalized
python -m analysis.fasttext.core.bootstrap  --experiment fasttext_wikiOLDI_native
python -m analysis.fasttext.core.bootstrap  --experiment fasttext_wikiOLDI_normalized

# Deep (HF cache + checkpoint required — typically HPC with GPU):
python -m analysis.canine.core.bootstrap    --experiment canine_zeroshot_native
python -m analysis.canine.core.bootstrap    --experiment canine_finetuned_wikiOLDI_dialects_native
python -m analysis.multilingual_xlmr.core.bootstrap --experiment xlmr_zeroshot_native
python -m analysis.multilingual_xlmr.core.bootstrap --experiment xlmr_finetuned_wikiOLDI_dialects_native
python -m analysis.labse.core.bootstrap     --experiment labse_zeroshot_native
python -m analysis.labse.core.bootstrap     --experiment labse_finetuned_oldi_dialects_native

# Once every bootstrap_results.csv exists, merge into final tables:
python -m evaluation.aggregate_bootstrap
```

The bootstrap is **stratified by variety**: within each language, the
N_var FLORES sentences are resampled with replacement N_var times.  A
parallel bootstrap (resampling FLORES rows) was rejected because
sentence drop-out (e.g. word2vec OOV-only sentences) breaks alignment.
