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

* `correlation_geographic_haversine.csv`
* `correlation_lexicostat_ldnd.csv`
