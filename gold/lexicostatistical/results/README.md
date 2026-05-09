# Correlations against gold reference matrices

This folder contains one CSV per gold reference matrix.  Each CSV reports
how well every model output (under ``analysis/<method>/experiments/<exp>/...``)
matches that gold, using two Spearman rank correlations:

| Column | What it measures |
|---|---|
| `Spearman ρ (full matrix)` | Spearman correlation on the FULL upper triangle of the shared distance matrix.  For 13 varieties this is 78 unordered pairs; it summarises how well the model recovers ALL pairwise relationships at once. |
| `Spearman ρ (dialect ↔ external)` | Spearman restricted to the cross-block of (dialect × external-non-Italian) pairs.  For 6 dialects × 6 external languages this is 36 pairs.  It excludes intra-dialect pairs and pairs involving standard Italian, isolating the harder genealogical-boundary-crossing signal. |

Range for both: −1 (anti-correlated) … 0 (random) … +1 (identical ordering).
ρ ≥ 0.7 is strong, 0.4–0.7 moderate, < 0.3 weak / noise.

Sub-variety roles are defined in ``gold/lexicostatistical/varieties.py``:
``dialect`` ∈ {fur, lij, lmo, sc, scn, vec};
``italian`` = {ita} (excluded from the dialect↔external column);
``external`` ∈ {fra, spa, cat, deu, slv, eng}.

When the variety set grows (more Wikipedia languages added), update
``varieties.py``, regenerate the gold matrices via the ``rebuild_*.slurm``
jobs, and rerun ``correlate_against_gold`` — the metric definitions and
column names stay the same.

## Files

* `correlation_lexicostat_ldnd.csv`

## How to interpret a result

Look at one model row.  If `Spearman ρ (full matrix)` is high (e.g. 0.8)
the model captures the OVERALL similarity structure well.  If
`Spearman ρ (dialect ↔ external)` is also high, the model handles the
dialect-to-external-language relations specifically.  A model can score
high on the full matrix but low on the dialect↔external column when it
is good at intra-dialect distinctions but poor at relating dialects to
the standard languages around them — that is precisely the kind of
imbalance worth flagging in the paper.
