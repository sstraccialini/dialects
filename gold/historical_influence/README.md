# Historical-influence gold

Per-dialect set of the **top-3 non-dialect varieties** that documented
historical contact / domination is most often associated with.  Sets are
**unordered** — only set membership matters.

This gold is **discrete (set-based)**, not a distance matrix, so it cannot
be evaluated with Spearman ρ.  It is evaluated by
``evaluation/check_historical_influence.py`` which computes a
top-3 set-overlap metric (see below).

## Structure

`influences.csv` has one row per dialect with columns
``dialect, influence_1, influence_2, influence_3, notes``.

The three "influence" columns are codes from
``gold/lexicostatistical/varieties.py::EXTERNAL_CODES`` only —
{{fra, spa, cat, deu, slv, eng}}.  Standard Italian (``ita``) is
**deliberately excluded** from both the gold and the candidate pool: it
is the standard reference for every dialect, so including it would
collapse the metric onto a trivial "is ita the closest?" question.
Order of the three columns is irrelevant (sets, not lists).

When a dialect has historical influences that are NOT in our 13-variety
set (e.g. Greek and Arabic for Sicilian), document them in the ``notes``
column but do not include them in the three columns — they cannot be
tested by any of our models.

## Evaluation metric — Mean Precision@3

For a given model:

1. For each dialect *d*, take the model's distance row, exclude all
   other dialects AND standard Italian (we measure
   dialect-vs-non-Italian-rest), sort the remaining external codes
   ascending by distance, and take the **top 3 closest**.
2. Compute ``|gold(d) ∩ model_top3(d)| / 3`` — fraction of the gold
   set recovered by the model's top-3 predictions.  This is precision
   = recall = F1 since both sets have size 3.
3. Average across the 6 dialects → ``Mean Precision@3``.

Range: 0 (no overlap on any dialect) … 1 (every model top-3 matches
gold exactly).  **Random-chance baseline**: with 6 non-Italian
candidates ({{fra, spa, cat, deu, slv, eng}}) and 3 picks, expected
overlap under uniform random is 3/6 = 0.500.  Beat this clearly to
demonstrate the model captures historical-contact signal beyond chance.

## Output of the evaluator

* ``results/historical_influence_summary.csv`` — one row per model with
  its mean precision@3.
* ``results/historical_influence_detail.csv`` — one row per (model, dialect)
  with the gold set, the model's predicted top-3, and the per-dialect
  precision@3.  Useful for error analysis.

## Why it complements lexicostatistical and geographic gold

* Lex gold (LDND): rewards lexical surface similarity — reflects
  cognate retention but not historical *contact* per se.
* Geographic gold: rewards areal proximity — flat against historical
  contact (e.g. Sardinian and Spain are far apart but Sardinian carries
  heavy Spanish influence).
* **Historical-influence gold**: directly tests whether the model
  recovers the documented contact set, which neither of the other two
  golds captures cleanly.

## How to update

When adding new varieties, edit ``influences.csv`` (one new row per
new dialect) and rerun the evaluator.  Cite your sources for the
chosen top-3 in the ``notes`` column (e.g. Maiden-Parry 1997, Loporcaro,
specific dialectology atlases).
