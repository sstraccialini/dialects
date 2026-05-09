# Lexicostatistical reference matrix

A team-curated parallel wordlist for the 13 varieties of Experiment 2:
`fur, lij, lmo, sc, scn, vec` (six Italian dialects) and
`ita, fra, spa, cat, deu, slv, eng` (seven standards).

## Why we are building this

No published cross-linguistic resource (URIEL, Grambank, PHOIBLE, Lexibank,
ASJP) provides differentiated, non-imputed distance values for the six
Italian dialects in our setup; see Limitations section of the paper.
We therefore build a small **lexicostatistical** reference, following the
ASJP / Goebl tradition: pick a fixed list of basic concepts, find the form
each variety uses, transliterate into a simplified phonetic alphabet,
compute pairwise normalised Levenshtein distance, and normalise by the
chance baseline (LDND).

## Methodology

1. **Concept list**: starts with the 40-item ASJP Swadesh subset (Wichmann
   et al., 2010) and may be extended to 100 / 200 items.
2. **Sub-variety choice** (fixed and documented):
   * `fur` — Central Friulian (Udine koiné)
   * `lij` — Genoese
   * `lmo` — Milanese (Western Lombard)
   * `sc`  — Logudorese Sardinian
   * `scn` — Palermo-area Sicilian
   * `vec` — Central Venetian (Venice / Padua)
3. **Form selection rule**: the most neutral, register-free form attested
   in the chosen sub-variety.  When multiple forms are common, the leftmost
   form in the cell is the one used for distance computation; alternates
   appear in `notes`.
4. **Orthography**: native orthography is used in the input table; a
   conversion routine in `build_lex_matrix.py` maps to ASJPcode (Wichmann
   et al.) for distance computation.
5. **Confidence**: every cell carries a confidence flag
   (`H`/`M`/`L`/`?`) so the team can prioritise verification.
6. **Sources**: cells should be verified against at least one of:
   * Wiktionary (per-language entry)
   * Manzini-Savoia, *I dialetti italiani* (linguistic database)
   * AIS — *Sprach- und Sachatlas Italiens und der Südschweiz*
   * Loporcaro, *Profilo linguistico dei dialetti italiani*
   * Pellegrini, *Carta dei dialetti d'Italia*

## Files

* `wordlist_v1_asjp40.csv` — the 40×13 table, native orthography, confidence + notes.
* `wordlist_v2_extended.csv` — extension to 100 / 200 concepts (TBD).
* `build_lex_matrix.py` — converts native orthography to ASJPcode,
  computes pairwise normalised Levenshtein distance, LDND, and
  bootstrap 95% CI on the per-pair distance.
* `matrices/` — output `.npz` files compatible with the rest of the
  edoardo/ pipeline.

## Distance computation

Per pair *(A, B)*:

* For each concept *c* in the shared list, compute
  `lev(A_c, B_c) / max(|A_c|, |B_c|)`.
* `LD(A, B) = mean_c lev_norm(A, B; c)` over concepts with both forms present.
* `E(A, B) = mean over (c1 ≠ c2) of lev(A_c1, B_c2) / max(|A_c1|, |B_c2|)`
  — chance-similarity baseline.
* `LDND(A, B) = LD / E` if E > 0, else LD.
* **Bootstrap 95% CI**: sample 1000 times with replacement from the concept
  list, recompute LDND, take 2.5%/97.5% percentiles.

## Validation strategy (must do before publication)

1. Independent recoding by a second team member; check inter-rater
   agreement on the dialect cells.
2. Have a Romance-philology PhD or native speaker of each dialect verify
   ~10 random cells per dialect.
3. Cross-check against ASJP for the four ASJP-covered dialects (`fur`,
   `lij`, `lmo`, `scn`) — Mantel-correlation between our LDND submatrix
   and the ASJP LDND submatrix should be > 0.7 if our coding is sound.

## Caveats

* The first draft was generated as a starting point; **every dialect cell
  must be reviewed by a competent speaker / dialectologist before paper
  inclusion**.
* Single sub-variety per dialect: variation within (e.g.) Lombard is
  considerable — Eastern Lombard differs from Western Lombard at many of
  these basic items.  Document the sub-variety choice in the paper.
* LDND captures lexical surface only; cognates with severe phonological
  evolution (e.g. Latin /aqua/ → Sicilian /akkwa/ vs French /o/) get a
  high distance even though they are the same etymological word.  This is
  a known limitation of the Goebl/Levenshtein approach.
