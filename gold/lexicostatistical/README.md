# Lexicostatistical gold matrix

Team-curated lexicostatistical reference for evaluating embedding models
on Italian-dialect varieties.

## Method

1. Pull the Swadesh-207 wordlist for our 17 varieties from English
   Wiktionary appendices (Italian-languages, Romance, Germanic, Slavic,
   Uralic).
2. Split rows that are semantically gendered (he/she/it, this, that,
   they, his/her) into masculine + feminine entries.  Other multi-form
   cells reduce to the first form.
3. Map each form to a simplified phonetic alphabet (ASJPcode, Wichmann
   et al. 2009) — 7 vowels and 27 consonants on ASCII.
4. Compute two pairwise distance matrices: the mean per-concept
   normalised Levenshtein distance (`lev_mean`) and its chance-baseline
   normalisation LDND (Levenshtein Distance Normalised Divided), each
   with bootstrap CIs on the per-pair distance. LDND is the gold cited
   in the paper.

LDND is the chance-baseline-normalised version of mean Levenshtein
distance:

```
LDND(A, B) = LD(A, B) / E(A, B)
LD = mean over concepts c of  lev(A_c, B_c) / max(|A_c|, |B_c|)
E  = same statistic on MISMATCHED concepts (chance-similarity baseline)
```

LDND ∈ [0, ~1.0]: closer to 0 means lexically very similar; closer to 1
means same as random pair.

## Files

```
varieties.py                 canonical variety codes + role assignment
fetch_swadesh.py             scrape Wiktionary appendices → wordlists/
expand_gendered.py           split rows into masc/fem versions
build_ldnd.py                ASJPcode + Levenshtein + LDND + bootstrap
wordlists/
    wordlist_swadesh207.csv         207-row raw Wiktionary table (17 variety cols + 2 meta)
    wordlist_swadesh207_split.csv   214 rows after expanding m/f-gendered concepts
matrices/
    lexicostat_ldnd.npz             LDND distance matrix (17×17) — the gold cited in the paper
    lexicostat_ldnd.csv             same, human-readable
    lexicostat_lev_mean.npz         mean per-concept normalised Levenshtein distance (17×17)
    lexicostat_lev_mean.csv         same, human-readable
    lexicostat_asjpcode.csv         audit: native orth → ASJPcode per cell
```

## How to rebuild

The pipeline is reproducible from Wiktionary in three steps:

```bash
# 1. download wordlist from Wiktionary
python -m gold.lexicostatistical.fetch_swadesh \
    --out gold/lexicostatistical/wordlists/wordlist_swadesh207.csv

# 2. split gendered rows
python -m gold.lexicostatistical.expand_gendered \
    --in  gold/lexicostatistical/wordlists/wordlist_swadesh207.csv \
    --out gold/lexicostatistical/wordlists/wordlist_swadesh207_split.csv

# 3. build LDND with 1000 bootstraps + 8-core parallel
python -m gold.lexicostatistical.build_ldnd \
    --wordlist gold/lexicostatistical/wordlists/wordlist_swadesh207_split.csv \
    --out-dir  gold/lexicostatistical/matrices \
    --bootstrap 1000 \
    --n-jobs   8
```

The full rebuild takes ~10 minutes on 8 cores; the bootstrap dominates
the cost.

## When the variety set grows

Update `varieties.py` and re-run all three steps. The gold matrix
shape will grow accordingly, and the correlation pipeline in
`evaluation/correlate_against_gold.py` adapts automatically because it
restricts every model output to the codes present in the gold's labels.

## Citations to use in the paper

* Wichmann, S., Holman, E. W., & Brown, C. H. (2010). *Sound Symbolism in
  Basic Vocabulary*. Entropy 12(4), 844-858. (LDND introduction)
* Wichmann, S., Müller, A., Velupillai, V., et al. (2009). *The ASJP
  Database*.
* Pellegrini, G. B. (1977). *Carta dei dialetti d'Italia*. Pacini.
* Maiden, M., & Parry, M. (eds.). (1997). *The Dialects of Italy*. Routledge.
