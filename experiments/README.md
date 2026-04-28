# Modeling Historical and Linguistic Relations Between Italian Dialects and Contemporary Languages Through Embedding Spaces

**Group 3 — 20879 Language Technology, Bocconi 2026**

This repository contains the code, data, and results for the project investigating the use of NLP methods (TF-IDF, Word2Vec, FastText, multilingual transformers) to model variety similarity among Italian dialects and neighbouring languages, with comparison against linguistic ground truths.

---

## TL;DR

We tested whether NLP-based methods can recover linguistically meaningful similarity among Italian dialects and contemporary languages. We found:

1. **Standard pretrained multilingual transformers (XLM-R, LaBSE, MiniLM) do not recover lexicostatistic ground truth** (Spearman correlation against ASJP: ρ ≈ +0.08 to −0.19, all non-significant). Their distance space is dominated by pretraining-coverage bias rather than linguistic relatedness.

2. **Classical methods (TF-IDF, Word2Vec, FastText, BPE-TFIDF) correlate strongly with linguistic ground truth** (ρ ≈ +0.49 to +0.81 across two independent datasets, ASJP and NorthEuraLex).

3. **Continued masked-LM pretraining on dialect Wikipedia does not fix the transformer misalignment** (ρ goes from +0.08 to −0.17, slightly worse). The asymmetric gradient regime — Italian standard being already well-modelled — prevents convergence.

4. **Contrastive fine-tuning on FLORES+ italian↔dialect parallel pairs DOES fix it** (ρ rises from +0.08 / −0.17 to **+0.71 against ASJP** and **+0.59 against NorthEuraLex**, comparable to or exceeding classical methods). The improvement generalises to held-out sentences (no memorisation).

5. The relevant ingredient is **the explicit alignment objective**, not model size or pretraining sophistication. Training cost: ~3 minutes on a single GPU MIG slice with ~10k parallel pairs.

6. **Lexical contact-driven similarity** (e.g., Sicilian↔Arabic) **is not detectable** through global distance methods on FLORES/Wikipedia (corpora too sparse in dialect-specific borrowed vocabulary). This requires either (a) dedicated dialect corpora or (b) targeted lexicon-based probes with documented borrowings.

---

## 1. Background and Original Proposal

### 1.1 The original research question

The proposal aimed to use word-level and sentence-level embedding spaces to recover (a) genealogical similarity between Italian dialects and contemporary languages, and (b) traces of contact-driven influence (Arabic on Sicilian, Greek on Salentino, Germanic on Friulian/Ladin, Slavic on borderland varieties). The hypothesis was that embedding-based methods would capture historically meaningful similarity beyond orthographic or phonetic overlap.

### 1.2 The honest critique

In retrospect, the proposal contained two methodological assumptions that we found to be naive:

1. **Embedding-based methods would naturally recover linguistic relatedness.** In fact, sentence embeddings are designed for *semantic* similarity, not *linguistic-form* similarity. What variety distances measure in embedding space is the residual language fingerprint after imperfect cross-lingual alignment — a signal that conflates lexical, syntactic, and pretraining-coverage factors.

2. **Contact-driven lexical influence (Arabisms in Sicilian, etc.) would be detectable through global distance computations.** In practice, documented borrowings appear too sparsely in the available corpora (~0.2% of Sicilian Wikipedia sentences contain known Arabisms) to register in centroid-based variety distances.

The project therefore evolved from a "discovery" framing (find new linguistic relations) to a **diagnostic and methodological evaluation** framing: characterising which NLP methods recover linguistic structure for under-described varieties, why some methods fail, and what training objective addresses the failure.

---

## 2. Data

### 2.1 FLORES+ (Meta, 2022)

Parallel evaluation corpus, 16 varieties × 2009 sentences, professionally translated from a shared source. Used as the primary benchmark because the parallel design controls for topic/content confounds.

Varieties used: Veneto, Sicilian, Lombard, Sardinian, Ligurian, Friulian, Ladin, Italian, Spanish, French, Catalan, German, English, Greek, Arabic, Slovenian.

### 2.2 Wikipedia dialect dumps (subsampled)

14 varieties × 16,000 sentences, sampled with random_state=42 for reproducibility. Used to expose models to richer lexical and topical material than FLORES allows. Caveat: dialect Wikipedias are written by small editor communities, often translating from Italian, leading to potential "Italian-in-dialect-orthography" bias.

Varieties: nap, scn, vec, lmo, sc, ita, es, fr, ca, de, el, ar, sl, en.

### 2.3 ASJP database (Holman, Wichmann, et al.)

Linguistic ground-truth dataset. ~5000 languages worldwide, each with ~40 standardised Swadesh-like base concepts transcribed in simplified IPA ("ASJPcode"). The standard tool of computational historical linguistics for two decades.

We extracted 21 varieties: italian, sicilian, lombard, ligurian, friulian, sardinian, neapolitan, piedmontese, emilian, gallurese, plus foreign comparators (spanish, catalan, french, portuguese, romanian, occitan, english, german, greek, arabic, slovenian).

### 2.4 NorthEuraLex (Dehghani-Tafti et al.)

Concept-aligned phonetic database. ~107 Eurasian languages × ~1016 concepts in IPA. Provides ~25× richer phonetic data per language than ASJP, but does not cover Italian dialects (only Italian standard).

We extracted 13 languages: italian, spanish, catalan, french, portuguese, romanian, english, german, greek, slovenian, bulgarian, czech, polish.

### 2.5 Manzini-Savoia corpus (CLDF/JSON v2.0)

A large digitised corpus of Italian dialect descriptions. 62,772 records, 486 dialect points, IPA transcriptions of dialect phrases with parallel Italian glosses, plus POS / morphological / dependency annotation, and word-level alignment between IPA and Italian gloss.

The CSV version contains corrupted encoding ("?" replacing IPA characters) but the JSON preserves UTF-8 IPA correctly. Aggregated to 23 Italian regions for stable distance estimation.

---

## 3. Methods

We tested five families of NLP-based variety distance methods (Phase 1) plus zero-shot perplexity (Phase 2) plus continued pretraining and contrastive fine-tuning (Phases 3, 5). Each method produces an embedding (or a word-form representation) per FLORES sentence; we compute a variety centroid as the L2-normalised mean, and pairwise cosine distance between centroids as the variety-distance matrix.

### Methods

| Family | Method | Folder |
|---|---|---|
| Surface | TF-IDF word | `analysis_flores/tfidf/`, `analysis_wiki/tfidf_baseline/` |
| Surface | TF-IDF char (n-gram 3-6) | same |
| Distributional | Word2Vec skip-gram, joint training | `analysis_*/word2vec/` |
| Distributional + subword | FastText | `analysis_*/subword_fasttext/` |
| Surface + subword | BPE TF-IDF | same |
| Pretrained encoder | XLM-R (`xlm-roberta-base`) | `analysis_*/multilingual*/` |
| Pretrained encoder | MiniLM (`paraphrase-multilingual-MiniLM-L12-v2`) | `analysis_*/sentence_baseline/` |
| Pretrained encoder + alignment | LaBSE | `analysis_*/labse/` |
| Adapted encoder | XLM-R + continued pretraining | `analysis_flores/multilingual_adapted/` |
| Adapted encoder + contrastive | XLM-R + CP + contrastive | `contrastive/` |
| Zero-shot probability | 6 BERT-family models | `analysis_flores/zeroshot_ppl/` |

### Linguistic ground truths

| Source | Coverage | Granularity |
|---|---|---|
| ASJP | 21 varieties (incl. dialects) | 40 IPA concepts |
| NorthEuraLex | 13 European languages | 1016 IPA concepts |
| Manzini-Savoia | 23 Italian regions | ~500–1000 IPA-glossed pairs/region |

For each ground truth, we compute a distance matrix using normalised Levenshtein on shared concepts, and Mantel correlation against the NLP-based matrices.

---

## 4. Phase 1: Multi-method variety similarity

### 4.1 Silhouette scores (FLORES, 16 varieties)

| Method | sil_family | sil_romance |
|---|---:|---:|
| TF-IDF word | −0.011 | +0.015 |
| TF-IDF char | −0.057 | +0.097 |
| Word2Vec | −0.073 | +0.284 |
| FastText | −0.127 | +0.255 |
| BPE TF-IDF | −0.058 | +0.159 |
| MiniLM | +0.276 | +0.252 |
| **XLM-R** | **+0.348** | +0.293 |
| LaBSE | +0.009 | +0.109 |

### 4.2 Silhouette scores (Wiki, 14 varieties)

| Method | sil_family | sil_romance |
|---|---:|---:|
| TF-IDF word | −0.006 | +0.016 |
| TF-IDF char | −0.041 | +0.114 |
| FastText | −0.064 | +0.274 |
| BPE TF-IDF | −0.057 | +0.237 |
| MiniLM | +0.039 | +0.211 |
| LaBSE | −0.053 | +0.102 |

### 4.3 Key observations from Phase 1

- All sane methods recover the macro Romance vs non-Romance distinction (silhouette romance > 0).
- Italo-Romance dialects form a robust internal cluster across methods (Northern Italian: Veneto/Lombard/Ligurian/Friulian/Ladin tightly grouped).
- TF-IDF and other surface methods saturate cross-script: distances between Latin-script Romance and Greek/Arabic become numerically meaningless.
- Pretrained transformers (XLM-R) yield the highest family silhouette but for the wrong reason: they cluster *high-resource pretrained languages* together (Italian standard with Spanish, French, English), and *all dialects* in a separate "non-standard" region. The recovered structure is dominated by pretraining coverage, not linguistic family.
- Continued pretraining (Phase 3 below) reduces the family silhouette artefact but does not fully fix the Italian-dialect distance gap.

### 4.4 Tokenizer artefacts in zero-shot perplexity (Phase 2)

We computed pseudo-perplexity (Salazar et al. 2020) for 16 varieties × 6 BERTs (italian, spanish, catalan, french, german, english). Two tokenizer-specific artefacts emerged:

1. **WordPiece UNK collapse** (BERT family): when the model's tokenizer maps non-Latin script (Arabic/Greek) entirely to `[UNK]`, predicting `[UNK]→[UNK]` is trivial, yielding artificially low perplexity on out-of-script text.

2. **Byte-level BPE collapse** (RoBERTa/CamemBERT): byte-level tokenisers produce no UNK, but multi-byte character sequences are highly predictable byte-by-byte, again yielding artificially low perplexity on out-of-script text.

Both biases produce misleading "high affinity" signals for unrelated languages and were addressed by excluding UNK tokens from the masking process and reporting UNK rates as a diagnostic.

---

## 5. Phase 3: Continued pretraining of XLM-R

### 5.1 Setup

Started from `xlm-roberta-base`. Continued masked-LM pretraining on combined Italian dialect Wikipedia: 6 varieties (nap, scn, vec, lmo, sc, ita), ~96k sentences total, ~3M tokens, 2 epochs, batch 16, lr 5e-5, fp16. Training time on MIG GPU: ~45 minutes.

### 5.2 Results

| Metric | Out-of-the-box XLM-R | Adapted XLM-R |
|---|---:|---:|
| Silhouette family | +0.348 | +0.009 |
| Silhouette romance | +0.293 | +0.125 |
| Italian NN top-3 | spagnolo, francese, inglese | greco, spagnolo, inglese |
| Sardinian NN top-3 | ligure, friulano, lombardo | catalano, francese, spagnolo |
| ρ vs ASJP ground truth | +0.08 | **−0.17** |

### 5.3 Findings

- The adapted model **did not move Italian standard closer to Italo-Romance dialects**. Italian remains in the high-resource European cluster.
- Intra-dialect cluster compaction: Northern Italian dialects became more tightly grouped (vec/lmo/lig/friul/lad pairwise distance dropped 5×).
- Sardinian shifted toward Iberian Romance (consistent with classical descriptions of Sardinian's conservative-Romance character with Iberian lexical affinities).
- ASJP correlation dropped: the adaptation moved the model further from linguistic ground truth.

### 5.4 Why CP does not align Italian with dialects

The asymmetric gradient regime is the cause. During CP:
- Italian tokens were already well-modelled by XLM-R (low loss → small gradients → small parameter movement). Italian's representation barely changes.
- Dialect tokens were OOD (high loss → large gradients → significant parameter movement). Dialect representations adapt strongly.
- Result: dialects compress *internally*, but their region is *not pulled toward* Italian. The gap is not closed.

This is a fundamental limit of standard MLM continued pretraining: it does not enforce inter-variety embedding proximity. An explicit contrastive objective is required.

---

## 6. Phase 4: Linguistic ground truths

### 6.1 ASJP-based variety distances

We extracted 21 varieties from the ASJP CLDF release and computed normalised Levenshtein on shared concepts. Selected pairwise distances (illustrative):

```
italian-spanish:   0.504        sicilian-sardinian: 0.423
italian-romanian:  0.484        sardinian-italian:  0.496
italian-arabic:    0.889        lombard-occitan:    0.560
sicilian-arabic:   0.885        piedmontese-french: 0.590
friulian-german:   0.871        gallurese-romanian: 0.489
```

### 6.2 Cross-family relations (top-3 closest non-Italian-Romance)

| Italian dialect | Top-3 foreign | Sub-family pattern |
|---|---|---|
| Italian | romanian, spanish, portuguese | Romance generic |
| Sicilian | portuguese, romanian, catalan | Iberian + Eastern Romance |
| Sardinian | romanian, portuguese, spanish | Iberian + Eastern Romance |
| Neapolitan | romanian, catalan, portuguese | Iberian + Eastern Romance |
| Friulian | romanian, catalan, spanish | mixed Romance |
| Lombard | **occitan, catalan**, romanian | **Gallo-Romance** |
| Ligurian | **occitan, portuguese**, catalan | Gallo-Romance + Iberian |
| Piedmontese | **catalan, french, occitan** | **Gallo-Romance** |

**Three sub-patterns emerge from the ground truth itself:**
1. Northern Italo-Romance gallo-italici (Lombard, Ligurian, Piedmontese) show stronger affinity with Gallo-Romance (French, Occitan).
2. Central-Southern dialects (Italian, Sicilian, Neapolitan, Sardinian) show stronger affinity with Iberian Romance and Romanian.
3. **No Italian dialect shows distance closer than ~0.79 to non-Romance languages** (Arabic, German, Greek, Slovenian). Contact-driven similarity is not detectable at this metric.

### 6.3 NorthEuraLex distances (1016 IPA words/lang)

Similar pattern at higher resolution for the languages it covers (no Italian dialects). Italian is closest to Spanish (0.64), Portuguese (0.72), Romanian (0.72), Catalan (0.77), French (0.80), then increasingly distant from Slavic and Germanic. Slavic cluster (Bulgarian, Slovenian, Czech, Polish) emerges cleanly. No connection to Arabic (not in NorthEuraLex).

### 6.4 Manzini-Savoia regional distances

Aggregated 486 dialect points to 23 Italian regions, computed Levenshtein-on-IPA distance over shared italian-glossed concepts. Selected results:

- Closest pairs: Abruzzo-Molise (0.30), Lazio-Molise (0.31), Campania-Molise (0.35), **Calabria-Sicilia (0.37)**, Canton Ticino-Lombardia (0.36), **Sardinian everywhere far** (avg distance 0.64).
- The structure recovers the canonical dialectological divisions: Italo-Meridionale Estremo (Sicilia + Calabria), Italo-Meridionale (Campania, Basilicata, Abruzzo, Molise, Lazio sud), Italo-Settentrionale Gallo-italico + reto-romanzo, Sardinian as distinct branch, Tuscany central.

---

## 7. Phase 5: Contrastive fine-tuning (the positive result)

### 7.1 Motivation

Continued pretraining failed to align Italian with its dialects (Phase 3) because MLM does not enforce inter-variety proximity. A contrastive objective on parallel translation pairs *does* enforce it explicitly.

### 7.2 Setup

- Base model: the CP-adapted XLM-R from Phase 3
- Training data: 75% of FLORES+ italian↔dialect parallel pairs, for 7 italo-romance dialects (vec, sic, lmo, sard, lig, friul, lad) — 10,549 positive pairs
- Loss: `MultipleNegativesRankingLoss` (sentence-transformers): pull (italian_i, dialect_i) close, push (italian_i, dialect_j) apart via in-batch negatives
- 3 epochs, batch 32, lr 2e-5, fp16
- Training time: ~3 minutes on MIG GPU

### 7.3 Results: Mantel correlation against linguistic ground truths

| Method | ρ vs ASJP (n=14) | ρ vs NorthEuraLex (n=8) |
|---|---:|---:|
| TF-IDF char | +0.81 *** | +0.56 ** |
| TF-IDF word | +0.81 *** | +0.55 ** |
| BPE TF-IDF | +0.80 *** | +0.49 * |
| Word2Vec | +0.72 *** | +0.52 ** |
| FastText | +0.71 *** | +0.52 * |
| **XLM-R + CP + contrastive** | **+0.71 ***** | **+0.59 *****
| MiniLM | −0.19 NS | +0.41 NS |
| LaBSE | −0.15 NS | +0.26 NS |
| XLM-R | +0.08 NS | −0.01 NS |
| XLM-R + CP only | −0.17 NS | −0.19 NS |

The contrastive model jumps from ρ = +0.08 (out-of-the-box) and ρ = −0.17 (CP-only) to **ρ = +0.71**, matching or slightly exceeding classical methods, on both ASJP and NorthEuraLex.

### 7.4 Held-out validation (no memorisation)

Concern: the original evaluation embedded all 2009 FLORES sentences per variety, including the 1507 sentences seen during contrastive training. We re-evaluated using only the 502 held-out indices per variety:

| Eval | ρ vs ASJP | ρ vs NorthEuraLex |
|---|---:|---:|
| Full FLORES (2009 sentences) | +0.71 | +0.59 |
| Holdout-only (502 sentences, **never seen in training**) | **+0.71** | **+0.56** |

The near-identical correlations confirm that the model **generalises**: the alignment between italian and dialects is a learned regularity, not memorisation of training pairs.

### 7.5 Note on anchor choice

The contrastive setup anchors training on `italian↔dialect` pairs only, not on all-pairwise FLORES translations. This is a deliberate methodological choice:

- **Italian-anchor**: pulls dialects toward Italian standard, leaves foreign languages (Spanish, French, Arabic, etc.) unaffected. Preserves the cross-lingual geometry of non-Italian languages.
- **All-pairwise** (LaBSE-style): would force convergence of *all* translations regardless of linguistic relatedness, flattening the variety-distance space and erasing the very signal we are trying to measure.

---

## 8. Phase 6: Italian dialects vs non-Italian languages

Combining the Mantel-validated ground truths and the recovered embedding spaces, we can now answer the proposal's question concretely:

**Are Italian dialects close to non-Italian languages?**
- Within Romance: yes, robustly. Northern Italian dialects show Gallo-Romance affinity (Lombard, Ligurian, Piedmontese closest to Occitan/French/Catalan). Central-Southern + insular dialects show Iberian + Romanian affinity. These are confirmed by ASJP and the Manzini-Savoia regional distances.
- Across families (to Arabic / Germanic / Slavic): **no**. Distance to Arabic is uniformly ≥ 0.88 across all Italian dialects in ASJP. Contact-driven lexical influence (Arabisms in Sicilian, etc.), while real linguistically, **does not register in global distance methods**. Detection requires either (a) targeted lexicon-based probes with curated borrowing lists, or (b) dedicated dialect corpora richer in domain-specific lexicon than FLORES/Wikipedia. We attempted the targeted probe (50 documented Arabisms, Trovato 2002) and found insufficient corpus coverage (~33/16,000 Sicilian Wiki sentences contain any listed Arabism; ~2/2009 in FLORES).

---

## 9. Discussion

### 9.1 Why classical methods recover linguistic similarity, transformers do not

Classical methods (TF-IDF, Word2Vec, FastText, BPE) work directly on the *form* of observable linguistic data:
- TF-IDF: shared character n-grams capture orthographic / phonological cognate-ness
- Word2Vec / FastText: distributional contexts of words reflect lexical and syntactic regularities
- These are aligned with what lexicostatistic ground truth (ASJP) measures: phonological-lexical cognate similarity.

Pretrained multilingual transformers compute *semantic* representations:
- They are trained to map paraphrases / translations close, not to preserve language-specific form.
- Their pretraining is dominated by high-resource languages; varieties they have not seen become "out-of-distribution" and cluster together as "non-standard" rather than by genealogy.
- The embedding distance between two varieties of equal-content text reflects the residual language-fingerprint after imperfect cross-lingual alignment, plus the OOD bias.

This is consistent with prior findings (Pires et al. 2019, Lauscher et al. 2020) that mBERT/XLM-R cross-lingual transfer is weak for low-resource languages.

### 9.2 Why contrastive on parallel pairs fixes it

Contrastive training on (italian_i, dialect_i) pairs explicitly enforces: "italian sentence i and its dialect translation should occupy the same position in embedding space." Combined with in-batch negatives ("italian_i should be far from dialect_j"), the model learns a directional alignment.

Crucially, this alignment generalises: held-out sentences (~502 indices never seen in training) produce nearly identical correlation with ASJP ground truth as full evaluation (Section 7.4). The model is not memorising; it is learning a transferable representational regularity.

The training cost is minimal: ~3 minutes on a single GPU MIG slice with ~10k parallel pairs. This contrasts with the sophisticated continued-pretraining pipeline (~45 minutes) that did not work, and with the assumption that "more data + bigger model = better representations". For variety similarity, the **objective matters more than the scale**.

### 9.3 The two faces of NLP for dialectology

Our evaluation reveals two distinct uses of NLP for variety analysis:
- **Language identification**: standard transformers work fine here, including for low-resource (the OOD bias is a feature, not a bug, for ID).
- **Variety similarity / phylogeny**: standard transformers fail. Need either classical methods (working on form) or contrastively-trained models (explicitly aligned).

For computational dialectology specifically, the field should consider classical methods more seriously than recent NLP trends suggest, or invest in contrastive variants of multilingual encoders.

---

## 10. Limitations

1. **Small parallel data**: 2009 FLORES sentences / variety. Adequate for centroid computation, marginal for from-scratch training, insufficient for detection of sparse phenomena like specific borrowings.
2. **Dialect Wikipedia is biased**: written by small editor communities, often translating from Italian. Not authentic spoken dialect.
3. **Coverage gaps**: Ladin, Venetian, several Sardinian sub-varieties not in ASJP/NorthEuraLex/WikiPron; Italian dialects entirely absent from NorthEuraLex.
4. **No phonetic/audio modality**: orthographic non-standardisation across dialects is not addressed. Audio (Common Voice + Wav2Vec2) would be a natural extension.
5. **Anchor choice**: Phase 5 uses italian as the contrastive anchor. Other anchors (Spanish, multi-language) would test different hypotheses.
6. **No native-speaker validation**: linguistic perception data would provide a stronger ground truth, but is logistically complex.
7. **Manzini-Savoia coverage limits**: ~130 sentences/dialect-point average is small for individual point analysis. Aggregation to 23 regions is necessary.
8. **Contact-driven analysis remains beyond reach**: detection of Arabic / Greek / Germanic / Slavic substrate via global distance methods on standard corpora is not viable. This is a real methodological limit for the kind of NLP-only "discovery" the original proposal contemplated.

---

## 11. Future Work

1. **Audio modality**: Common Voice has small but usable IPA-free audio for several Italian dialects (scn, vec, lij, lmo, sc, fur). Wav2Vec2-XLSR or HuBERT representations would bypass orthographic confounds and capture phonological similarity directly.
2. **WikiPron + PanPhon phonetic distance**: a lighter "audio without audio" extension using IPA transcriptions and phonetic feature distance.
3. **Targeted contact probe with dedicated corpora**: scrape regional cuisine/folk-tale corpora rich in dialect-specific lexicon, then probe for documented borrowings (Arabic in Sicilian, Greek in Salentine, etc.) via word-level embedding comparison.
4. **Phylogenetic tree reconstruction (NeighborNet)**: from cognate distances, with reticulation to capture contact effects.
5. **Concept Sliders**: parameter-efficient LoRA-based "directional sliders" trained on FLORES parallel pairs as contrastive data, providing a parameter-space measure of variety direction and enabling controllable text generation.
6. **All-pair vs single-anchor contrastive ablation**: confirm that the italian-anchor choice is preferable for our research question.
7. **Multi-source transfer learning evaluation**: take the recovered "best similarity" varieties as starting points for fine-tuning low-resource dialect downstream tasks (POS tagging, NER), test whether similarity rankings predict transfer effectiveness.
8. **Native-speaker perceptual study**: collect dialect-similarity judgements from native speakers of multiple Italian regions and correlate with computational distances.

---

## 12. Repository structure

```
experiments/
├── README.md                                this file
├── flores_data/                             FLORES+ corpus (16 varieties × 2009 sentences)
├── wiki_data/                               Wikipedia subsamples (14 varieties × 16k sentences)
├── MS_corpus_aligned_tagged_v2.0.json       Manzini-Savoia (62k records, 486 dialect points)
├── analysis_flores/                         Phase 1 + 2 + 3 — methods on FLORES+
│   ├── tfidf/                               TF-IDF word + char
│   ├── word2vec/                            Word2Vec joint
│   ├── subword_fasttext/                    FastText + BPE TF-IDF
│   ├── multilingual/                        XLM-R out-of-the-box
│   ├── multilingual_adapted/                XLM-R after continued pretraining (Phase 3)
│   ├── sentence_baseline/                   MiniLM (paraphrase-multilingual)
│   ├── labse/                               LaBSE
│   └── zeroshot_ppl/                        Phase 2 — pseudo-PPL with 6 BERTs
├── analysis_wiki/                           Phase 1 — same methods on Wikipedia data
├── continued_pretraining/                   Phase 3 — CP training pipeline
├── contrastive/                             Phase 5 — contrastive fine-tuning on FLORES+
├── manzini_savoia/                          Phase 4 — ground truth analyses
│   ├── src/compute_phonetic_distance.py     M-S → 23 regions × IPA Levenshtein
│   ├── src/asjp_distance.py                 ASJP download + 21-language distance
│   ├── src/northeuralex_distance.py         NorthEuraLex download + 13-language distance
│   ├── src/mantel_vs_nlp.py                 Mantel(ASJP, NLP-methods)
│   ├── src/mantel_neL_vs_nlp.py             Mantel(NorthEuraLex, NLP-methods)
│   └── src/mantel_compare_full_vs_holdout.py  contrastive memorisation check
└── slurm/                                   HPC SLURM scripts
```

### How to reproduce

1. Set up environment on HPC (login node):
   ```
   cd experiments
   bash slurm/setup_env.sh
   ```
2. Run the full Phase 1 + 2 pipeline:
   ```
   sbatch slurm/run_all.slurm
   ```
3. Run Phase 3 continued pretraining + Phase 5 contrastive:
   ```
   sbatch slurm/14_continued_pretraining_xlmr.slurm
   sbatch slurm/15_flores_multilingual_adapted.slurm   # depends on 14
   sbatch slurm/16_contrastive.slurm
   sbatch slurm/17_eval_holdout.slurm                  # validates Phase 5
   ```
4. Phase 4 ground-truth analyses run locally (no GPU needed):
   ```
   python manzini_savoia/src/compute_phonetic_distance.py
   python manzini_savoia/src/asjp_distance.py
   python manzini_savoia/src/northeuralex_distance.py
   python manzini_savoia/src/mantel_vs_nlp.py
   python manzini_savoia/src/mantel_neL_vs_nlp.py
   ```

---

## 13. Key references

- Holman, E. W., Wichmann, S., et al. (2008). *Advances in automated language classification*. ASJP database.
- Dehghani-Tafti, J., et al. NorthEuraLex.
- Manzini, M. R., & Savoia, L. M. (2005). *I dialetti italiani e romanci: Morfosintassi generativa*. (3 vols.)
- Conneau, A., et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale* (XLM-R).
- Feng, F., et al. (2020). *Language-agnostic BERT Sentence Embedding* (LaBSE).
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT*.
- Salazar, J., et al. (2020). *Masked Language Model Scoring* (pseudo-perplexity).
- Pires, T., et al. (2019). *How Multilingual is Multilingual BERT?*
- Lauscher, A., et al. (2020). *From Zero to Hero: On the Limitations of Zero-Shot Cross-Lingual Transfer with Multilingual Transformers*.
- Lin, Y.-H., et al. (2019). *Choosing Transfer Languages for Cross-Lingual Learning*.
- Gururangan, S., et al. (2020). *Don't Stop Pretraining*.
- Pfeiffer, J., et al. (2020). *MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer*.
- Gandikota, R., et al. (2023). *Concept Sliders: LoRA Adaptors for Precise Control in Diffusion Models* (cited as inspiration for Phase 5 framing).
- Trovato, S. C. (2002). On Sicilian Arabisms.
- Pellegrini, G. B. (1977). *Carta dei dialetti d'Italia*.
- Maiden, M. (2014). *Italian Linguistics*.
