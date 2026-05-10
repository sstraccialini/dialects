# EXPERIMENTAL PLAN — FINAL RUN

> **Scopo**: documento di riferimento PERMANENTE per la riorganizzazione totale e il submit finale degli esperimenti per il paper. Tenere allineato durante l'esecuzione.
>
> **Data inizio piano**: 2026-05-10
> **Stato**: 🟡 In definizione — aspettando dati FLORES/OLDI cleaned definitivi

---

## 1. Variety Registry — 17 Varietà

### 6 Dialetti italo-romance (Group A — OLDI ∩ FLORES)
| Code | Variety | Family |
|---|---|---|
| fur | Friulian | italo_romance |
| lij | Ligurian | italo_romance |
| lmo | Lombard  | italo_romance |
| sc  | Sardinian | italo_romance |
| scn | Sicilian  | italo_romance |
| vec | Venetian  | italo_romance |

### 11 Standard
| Code | Variety | Family |
|---|---|---|
| ita | Italian | italian |
| spa | Spanish | romance |
| fra | French  | romance |
| cat | Catalan | romance |
| por | Portuguese | romance |
| oci | Occitan | romance |
| deu | German  | germanic |
| eng | English | english |
| slv | Slovenian | slavic |
| hrv | Croatian  | slavic |
| hun | Hungarian | uralic |

### Drop confermato
- `ron` (Rumeno), `glg` (Galiziano), `sqi` (Albanese)

---

## 2. Dati

### 2.1 FLORES — cleaned manually by user
- **Path**: `Dataset/flores/cleaned/flores.csv` (singolo CSV con colonne parallele per varietà)
- **Versione normalized da generare**: `Dataset/flores/cleaned_normalized/flores.csv`
  - Applicare `aggressive_normalize()` cella per cella

### 2.2 OLDI — cleaned manually by user
- **Path**: `Dataset/oldi/cleaned/oldi.csv` (idem layout)
- **Versione normalized da generare**: `Dataset/oldi/cleaned_normalized/oldi.csv`

### 2.3 Wiki — kept as-is
- `Dataset/wiki/{normalized,not_normalized}/{dialects_in_both_OLDI_and_Flores,languages}/<code>.csv`
- I CSV già esistenti vanno bene
- Le 3 standard scartate (ron/glg/sqi) restano su disco ma il registro non le carica

### 2.4 Vecchi dati FLORES/OLDI → ARCHIVIATI
- Sposta `Dataset/flores/{normalized,not_normalized}/` → `Dataset_archive/flores/`
- Sposta `Dataset/oldi/{normalized,not_normalized}/` → `Dataset_archive/oldi/`

---

## 3. 12 Esperimenti FINALI

### 3.1 Surface methods (6) — sia norm sia native

Regole:
- Surface methods usano "Wiki + OLDI dialect" come training (vedi 3.4)
- Eval su FLORES con text variant **coerente** col training

| # | Method   | Train data            | Text variant | Eval                            |
|---|----------|-----------------------|--------------|---------------------------------|
| 1 | TF-IDF   | Wiki + OLDI dial only | normalized   | FLORES cleaned normalized       |
| 2 | TF-IDF   | Wiki + OLDI dial only | native       | FLORES cleaned native           |
| 3 | FastText | Wiki + OLDI dial only | normalized   | FLORES cleaned normalized       |
| 4 | FastText | Wiki + OLDI dial only | native       | FLORES cleaned native           |
| 5 | Word2Vec | Wiki + OLDI dial only | normalized   | FLORES cleaned normalized       |
| 6 | Word2Vec | Wiki + OLDI dial only | native       | FLORES cleaned native           |

> **TF-IDF setup CONFIRMED — opzione (b)**: fit su Wiki+OLDI a livello di FRASE (non super-doc), poi transform sulle 1827 frasi FLORES parallele, centroide = media dei vettori per varietà. Coerente con tutti gli altri 11 metodi (stesso eval set parallelo). Cap rule 3.4 applicata. Vecchio approccio super-doc (`old_experiments/wiki_only_superdoc/`) abbandonato: IDF instabile su 13 doc, topic-confound non rimosso, non confrontabile cogli altri.

### 3.2 Pretrained zero-shot (3) — native only, no fine-tuning

| # | Model       | Eval                     |
|---|-------------|--------------------------|
| 7 | LaBSE base  | FLORES cleaned native    |
| 8 | XLM-R base  | FLORES cleaned native    |
| 9 | CANINE base | FLORES cleaned native    |

### 3.3 Pretrained fine-tuned (3) — native only, dialect-only training

| #  | Model  | Loss             | Train data                      | Eval                  |
|----|--------|------------------|---------------------------------|-----------------------|
| 10 | LaBSE  | MNRL contrastive | OLDI parallel ita↔dial (6 dial) | FLORES cleaned native |
| 11 | XLM-R  | MLM monolingual  | Wiki+OLDI dial only (6 dial)   | FLORES cleaned native |
| 12 | CANINE | MLM monolingual char | Wiki+OLDI dial only (6 dial) | FLORES cleaned native |

### 3.4 Regola "Wiki + OLDI" (per FastText/Word2Vec/XLM-R/CANINE)

**Per i 6 dialetti**:
1. Prendi TUTTE le frasi OLDI di quel dialetto (colonna corrispondente, ~6193 frasi)
2. Aggiungi frasi Wiki di quel dialetto fino a un totale di 100k
3. Se OLDI > 100k (mai per i nostri dialetti), tieni TUTTO OLDI senza cap

Esempi:
- **Friulano**: 22k Wiki + 6.2k OLDI = 28k tot, NO cap
- **Lombardo**: 135k Wiki + 6.2k OLDI = 141k tot, CAP a 100k mantenendo TUTTI i 6.2k OLDI + 93.8k Wiki random

**Per le 11 standard** (solo XLM-R/CANINE training Cell 11/12):
- Le standard **NON** entrano nel training di Cell 11/12 (training solo dialetti)
- Per FastText/Word2Vec (Cell 3-6), le standard usano SOLO Wiki, cap 100k. NON includere OLDI delle standard.

### 3.5 LaBSE fine-tuning (Cell 10) — setup specifico

- **Loss**: MNRL contrastive
- **Pairs**: OLDI parallel ita↔dialect (6 dialetti × ~6193 = ~37k pair tot)
- **Source**: `Dataset/oldi/cleaned/oldi.csv` (estrai colonna ita + colonna dialetto come pair)
- **Hyperparams**: 5 epochs, batch 64, lr 2e-5, max_length 128

---

## 4. Hyperparametri (confermati)

| Modello | Loss | Hyperparams |
|---|---|---|
| TF-IDF | n-gram | max_features auto, char + word variants |
| FastText | skip-gram subword | epochs=5, vector_size=200, window=5, min_count=5, sg=1, char n-gram 3–6 |
| Word2Vec | skip-gram word | epochs=5, vector_size=100, window=5, min_count=5, sg=1 |
| XLM-R MLM | continued pretraining | 3 epochs, lr 3e-5, batch 16 × grad_accum 4, max_length 512 |
| CANINE MLM | continued pretraining (char) | 3 epochs, lr 3e-5, batch 8 × grad_accum 8, max_length 512 |
| LaBSE MNRL | contrastive | 5 epochs, lr 2e-5, batch 64, max_length 128 |

---

## 5. Metriche di valutazione

Per ogni esperimento, output in `evaluation_results/flores/centroid/`:
- `silhouette_report.txt` con: `silhouette_family`, `silhouette_romance_vs_rest`, `silhouette_romance_no_dialects`
- `dendrogram.png`, `similarity_heatmap.png`
- `projection_mds.png`, `projection_tsne.png`, `projection_umap.png`
- `distances.csv`, `nearest_neighbors.csv`, `per_variety_profiles.csv`
- `per_variety_plots/<code>.png`
- `clustering_metrics.csv` (DB, CH, ARI, NMI)
- `gold_correlations.csv` (Mantel se reference matrices presenti)

> **Note**: anche `parallel_eval` (sentence-pair alignment) per i metodi che ne traggono valore.
>
> **Re-eval futuro**: tutti i `variety_vectors.npz` vengono salvati. Per cambiare metriche/setup eval senza re-runnare il fine-tuning, lanciare uno script che ri-evalua dal `.npz` (~1 min CPU).

---

## 6. Riorganizzazione cartelle

### 6.1 Naming convention nuova
- **old_experiments/**: nomi descrittivi (es. `canine_fullFT_wiki_dialOnly_norm_100k_2026-05-09`)
- **experiments/**: nomi descrittivi senza prefisso numerico, es:
  - `tfidf_wiki_normalized`
  - `tfidf_wiki_native`
  - `fasttext_wikiOLDI_normalized`
  - `fasttext_wikiOLDI_native`
  - `word2vec_wikiOLDI_normalized`
  - `word2vec_wikiOLDI_native`
  - `labse_zeroshot_native`
  - `xlmr_zeroshot_native`
  - `canine_zeroshot_native`
  - `labse_finetuned_oldi_dialects_native`
  - `xlmr_finetuned_wikiOLDI_dialects_native`
  - `canine_finetuned_wikiOLDI_dialects_native`

### 6.2 Mapping vecchio → nuovo (old_experiments)

| Vecchio path                                                          | Nuovo path in old_experiments/        |
|-----------------------------------------------------------------------|----------------------------------------|
| `analysis/canine/experiments/mlm_wiki_to_flores_oldi`                 | `canine/old_experiments/canine_fullFT_wiki_rehearsal13_native_100k_2026-05-05/` |
| `analysis/canine/experiments/mlm_wiki_dialects_normalized`            | `canine/old_experiments/canine_fullFT_wiki_dialOnly_norm_100k_2026-05-09/` |
| `analysis/canine/experiments/mlm_wiki_dialects_native`                | `canine/old_experiments/canine_fullFT_wiki_dialOnly_native_100k_2026-05-10/` |
| `analysis/canine/experiments/mlm_wiki_rehearsal_normalized`           | `canine/old_experiments/canine_fullFT_wiki_rehearsal13_norm_50k_2026-05-09_edoardo/` |
| `analysis/canine/experiments/lora_wiki_dialects_normalized`           | `canine/old_experiments/canine_LoRA_r16_wiki_dialOnly_norm_50k_2026-05-09/` |
| `analysis/canine/experiments/mlm_wiki_then_tlm_oldi_to_flores`        | `canine/old_experiments/canine_mlmwiki_then_tlmoldi_2026-05-05/` |
| `analysis/canine/experiments/tlm_oldi_to_flores`                      | `canine/old_experiments/canine_tlmoldi_only_2026-05-04/` |
| `analysis/multilingual_xlmr/experiments/mlm_wiki_to_flores_oldi`      | `multilingual_xlmr/old_experiments/xlmr_fullFT_wiki_rehearsal13_native_100k_2026-05-05/` |
| `analysis/multilingual_xlmr/experiments/mlm_wiki_rehearsal_normalized`| `multilingual_xlmr/old_experiments/xlmr_fullFT_wiki_rehearsal13_norm_50k_2026-05-09_edoardo/` |
| `analysis/multilingual_xlmr/experiments/mlm_wiki_dialects_native`     | `multilingual_xlmr/old_experiments/xlmr_fullFT_wiki_dialOnly_native_100k_INCOMPLETE/` |
| `analysis/multilingual_xlmr/experiments/tlm_oldi_to_flores`           | `multilingual_xlmr/old_experiments/xlmr_tlmoldi_only_2026-05-04/` |
| `analysis/multilingual_xlmr/experiments/mlm_wiki_then_tlm_oldi_to_flores` | `multilingual_xlmr/old_experiments/xlmr_mlmwiki_then_tlmoldi_2026-05-05/` |
| `mlm_wiki_dialects_native/` (root, Edoardo first try)                 | `multilingual_xlmr/old_experiments/xlmr_fullFT_wiki_dialOnly_native_50k_2026-05-09_edoardo/` |
| `mlm_wiki_rehearsal_normalized/` (root, Edoardo)                      | `canine/old_experiments/canine_fullFT_wiki_rehearsal13_norm_50k_2026-05-09_edoardo_DUPLICATE/` |
| `mlm_wiki_rehearsal_normalized-1/` (root, Edoardo)                    | `multilingual_xlmr/old_experiments/xlmr_fullFT_wiki_rehearsal13_norm_50k_2026-05-09_edoardo_DUPLICATE/` |

### 6.3 SLURM
- Tutti gli script in `slurm/jobs/` → `slurm/old_jobs/`
- Nuovi script per i 12 esperimenti → `slurm/new_jobs/`

---

## 7. Status checklist

- [ ] **Step 0** — Salva NOTES decisioni in memoria persistente
- [ ] **Step 1** — Aspetta cleaned data definitivo da utente
- [ ] **Step 2** — Genera versione `cleaned_normalized/` (script da scrivere)
- [ ] **Step 3** — Backup `Dataset/{flores,oldi}/{normalized,not_normalized}/` → `Dataset_archive/`
- [ ] **Step 4** — Aggiorna data loaders per leggere il nuovo CSV cleaned formato
- [ ] **Step 5** — Riorganizza folders: experiments/ → old_experiments/, jobs/ → old_jobs/
- [ ] **Step 6** — Scrivi 12 nuovi run.py in `analysis/<method>/experiments/<name>/`
- [ ] **Step 7** — Scrivi 12 SLURM in `slurm/new_jobs/`
- [ ] **Step 8** — Sanity check locale dei nuovi run (smoke test)
- [ ] **Step 9** — Push tutto su git
- [ ] **Step 10** — Submit job HPC

---

## 8. Decisioni metodologiche STABILI (mai più discusse)

1. **Test sempre su FLORES** (parallelo, controllato per topic). Mai test su OLDI (manca per nuove standard).
2. **Pretrained encoders sempre native text** (matches pretraining tokenizer). Aggressive normalize è OOD per loro.
3. **Surface methods**: text variant matching tra train e eval (norm-norm, native-native).
4. **Fine-tuning solo dialetti** per pretrained encoders → preserva struttura cross-lingua delle standard (no catastrophic forgetting).
5. **Cap 100k Wiki per varietà** (per training). Per dialect+OLDI: OLDI tutta + Wiki fino a 100k.
6. **`isotropy=False`** default in evaluation (top-1 PC removal distrugge il segnale Romance).
7. **`silhouette_romance_no_dialects`** come metrica chiave per catastrophic forgetting check.
8. **CSV Wiki troppo grandi per git** (>100MB). Distribuiti via Drive/HF.
9. **TF-IDF a livello frase, non super-doc** (b). Fit su Wiki+OLDI (~1.7M frasi → IDF stabile), transform su FLORES parallele, centroide = media. Confrontabile con gli altri 11 metodi.

---

## 9. Note operative

- Compute nodes HPC **NO internet** → `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` in env vars SLURM
- Pre-download modelli (LaBSE, XLM-R, CANINE) sul login node prima dei job
- `peft` installato nel venv HPC per (eventuali) future LoRA
- Job naming: prefisso `ltp_` per filtrare squeue facilmente
