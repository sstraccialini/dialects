# CANINE MLM Wiki — guida ai risultati

Questa cartella contiene i risultati di **due run distinti** dello stesso
esperimento, ottenuti con setup metodologici diversi.

## Quale cartella corrisponde a cosa

```
evaluation_results/
├── flores/
│   ├── centroid/                ← RUN 1 (vecchio, riferimento — il MIGLIORE finora)
│   ├── centroid_with_iso/       ← RUN 2 (nuovo, con isotropy fix)
│   ├── centroid_no_iso/         ← RUN 2 (nuovo, senza isotropy fix)
│   └── parallel/                ← RUN 1 parallel alignment
└── oldi/
    └── (stessa struttura)
```

## RUN 1 — `centroid/` e `parallel/`

**Setup**:
- Text variant: **normalized** (lowercase ASCII)
- Training set: **6 dialetti italo-romance soltanto**
- Isotropy correction: nessuna
- Continual MLM 3 epoche su CANINE-c

**Numeri**:
- `flores/centroid/`: silhouette family = -0.191, silhouette romance = **+0.598** ★
- `oldi/centroid/`:   silhouette family = -0.201, silhouette romance = **+0.575** ★

★ = miglior risultato silhouette romance dell'intera suite.

## RUN 2 — `centroid_with_iso/` e `centroid_no_iso/`

**Setup**:
- Text variant: **not_normalized**
- Training set: **tutte 13 varietà** (rehearsal-style), cap 100k frasi cad.
- Continual MLM 3 epoche

**Le due varianti differiscono solo per l'evaluation**:

| Sotto-cartella | Isotropy | sil_rom (FLORES) | Note |
|---|---|---:|---|
| `centroid_with_iso/` | top-1 PC removal | -0.036 | iso fix distrugge segnale — non usare |
| `centroid_no_iso/` | nessuna | +0.217 | versione corretta del run 2 |

## Regressione importante: RUN 2 < RUN 1

```
                        sil_family   sil_romance
RUN 1 (norm + 6 dialetti)        -0.191        +0.598    ← baseline OLD ★
RUN 2 (native + 13 var, no iso)  -0.124        +0.217    ← caduta -0.38 !!
```

CANINE ha **perso 0.38 di silhouette romance** passando da RUN 1 a RUN 2.

**Cause probabili (in ordine di peso atteso):**

1. **Rehearsal training con 13 varietà**: 700k frasi standard vs 410k
   dialetto → gradient diluito sui dialetti. CANINE (char-level, ~130M
   params) ha **capacità inferiore** a XLM-R-base (270M) per assorbire
   sia adattamento dialettale che preservazione delle standard. Probabile
   contributo: ~70% della regressione.

2. **Not_normalized text**: introduce varianza superficiale (case,
   diacritici) che CANINE deve ricontrollare. Probabile contributo: ~30%.

3. **Isotropy fix attivato (poi disattivato)**: contribuiva un altro
   -0.25 oltre alla regressione, già rimosso.

## Implicazione per il paper

Char-level encoders (CANINE) sembrano **più sensibili** a gradient
dilution sotto rehearsal-style training rispetto a sub-word encoders
(XLM-R). Per dialect adaptation con CANINE, **dialect-only training**
sembra preferibile.

Possibile ablation per disambiguare A vs B:
- **Run X1**: CANINE + normalized + dialect-only + 13var → isola effetto rehearsal
- **Run X2**: CANINE + not_normalized + dialect-only-6 → isola effetto text variant
- Costo HPC: ~16h totali

## Variety vectors

```
method_outputs/
├── flores/variety_vectors.npz       ← centroidi RUN 2 (più recenti)
├── oldi/variety_vectors.npz
└── models/mlm_wiki_dialects/        ← checkpoint CANINE adapted (RUN 2)
```

I `variety_vectors.npz` di RUN 1 NON sono salvati: solo le matrici di
distanza già calcolate (`centroid/distances.csv` + plot). Per ri-runnare
RUN 2 evaluation con setup diversi (isotropy on/off), usa lo script
in `slurm/jobs/reeval_no_isotropy.slurm`.
