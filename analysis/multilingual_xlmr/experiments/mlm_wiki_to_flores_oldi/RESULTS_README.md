# XLM-R MLM Wiki — guida ai risultati

Questa cartella contiene i risultati di **due run distinti** dello stesso
esperimento, ottenuti con setup metodologici diversi. Le sotto-cartelle
in `evaluation_results/{flores,oldi}/` riflettono la differenza.

## Quale cartella corrisponde a cosa

```
evaluation_results/
├── flores/
│   ├── centroid/                ← RUN 1 (vecchio, riferimento)
│   ├── centroid_with_iso/       ← RUN 2 (nuovo, con isotropy fix)
│   ├── centroid_no_iso/         ← RUN 2 (nuovo, senza isotropy fix)
│   └── parallel/                ← RUN 1 parallel alignment frase-per-frase
└── oldi/
    └── (stessa struttura)
```

## RUN 1 — `centroid/` e `parallel/`

**Setup**:
- Text variant: **normalized** (lowercase ASCII, no diacritici)
- Training set: **6 dialetti italo-romance soltanto** (fur, lij, lmo, sc, scn, vec)
- Isotropy correction: **non applicata**
- Pretraining: continual MLM su Wiki normalized + TLM su OLDI dove specificato

**Numeri principali**:
- `flores/centroid/`: silhouette family = -0.157, silhouette romance = +0.113
- `oldi/centroid/`:   silhouette family = -0.158, silhouette romance = +0.121

## RUN 2 — `centroid_with_iso/` e `centroid_no_iso/`

**Setup**:
- Text variant: **not_normalized** (con diacritici, maiuscole, punteggiatura)
- Training set: **tutte 13 varietà** (rehearsal-style), cap 100k frasi cad.
  - 6 dialetti (fur, lij, lmo, sc, scn, vec)
  - 7 standard (ita, spa, fra, cat, deu, slv, eng)
  - Total: ~1.1M frasi training
- Continual MLM 3 epoche, lr=3e-5, batch=32

**Le due varianti differiscono solo per l'evaluation**:

| Sotto-cartella | Isotropy correction | sil_rom (FLORES) | Note |
|---|---|---:|---|
| `centroid_with_iso/` | Mu & Viswanath top-1 PC removal | +0.020 | top-PC distrugge il segnale Romance — non usare |
| `centroid_no_iso/` | nessuna correzione | +0.062 | versione corretta del run 2 |

## Confronto e interpretazione

```
                       sil_family   sil_romance
RUN 1  (norm + 6 dialetti)      -0.157        +0.113     ← baseline OLD
RUN 2  (native + 13 var)        -0.026        +0.062     (no iso, FLORES)
```

XLM-R è leggermente regredito sul `sil_romance` passando dal RUN 1 al RUN 2.
Probabile causa: la **rehearsal training** con 7 standard (700k frasi) ha
diluito il segnale dialettale (~410k frasi). XLM-R ha capacità sufficiente
per assorbire entrambi senza catastrophic forgetting, ma il segnale
dialect-specific si è attenuato.

Per il paper: documentare questo trade-off come finding metodologico —
"rehearsal training preserves standard-language representations at the
cost of slightly reduced dialect-specific discrimination".

## Variety vectors

```
method_outputs/
├── flores/
│   └── variety_vectors.npz      ← centroidi RUN 2 native+13var (più recente)
├── oldi/
│   └── variety_vectors.npz      ← idem
└── models/
    └── mlm_wiki_dialects/       ← checkpoint XLM-R adapted (RUN 2)
```

Per generare i `centroid_*_iso/` localmente da questi `.npz`,
vedi `slurm/jobs/reeval_no_isotropy.slurm` o lo script Python
nelle conversation logs (caricare .npz, chiamare `run_evaluation`
con `isotropy=True/False`).
