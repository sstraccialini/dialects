# analysis_flores — pipelines sul corpus FLORES+

Questa cartella contiene i **quattro approcci di embedding** del progetto
"Modeling Historical and Linguistic Relations Between Italian Dialects
and Contemporary Languages Through Embedding Spaces", riscritti per
lavorare sul nuovo dataset pulito **FLORES+**.

La cartella gemella `analysis_wiki/` (con `tfidf_baseline/`, `word2vec/`,
`subword_fasttext/`, `multilingual_embeddings/`) contiene la versione
precedente che girava sui dump Wikipedia: resta intatta come
riferimento.

## Perche' FLORES+ e non Wikipedia

FLORES+ e' un **corpus parallelo tradotto professionalmente**: 2009
frasi identiche in 16 lingue diverse (997 `dev` + 1012 `devtest`).

Rispetto a Wikipedia:

- Il contenuto e' lo stesso in tutte le lingue. Qualunque differenza
  nelle distanze tra varieta' riflette **solo** differenze linguistiche,
  non differenze di cosa parla ogni Wikipedia.
- I dati sono puliti, deduplicati e tradotti da umani. Zero rumore
  editoriale, zero template Wikipedia, zero boilerplate.
- Tutte le 16 varieta' hanno esattamente lo stesso numero di frasi,
  quindi non serve sub-sampling per bilanciare.

## Le 16 varieta' analizzate

| Varieta' | File FLORES+ | Famiglia |
|---|---|---|
| Veneto | `veneto.txt` | italo_romance |
| Siciliano | `siciliano.txt` | italo_romance |
| Lombardo | `lombardo.txt` | italo_romance |
| Sardo | `sardo.txt` | italo_romance |
| Ligure (Genovese) | `ligure.txt` | italo_romance |
| Friulano | `friulano.txt` | italo_romance |
| Ladino | `ladino.txt` | italo_romance |
| Italiano | `italiano.txt` | italian |
| Spagnolo | `spagnolo.txt` | romance |
| Francese | `francese.txt` | romance |
| Catalano | `catalano.txt` | romance |
| Tedesco | `tedesco.txt` | germanic |
| Inglese | `inglese.txt` | germanic |
| Greco | `greco.txt` | greek |
| Arabo | `arabo.txt` | semitic |
| Sloveno | `sloveno.txt` | slavic |

**Differenze rispetto ai 14 varieta' originali (Wikipedia):**

- PERSO: `napoletano` — FLORES+ non lo include (limite noto del dataset).
- GUADAGNATO: `ligure`, `friulano`, `ladino` — non erano nella lista
  originale ma sono dialetti italoromanzi di alta qualita' disponibili
  in FLORES+.

Net: 16 varieta' invece di 14, con 3 dialetti italoromanzi in piu'.

## Struttura della cartella

```
analysis_flores/
├── README.md                       questo file
├── tfidf/                          TF-IDF baseline (word + char n-gram)
├── word2vec/                       Word2Vec skip-gram (gensim)
├── subword_fasttext/               FastText + BPE (gensim + SentencePiece)
└── multilingual/                   XLM-R / mBERT feature extraction
```

Ogni sotto-cartella ha:

```
<metodo>/
├── README.md                       specifiche del metodo
├── requirements.txt                dipendenze
├── src/                            codice (1 modulo per step)
│   ├── config.py                   path, varieta', iperparametri
│   ├── data_loader.py              lettura FLORES+ .txt
│   ├── preprocess.py               normalizzazione testo
│   ├── ...                         moduli specifici per metodo
│   └── run_*.py                    orchestrator end-to-end
└── results/                        output (distances, dendrogram, plot, ...)
```

## Output comune a tutti i metodi

Per poter confrontare i 4 approcci sulla stessa scala, **tutti**
producono nella rispettiva cartella `results/`:

- `distances.csv` — matrice 16×16 di distanze coseno
- `nearest_neighbors.csv` — top-3 vicini per ciascuna varieta'
- `dendrogram.png` — clustering gerarchico (linkage = average, coseno)
- `projection_mds.png` — proiezione MDS 2D (distanze globali fedeli)
- `projection_tsne.png` — proiezione t-SNE 2D (cluster locali)
- `silhouette_report.txt` — silhouette contro etichette di famiglia
  e contro romance-vs-rest

Colori e nomi delle famiglie sono identici tra i 4 metodi (definiti
nei `config.py`), quindi i plot sono visivamente confrontabili.

## Come lanciare tutto

Dal root del repo, con il virtual environment attivato:

```bash
# TF-IDF
pip install -r analysis_flores/tfidf/requirements.txt
python analysis_flores/tfidf/src/run_baseline.py

# Word2Vec
pip install -r analysis_flores/word2vec/requirements.txt
python analysis_flores/word2vec/src/run_word2vec.py

# FastText + BPE
pip install -r analysis_flores/subword_fasttext/requirements.txt
python analysis_flores/subword_fasttext/src/run_approach.py

# Multilingual (mBERT / XLM-R)
pip install -r analysis_flores/multilingual/requirements.txt
python analysis_flores/multilingual/src/run_pipeline.py
```

I primi tre girano in 1-10 minuti su una CPU moderna. Il quarto
(multilingual) richiede idealmente una GPU o puo' girare su CPU in
qualche minuto per 2009 frasi × 16 lingue.

## Prerequisito dati

Prima di lanciare qualsiasi pipeline bisogna avere i file FLORES+:

```
flores_data/flores_plus/<lingua>.txt
flores_data/flores_plus/parallel.tsv
```

Se non ci sono:

```bash
source venv/bin/activate
pip install -r flores_data/scripts/requirements.txt
hf auth login   # solo la prima volta
python flores_data/scripts/download_flores.py
```

Istruzioni complete in `flores_data/README.md`.
