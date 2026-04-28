# Multilingual embeddings (FLORES+)

Approccio contestuale: ogni frase viene trasformata in un vettore
d'embedding con un transformer multilingua pre-addestrato (XLM-R di
default, mBERT disponibile via `--model-name`). Le 16 varieta' FLORES+
sono poi rappresentate come centroide L2-normalizzato delle loro frasi.

A differenza delle pipeline TF-IDF, Word2Vec e FastText/BPE,
XLM-R/mBERT hanno visto miliardi di token in pre-training, quindi
forniscono una rappresentazione gia' **allineata** cross-lingua. Ci si
aspetta la migliore separazione strutturale tra famiglie, soprattutto
per varieta' con script diversi (arabo, greco, slavo).

## Design decisions

Tutto in `src/config.py`. In sintesi:

| Scelta | Valore | Motivazione |
|---|---|---|
| Modello di default | **xlm-roberta-base** | Copertura multilingua ampia, objective MLM pulito |
| Alternativa | `bert-base-multilingual-cased` | mBERT come confronto classico |
| Pooling | attention-masked **mean pooling** del last hidden layer | CLS funziona male senza fine-tuning supervisionato |
| L2 normalisation | sul vettore frase + sul centroide varieta' | Cosine distance stabile |
| Max length | **128** token | FLORES+ ha frasi corte (~30-80 token) |
| Batch size | **32** | Compromesso CPU/GPU |
| Nessun preprocessing | testo grezzo al tokenizer | Il tokenizer del modello gestisce la normalizzazione |
| Linkage | **average** | Coerente con cosine distance (corretto vs Ward, che richiede distanze Euclidee — fix vs baseline originale) |
| Stessa lista di 16 varieta' | | Allineata alle altre pipeline analysis_flores |

## Come lanciarlo

Dal root del repo, con il venv attivo:

```bash
pip install -r analysis/multilingual_xlmr/flores/requirements.txt
python analysis/multilingual_xlmr/flores/src/run_pipeline.py
```

Opzioni utili:

```bash
# mBERT invece di XLM-R
python analysis/multilingual_xlmr/flores/src/run_pipeline.py --model-name bert-base-multilingual-cased

# Sentence-Transformer gia' allineato (piu' rapido, vettori migliori out-of-the-box)
python analysis/multilingual_xlmr/flores/src/run_pipeline.py \
    --model-name sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# forza CPU (utile per debug)
python analysis/multilingual_xlmr/flores/src/run_pipeline.py --device cpu

# sottocampionamento per sensitivity
python analysis/multilingual_xlmr/flores/src/run_pipeline.py --sample-size 500
```

Nota: XLM-R base pesa ~1.1 GB (download al primo run via HuggingFace).
Con GPU l'embedding di 32k frasi FLORES+ richiede qualche minuto; su
CPU e' piu' lento ma rimane fattibile.

## Struttura

```
multilingual/
|- README.md              questo file
|- requirements.txt       dipendenze (torch, transformers, ...)
|- src/
|   |- config.py          path, iperparametri, 16 varieta'
|   |- data_loader.py     legge FLORES+ .txt
|   |- embedder.py        AutoTokenizer + AutoModel + mean pooling
|   |- similarity.py      matrice cosine + nearest neighbors
|   |- cluster.py         silhouette + dendrogram
|   |- visualize.py       MDS / t-SNE 2D
|   |- run_pipeline.py    orchestrator end-to-end
|- results/               output (creato runtime)
```

## Output attesi in `results/`

- `run_stats.csv`              per-varieta' n_available / n_used / model_name
- `sentence_vectors.npz`       (n_sent x D) + aligned codes + model_name
- `variety_vectors.csv`        16 x D, row-indexed by slug
- `variety_vectors.npz`        stesso contenuto in npz
- `distances.csv`              16 x 16 cosine distance
- `nearest_neighbors.csv`      top-3 vicini per varieta'
- `dendrogram.png`
- `projection_mds.png`, `projection_tsne.png`
- `silhouette_report.txt`      silhouette family + romance-vs-rest

## Cosa aspettarsi

XLM-R / mBERT dovrebbero:

1. Dare la **migliore silhouette** sulla divisione Romance vs non-Romance,
   perche' gli embedding contestuali sono semanticamente allineati tra
   lingue anche con script diversi.
2. Avvicinare di piu' le varieta' cross-script (es. Arabic / Greek non
   sono piu' outlier "orto-grafici" ma si posizionano in un sottospazio
   semantico condiviso).
3. Mostrare pero' un avvicinamento meno marcato tra dialetti italo-romanzi
   vicini (veneto vs lombardo) perche' il modello di base "smussa"
   differenze dialettali sottili che invece Word2Vec/FastText catturano.

E' il confronto con TF-IDF e subword a rendere questa pipeline
interessante: quando la **struttura semantica globale** domina
(multilingua) e quando invece la **forma ortografica locale** conta
(dialetti).
