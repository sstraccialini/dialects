# Subword / FastText approach (FLORES+)

Due sotto-pipeline per rappresentare le 16 varieta' FLORES+ con unita'
**subword**, in modo da gestire meglio la morfologia dialettale
rispetto ai semplici word n-grammi della baseline TF-IDF.

1. **FastText** (gensim, skip-gram + char n-grammi 3-6)
   - un **unico modello condiviso** su tutte le 16 varieta' (2009 frasi
     ciascuna, ~32k frasi in totale)
   - embedding di frase = media dei vettori token
   - vettore per varieta' = media L2-normalizzata dei vettori di frase

2. **BPE + TF-IDF** (SentencePiece, vocab=8000)
   - un BPE condiviso, addestrato su tutte le frasi
   - ogni varieta' -> un super-documento di "pezzi" BPE
   - TF-IDF (stessi iperparametri della baseline word di Person 1)

Entrambe le sotto-pipeline producono una matrice 16x16 di distanze
coseno, confrontabile direttamente con la baseline TF-IDF.

## Design decisions

Tutto in `src/config.py`. In sintesi:

| Scelta | Valore | Motivazione |
|---|---|---|
| Modello FastText | **condiviso** tra varieta' | 2009 frasi/varieta' sono poche per un modello per-varieta'; il modello condiviso permette ai subword di generalizzare tra forme morfologicamente vicine |
| FastText algoritmo | **skip-gram** (`sg=1`) | Meglio per parole rare / corpus morfologicamente vario (Bojanowski et al. 2017) |
| `min_n, max_n` | **3, 6** | Copre morfemi tipici di italiano e lingue romanze |
| `min_count` | **2** | Parole uniche sono tipicamente typo / nomi propri; gli OOV vengono comunque embeddati via subword |
| `vector_size` | **200** | Dimensione abbastanza espressiva per ~32k frasi |
| `epochs` | **15** | Alzato da 10 (baseline Wikipedia) perche' il corpus FLORES+ e' molto piu' piccolo |
| BPE vocab | **8000** | Sufficiente per coprire 16 varieta' senza esplodere |
| BPE coverage | **0.9995** | Standard SentencePiece; adeguato con Arabico e Greco |
| TF-IDF su BPE | `sublinear_tf=True`, `norm='l2'`, `min_df=1`, `max_df=1.0` | Identico alla baseline word, per confronto pulito |
| Preprocessing | lowercase + mask numbers (`NUM`) + **tiene** diacritici e punteggiatura | Apostrofi e accenti sono tratti distintivi dialettali |
| Linkage | **average** | Coerente con distanza coseno |

## Come lanciarlo

Dal root del repo, con il venv attivo:

```bash
pip install -r analysis/fasttext/flores/requirements.txt
python analysis/fasttext/flores/src/run_approach.py
```

Opzioni utili:

```bash
# solo FastText
python analysis/fasttext/flores/src/run_approach.py --pipeline fasttext

# solo BPE + TF-IDF
python analysis/fasttext/flores/src/run_approach.py --pipeline bpe

# sottocampionamento per sensitivity
python analysis/fasttext/flores/src/run_approach.py --sample-size 1000
```

## Struttura

```
subword_fasttext/
|- README.md              questo file
|- requirements.txt       dipendenze
|- src/
|   |- config.py          path, iperparametri, 16 varieta'
|   |- data_loader.py     legge FLORES+ .txt, sub-sampling opzionale
|   |- preprocess.py      tokenize_for_fasttext / preprocess_for_subword
|   |- embed_fasttext.py  training FastText + embedding per varieta'
|   |- embed_bpe.py       SentencePiece BPE + TF-IDF su pezzi
|   |- similarity.py      matrice cosine + nearest neighbors
|   |- cluster.py         silhouette + dendrogram
|   |- visualize.py       MDS / t-SNE 2D
|   |- run_approach.py    orchestrator end-to-end
|- results/
|   |- fasttext/          output FastText
|   |- bpe/               output BPE + TF-IDF
|   |- models/            fasttext_model.bin, bpe_model.model/.vocab
|   |- shared/            silhouette_report.txt, run_stats.csv
```

## Output attesi in `results/`

`fasttext/` e `bpe/` contengono:

- `distances.csv`           (16x16 distanze coseno)
- `nearest_neighbors.csv`   (top-3 vicini per varieta')
- `dendrogram.png`
- `projection_mds.png`, `projection_tsne.png`

Inoltre:

- `fasttext/variety_vectors.csv`  (16 x 200)
- `fasttext/sentence_vectors.npz` (~32k x 200 + codes)
- `bpe/top_features.csv`          (top-30 pezzi BPE per varieta')

`models/` contiene i modelli addestrati (`fasttext_model.bin`,
`bpe_model.model`, `bpe_model.vocab`), che possono essere ricaricati
senza rifare il training.

`shared/` contiene `silhouette_report.txt` e `run_stats.csv`.

## Cosa aspettarsi

Rispetto alla baseline TF-IDF ci si aspetta che l'approccio subword:

1. Avvicini dialetti italo-romanzi morfologicamente simili (veneto,
   lombardo, ligure, friulano, ladino) perche' condividono suffissi
   e radici ma hanno spelling superficiali diversi.
2. Mantenga ben separati i gruppi non-romanze (Arabic / Greek con
   script diversi restano outlier).
3. Migliori lievemente la silhouette vs TF-IDF char n-grams, perche' i
   BPE merge sono *appresi* dai dati invece che di lunghezza fissa.

Cosa NON cattura (e lo fa `multilingual/` con XLM-R / mBERT):

- Similarita' semantica cross-lingua quando gli script sono diversi.
- Contatti storici che non lasciano tracce ortografiche.
