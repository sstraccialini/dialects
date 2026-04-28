# TF-IDF baseline (FLORES+)

TF-IDF pipeline for the 16 FLORES+ varieties. Two vectorizers:

1. **Word n-grams** (1-2): unigrammi + bigrammi.
2. **Char n-grams** (3-5, `char_wb`): n-grammi di caratteri dentro i
   confini di parola. Particolarmente utili per dialetti, dove le
   variazioni ortografiche e fonetiche si catturano al livello del
   carattere.

## Obiettivo

Per ciascuna delle 16 varieta' produce un vettore TF-IDF e poi calcola
la matrice 16x16 di distanze coseno. Questa matrice e' il **baseline
di riferimento** contro cui confrontare gli approcci piu' sofisticati
(Word2Vec, FastText/BPE, mBERT/XLM-R).

## Design decisions

Tutto documentato in `src/config.py`. In sintesi:

| Scelta | Valore | Motivazione |
|---|---|---|
| Aggregazione | 1 super-documento per varieta' | Standard in dialettologia computazionale, interpretabile |
| Sample | 2009 frasi / varieta' (default) | Tutto FLORES+ (997 dev + 1012 devtest) |
| Random state | 42 | Riproducibilita' |
| Lowercase | Si' | Standard |
| Mask numbers | Si' (`<NUM>`) | Evita bias da date/numeri |
| Punteggiatura | Tolta per word, tenuta per char | Apostrofi distintivi nei dialetti |
| Diacritici | Tenuti | Accenti e spelling sono tratti distintivi |
| Char n-gram | (3, 5), `char_wb` | Standard VarDial / language ID |
| Word n-gram | (1, 2) | Unigrammi + bigrammi |
| `sublinear_tf` | True | Smorza i termini iperfrequenti |
| `min_df` / `max_df` | 1 / 1.0 | Conserva le feature uniche per varieta' (segnale) |
| `norm` | l2 | Distanza coseno stabile |
| Linkage | average | Coerente con la distanza coseno (ward vorrebbe euclidea) |

## Come lanciarlo

Dal root del repo, con il venv attivo:

```bash
pip install -r analysis_flores/tfidf/requirements.txt
python analysis_flores/tfidf/src/run_baseline.py
```

Opzioni utili:

```bash
# solo la pipeline char
python analysis_flores/tfidf/src/run_baseline.py --pipeline char

# sottocampionamento per sensitivity
python analysis_flores/tfidf/src/run_baseline.py --sample-size 1000

# seed diverso
python analysis_flores/tfidf/src/run_baseline.py --random-state 7
```

## Struttura

```
tfidf/
|- README.md              questo file
|- requirements.txt       dipendenze
|- src/
|   |- config.py          path, iperparametri, varieta'
|   |- data_loader.py     legge FLORES+ .txt, (eventuale) sub-sampling
|   |- preprocess.py      lowercase, mask numbers, strip punct (solo word)
|   |- vectorize.py       TfidfVectorizer word + char
|   |- similarity.py      matrice distanza coseno + nearest neighbors
|   |- cluster.py         clustering gerarchico, silhouette, dendrogram
|   |- visualize.py       proiezioni MDS / t-SNE 2D
|   |- run_baseline.py    orchestrator end-to-end
|- results/               output
|   |- word/              pipeline word
|   |- char/              pipeline char
|   |- shared/            silhouette report, run stats
```

## Output attesi in `results/`

`word/` e `char/` contengono:

- `distances.csv` (16x16 distanze coseno)
- `top_features.csv` (top-30 feature per varieta')
- `nearest_neighbors.csv` (top-3 vicini per varieta')
- `dendrogram.png`
- `projection_mds.png`, `projection_tsne.png`

`shared/` contiene:

- `silhouette_report.txt`
- `run_stats.csv`

## Cosa aspettarsi

Il baseline dovrebbe:

1. Separare Romance (italo_romance + italian + romance) da non-Romance.
2. Posizionare `Italian` vicino al cluster italo-romanzo (e' la sua
   forma standard).
3. Avvicinare `Catalan` a `Spanish` / romanze.
4. Tenere `Arabic` e `Greek` come outlier (script diversi
   -> n-grammi di caratteri quasi ortogonali).
5. Mostrare la pipeline **char** migliore della **word** (struttura
   piu' compatta, silhouette romance-vs-rest piu' alto).

Su FLORES+ ci si aspetta una separazione **piu' netta** rispetto al
baseline su Wikipedia, perche' il contenuto e' identico tra le varieta'
e quindi le differenze sono puramente linguistiche.

Cosa il baseline NON cattura (e lo fanno gli altri approcci):

- Contatti storici tra script diversi (influenza araba sul siciliano,
  greca sul napoletano, ...).
- Similitudine strutturale/sintattica profonda.
- Allineamento semantico cross-lingua.
