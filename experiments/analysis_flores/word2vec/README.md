# Word2Vec approach (FLORES+)

Pipeline Word2Vec condivisa sulle 16 varieta' FLORES+ (997 dev + 1012
devtest = 2009 frasi/varieta', ~32k frasi totali).

A differenza della baseline TF-IDF (che non ha una nozione di
similarita' distribuzionale) e della pipeline FastText (che sfrutta le
subword n-grammi), Word2Vec rappresenta ogni parola solo in base al
suo contesto. Le parole OOV non ottengono un vettore.

## Design decisions

Tutto in `src/config.py`. In sintesi:

| Scelta | Valore | Motivazione |
|---|---|---|
| Modello | **condiviso** tra varieta' | 2009 frasi/varieta' sono poche per un per-varieta' |
| Algoritmo | **skip-gram** (`sg=1`) | Meglio su parole rare / corpus morfologicamente vario |
| `vector_size` | **100** | Come la baseline Wikipedia, per confronto diretto |
| `window` | **5** | Standard |
| `min_count` | **2** | Scarta typo / nomi propri rari |
| `epochs` | **15** | Sufficiente per convergenza su ~32k frasi |
| Tokenizer | regex Unicode, tiene apostrofi | Fondamentale per dialetti (`l'università`, `c'è`, ...) |
| Preprocessing | lowercase + mask `NUM` + keep diacritics | Allineato a TF-IDF baseline |
| Vettore frase | media dei token in-vocab | Nessun vettore se tutti i token sono OOV |
| Vettore varieta' | media delle frasi, L2-normalizzata | Cosine-friendly |
| Linkage | **average** | Coerente con cosine distance |

## Come lanciarlo

Dal root del repo, con il venv attivo:

```bash
pip install -r analysis_flores/word2vec/requirements.txt
python analysis_flores/word2vec/src/run_word2vec.py
```

Opzioni utili:

```bash
# sottocampionamento per sensitivity
python analysis_flores/word2vec/src/run_word2vec.py --sample-size 1000

# seed diverso
python analysis_flores/word2vec/src/run_word2vec.py --random-state 7
```

## Struttura

```
word2vec/
|- README.md              questo file
|- requirements.txt       dipendenze
|- src/
|   |- config.py          path, iperparametri, 16 varieta'
|   |- data_loader.py     legge FLORES+ .txt, sub-sampling opzionale
|   |- preprocess.py      tokenize() robusto agli apostrofi
|   |- train.py           training Word2Vec skip-gram
|   |- build_vectors.py   embedding frase + embedding varieta'
|   |- similarity.py      matrice cosine + nearest neighbors
|   |- cluster.py         silhouette + dendrogram
|   |- visualize.py       MDS / t-SNE 2D
|   |- run_word2vec.py    orchestrator end-to-end
|- results/               output (creato runtime)
|   |- models/            word2vec.model
```

## Output attesi in `results/`

- `run_stats.csv`              per-varieta' n_available / n_used
- `models/word2vec.model`      modello gensim (Word2Vec.load)
- `sentence_vectors.npz`       (n_sent x 100) + aligned codes
- `variety_vectors.csv`        16 x 100, row-indexed by slug
- `variety_vectors.npz`        stesso contenuto in npz
- `distances.csv`              16 x 16 cosine distance
- `nearest_neighbors.csv`      top-3 vicini per varieta'
- `dendrogram.png`
- `projection_mds.png`, `projection_tsne.png`
- `silhouette_report.txt`      silhouette family + romance-vs-rest

## Cosa aspettarsi

Word2Vec puro si comporta *peggio* di FastText/BPE per i dialetti,
perche' non condivide parametri tra forme morfologicamente vicine. Ci
si aspetta quindi:

1. Separazione romance vs non-romance buona ma con silhouette piu'
   bassa rispetto a char-TF-IDF e FastText.
2. Italo-romanzi piu' dispersi perche' molte parole dialettali sono
   OOV nel resto del corpus.

Utile come termine di paragone distribuzionale (parallelo a
FLORES+) contro le rappresentazioni subword e multilingua.
