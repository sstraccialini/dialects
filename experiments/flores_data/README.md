# flores_data — corpora paralleli per il progetto NLP dialetti italiani

Contiene **solo FLORES+** scaricato da HuggingFace: corpus parallelo di
2009 frasi (997 `dev` + 1012 `devtest`) tradotte professionalmente.
Struttura piatta: **un file per lingua** (nome per esteso), più un TSV parallelo unico.

> **Nota metodologica.** FLORES+ originariamente separa `dev` (tuning) da
> `devtest` (test finale) per evitare overfitting nel training di modelli
> di traduzione. Per analisi di distanza linguistica (non stiamo allenando
> nulla) la distinzione non serve: li uniamo in 2009 frasi per avere più
> statistica. La colonna `split` nel TSV conserva comunque l'informazione
> originale nel caso servisse.

## Struttura

```
flores_data/
├── README.md
├── stats.csv                       # riepilogo: codice, categoria, n_frasi, n_caratteri, path, note
├── flores_plus/
│   ├── <lingua>.txt                # 2009 frasi, una per riga (es. veneto.txt, napoletano.txt)
│   └── parallel.tsv                # 2009 righe, colonne: sentence_id, split, <tutte le lingue>
└── scripts/
    ├── download_flores.py          # scarica tutto + genera parallel.tsv in un solo passaggio
    └── requirements.txt
```

## Lingue target (17 se tutte disponibili)

**Varieta' italiane (core, confermate):**

| File | Codice ISO | Regione |
|---|---|---|
| `veneto.txt` | vec_Latn | Veneto |
| `siciliano.txt` | scn_Latn | Sicilia |
| `lombardo.txt` | lmo_Latn | Lombardia |
| `sardo.txt` | srd_Latn | Sardegna |

**Varieta' italiane extra (da verificare caso per caso):**

| File | Codice ISO | Regione |
|---|---|---|
| `ligure.txt` | lij_Latn | Liguria |
| `friulano.txt` | fur_Latn | Friuli-V.G. |
| `ladino.txt` | lld_Latn | Dolomiti |
| `napoletano.txt` | nap_Latn | Campania |

**Italiano standard + 8 lingue esterne:**

`italiano.txt` · `inglese.txt` · `spagnolo.txt` · `francese.txt` · `catalano.txt` · `tedesco.txt` · `greco.txt` · `arabo.txt` · `sloveno.txt`

### Probabilmente NON presenti in FLORES+

Piemontese, emiliano-romagnolo, corso, romancio, sassarese, gallurese.
Se una delle varieta' extra non dovesse esistere, lo script la logga nel
`stats.csv` come errore e prosegue con le altre.

## Come scaricare

```bash
source venv/bin/activate
pip install -r flores_data/scripts/requirements.txt
hf auth login                              # solo la prima volta, salva il token
python flores_data/scripts/download_flores.py
```

5-10 minuti in totale. Alla fine:

- `flores_data/flores_plus/veneto.txt` → 2009 righe (una frase per riga)
- `flores_data/flores_plus/parallel.tsv` → 2010 righe (header + 2009)
- `flores_data/stats.csv` → una riga per ogni corpus, con dimensioni

## Caricare i dati in Python

**Per lingua (allineate per posizione):**

```python
italiano = open("flores_data/flores_plus/italiano.txt").read().splitlines()
veneto   = open("flores_data/flores_plus/veneto.txt").read().splitlines()
# italiano[i] e' la traduzione di veneto[i] per ogni i in [0, 2009)
```

**TSV parallelo (pandas):**

```python
import pandas as pd
df = pd.read_csv("flores_data/flores_plus/parallel.tsv", sep="\t")
df.columns          # ['sentence_id', 'split', 'veneto', 'siciliano', ..., 'sloveno']
df.iloc[0]          # prima frase in tutte le lingue
df["veneto"]        # intera colonna veneto
df[df.split=="dev"] # solo le prime 997 frasi, se serve la distinzione originale
```
