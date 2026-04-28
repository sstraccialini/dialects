# Modeling Historical and Linguistic Relations Between Italian Dialects and Contemporary Languages Through Embedding Spaces

Code and data for the project of `20879 - Language Technology` (Spring
2026, Bocconi University). We model historical and linguistic relations
between Italian dialects and contemporary languages through embedding
spaces, comparing several embedding methods (TF-IDF, Word2Vec, FastText,
XLM-R, LaBSE, ...) on three corpora (FLORES+, Wikipedia, OLDI).

## Repository layout

```
Language-Technology-Project/
├── Dataset/
│   ├── flores/                 FLORES+ parallel corpus (per-variety .txt)
│   ├── wiki/                   Wikipedia per-variety CSVs
│   └── oldi/                   OLDI seed parquet files
│
├── evaluation/
│   └── evaluation.py           single entry point used by every method
│                               at the end of its pipeline; produces
│                               distances.csv, dendrogram.png,
│                               nearest_neighbors.csv, projection_*.png,
│                               silhouette_report.txt
│
├── analysis/                   one folder per method; each method has
│   ├── tfidf/                    one sub-folder per dataset variant.
│   │   ├── flores/{src,method_outputs,evaluation_results}/
│   │   └── wiki/  {src,method_outputs,evaluation_results}/
│   ├── word2vec/...
│   ├── fasttext/...
│   ├── multilingual_xlmr/...
│   ├── multilingual_xlmr_adapted/flores/...
│   ├── sentence_baseline/...
│   ├── sentence_labse/...
│   └── zeroshot_ppl/flores/...
│
└── slurm/
    ├── jobs/                   one .slurm per pipeline (sbatch <file>)
    ├── tools/                  shared env.sh + per-user env.local.sh
    └── logs/                   SLURM stdout/stderr per job
```

For each method:
- `src/` is the code (data loading, embedding, training, ...).
- `method_outputs/` is the method's own artefacts (vectors, models,
  run stats, top features).
- `evaluation_results/` is whatever `evaluation/evaluation.py` produces:
  the same dendrogram / distances / projections / silhouette files for
  every method, so methods can be compared apples-to-apples.

To swap the evaluation logic, edit a single file:
[evaluation/evaluation.py](evaluation/evaluation.py).

## Run on HPC (SLURM)

One-time bootstrap on a login node:

```bash
cp slurm/tools/env.local.example.sh slurm/tools/env.local.sh
${EDITOR:-nano} slurm/tools/env.local.sh   # set LTP_VENV / FLUX_ENV / LTP_HF_CACHE
bash slurm/tools/setup_env.sh
```

`env.local.sh` is gitignored — every group member maintains their own
paths there without colliding with the others.

Submit:

```bash
sbatch slurm/jobs/run_all.slurm                 # everything sequentially
sbatch slurm/jobs/04_flores_multilingual.slurm  # just one pipeline
bash   slurm/jobs/submit_all.sh                 # one job per pipeline
```

See [slurm/tools/README.md](slurm/tools/README.md) for the full pipeline
table.

## Run a single method locally

```bash
source venv/bin/activate
pip install -r requirements.txt

python analysis/tfidf/flores/src/run_baseline.py
python analysis/word2vec/flores/src/run_word2vec.py
# ... etc.
```

Each `run_*.py` ends with a call to `evaluation.run_evaluation(...)`,
which writes the standard evaluation artefacts under that method's
`evaluation_results/`.

## License

MIT — see [LICENSE](LICENSE).
