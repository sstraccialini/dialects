# slurm/tools — environment & setup

Shared SLURM tooling. Per-user paths live in a gitignored
`env.local.sh`, so each group member can keep their own settings without
colliding with the others.

Bocconi cluster, partition `stud` (qos `stud`), 1 GPU per job.

## One-time bootstrap (login node)

```bash
cp slurm/tools/env.local.example.sh slurm/tools/env.local.sh
${EDITOR:-nano} slurm/tools/env.local.sh   # set LTP_VENV / FLUX_ENV / LTP_HF_CACHE
bash slurm/tools/setup_env.sh
```

`setup_env.sh` creates the project's clean Python venv at `LTP_VENV`,
optionally piggy-backs on a shared read-only `flux_env`'s site-packages
to skip ~3 GB of duplicate downloads, and installs the rest of
`requirements.txt`.

## Files

| File | Purpose |
|---|---|
| `env.sh` | sourced by every SLURM job; activates `LTP_VENV` and exports `PROJECT_ROOT`, `HF_HOME`, ... reads per-user paths from `env.local.sh`. |
| `env.local.example.sh` | template — copy to `env.local.sh` and edit. |
| `env.local.sh` | gitignored, per-user paths (`LTP_VENV`, `FLUX_ENV`, `LTP_HF_CACHE`). |
| `setup_env.sh` | one-time bootstrap on a login node. |
| `requirements.txt` | additional pip packages (on top of any flux_env). |

## Submit jobs

```bash
sbatch slurm/jobs/run_all.slurm                 # everything sequentially
sbatch slurm/jobs/04_flores_multilingual.slurm  # one pipeline
bash   slurm/jobs/submit_all.sh                 # submit each numbered job separately
```

Logs land in `slurm/logs/<jobname>_<jobid>.{out,err}`.

## Pipelines

| # | Script | Method path |
|---|---|---|
| 01 | flores TF-IDF                | `analysis/tfidf/flores/` |
| 02 | flores Word2Vec              | `analysis/word2vec/flores/` |
| 03 | flores fastText + BPE        | `analysis/fasttext/flores/` |
| 04 | flores XLM-R multilingual    | `analysis/multilingual_xlmr/flores/` |
| 05 | flores sentence baseline     | `analysis/sentence_baseline/flores/` |
| 06 | wiki TF-IDF                  | `analysis/tfidf/wiki/` |
| 07 | wiki Word2Vec                | `analysis/word2vec/wiki/` |
| 08 | wiki fastText + BPE          | `analysis/fasttext/wiki/` |
| 09 | wiki XLM-R multilingual      | `analysis/multilingual_xlmr/wiki/` |
| 10 | wiki sentence baseline       | `analysis/sentence_baseline/wiki/` |
| 11 | flores LaBSE                 | `analysis/sentence_labse/flores/` |
| 12 | wiki   LaBSE                 | `analysis/sentence_labse/wiki/` |
| 13 | zero-shot perplexity         | `analysis/zeroshot_ppl/flores/` |
| 15 | flores XLM-R adapted         | `analysis/multilingual_xlmr_adapted/flores/` |

All use `--gpus=1`. Lighter pipelines (TF-IDF, Word2Vec, FastText) don't
need a GPU but the `stud` partition requires one.

Each method writes to:
- `<method>/<dataset>/method_outputs/`     vectors, models, run stats, top features
- `<method>/<dataset>/evaluation_results/` dendrogram.png, distances.csv, projections, silhouette
