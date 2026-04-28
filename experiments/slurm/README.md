# SLURM scripts for HPC

Bocconi cluster, partition `stud` (qos `stud`), 1 GPU per job.

## One-time bootstrap (login node)

```bash
cd experiments
bash slurm/setup_env.sh
```

No venv created. Uses flux_env's python (read-only) and installs only
the 6 missing packages (pandas, scikit-learn, seaborn, gensim,
sentence-transformers, datasets) into `~/ltp_extras` via
`pip install --target`. Total disk: ~150 MB extras + HF cache.

flux_env is never modified.

Override paths if needed:
```bash
FLUX_ENV=/path/to/flux_env \
LTP_EXTRAS=/some/path/extras \
LTP_HF_CACHE=/some/path/hf_cache \
bash slurm/setup_env.sh
```

## Submit jobs

**Default (single job, all 10 pipelines sequentially, ~75 min wall time):**
```bash
cd experiments
sbatch slurm/run_all.slurm
```

Submit just one:
```bash
sbatch slurm/04_flores_multilingual.slurm
```

Submit all 10 as separate jobs (only if the queue allows multiple):
```bash
bash slurm/submit_all.sh
```

Logs go to `slurm/logs/<jobname>_<jobid>.{out,err}`.

## Pipelines

| # | Script | Time | Mem |
|---|---|---|---|
| 01 | flores TF-IDF | 30m | 8G |
| 02 | flores Word2Vec | 30m | 8G |
| 03 | flores fastText + BPE | 30m | 8G |
| 04 | flores XLM-R multilingual | 1h | 16G |
| 05 | flores sentence_baseline | 1h | 16G |
| 06 | wiki TF-IDF | 1h | 16G |
| 07 | wiki Word2Vec | 2h | 24G |
| 08 | wiki fastText + BPE | 2h | 24G |
| 09 | wiki XLM-R multilingual | 2h | 24G |
| 10 | wiki sentence_baseline | 3h | 24G |

All use `--gpus=1`. Lights (TF-IDF / W2V / fastText) don't actually need
the GPU but the `stud` partition requires one.
