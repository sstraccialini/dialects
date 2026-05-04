"""MiniLM fine-tuning module — TSDAE/MNRL training routines and the 4-condition pipeline.

Specific to fine-tuning the paraphrase-multilingual-MiniLM backbone
(training hyperparameters, model checkpoint paths, denoising/contrastive
loops, training data loaders for parallel pairs and dialect Wiki text).
Anything shared with the unadapted baseline lives in
`analysis.sentence_minilm.core`.
"""
