"""Cross-method shared utilities (variety registry, run-meta helper).

Single source of truth for everything that's identical across the 8
analysis methods (tfidf, word2vec, fasttext, sentence_minilm,
sentence_labse, multilingual_xlmr, multilingual_xlmr_adapted, canine).
Each method's `core/config.py` re-exports from here.
"""
