import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
for p in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import (
    BASE_MODEL,
    BATCH_SIZE,
    CONDITIONS,
    GROUP_COLORS,
    GROUP_NAMES,
    MAX_WIKI_SAMPLES,
    TSDAE_EPOCHS,
    MNRL_EPOCHS,
    VARIETY_CODES,
    VARIETY_GROUP,
    VARIETY_NAMES,
    evaluation_subdir,
    model_dir,
    outputs_subdir,
)
from data_loader import (
    iter_labeled_sentences,
    load_all_flores,
    load_all_oldi_pairs,
    load_wiki_texts,
)
from embedder import embed_sentences
from trainer import run_tsdae_training, run_mnrl_training
from evaluation.evaluation import run_evaluation

ROMANCE_FAMILIES = {"italo_romance", "italian", "romance"}

def embed_and_evaluate(model_path: str, flores_data: dict, condition: str, device: str = None):
    print(f"  Evaluating {condition}...")
    t0 = time.time()

    codes = [slug for slug in VARIETY_CODES if slug in flores_data]
    sents, slugs = iter_labeled_sentences(flores_data, codes)

    print(f"  Encoding {len(sents)} sentences...")
    sent_embeddings = embed_sentences(sents, model_path, batch_size=BATCH_SIZE)

    df_embeds = pd.DataFrame(sent_embeddings)
    df_embeds["slug"] = slugs

    print("  Aggregating variety vectors...")
    X = []
    for code in codes:
        code_embeds = df_embeds[df_embeds["slug"] == code].drop(columns=["slug"]).values
        centroid = code_embeds.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm
        X.append(centroid)
    X = np.array(X)

    out = outputs_subdir(condition)
    np.savez_compressed(out / "variety_vectors.npz", matrix=X, labels=np.asarray(codes))
    pd.DataFrame(X, index=codes).to_csv(out / "variety_vectors.csv", float_format="%.6f")

    report = run_evaluation(
        variety_vectors=X,
        variety_codes=codes,
        out_dir=evaluation_subdir(condition),
        method_label=f"SentenceTransformer ({condition})",
        family_groups=VARIETY_GROUP,
        family_colors=GROUP_COLORS,
        family_display_names=GROUP_NAMES,
        display_names=VARIETY_NAMES,
        romance_families=ROMANCE_FAMILIES,
    )
    return report, time.time() - t0

def main():
    parser = argparse.ArgumentParser(description="Sentence Transformer fine-tuning experiments")
    parser.add_argument(
        "--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS,
    )
    parser.add_argument(
        "--skip-train", action="store_true",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--tsdae-epochs", type=int, default=None)
    parser.add_argument("--mnrl-epochs", type=int, default=None)
    parser.add_argument("--wiki-samples", type=int, default=None)
    args = parser.parse_args()

    tsdae_epochs = args.tsdae_epochs or TSDAE_EPOCHS
    mnrl_epochs = args.mnrl_epochs or MNRL_EPOCHS
    wiki_samples = args.wiki_samples or MAX_WIKI_SAMPLES

    flores_data, flores_stats = load_all_flores()
    flores_stats.to_csv(outputs_subdir() / "flores_stats.csv", index=False)

    wiki_texts = None
    oldi_pairs = None
    tsdae_model_path: str | None = None

    results: dict = {}

    for condition in args.conditions:
        if condition == "baseline":
            model_path = BASE_MODEL
        elif condition == "tsdae_wiki":
            mdir = model_dir("tsdae_wiki")
            if args.skip_train and (mdir / "modules.json").exists():
                model_path = str(mdir)
            else:
                if wiki_texts is None:
                    wiki_texts = load_wiki_texts(max_per_lang=wiki_samples)
                run_tsdae_training(BASE_MODEL, wiki_texts, mdir, epochs=tsdae_epochs)
                model_path = str(mdir)
            tsdae_model_path = model_path
        elif condition == "mnrl_oldi":
            mdir = model_dir("mnrl_oldi")
            if args.skip_train and (mdir / "modules.json").exists():
                model_path = str(mdir)
            else:
                if oldi_pairs is None:
                    oldi_pairs = load_all_oldi_pairs()
                run_mnrl_training(BASE_MODEL, oldi_pairs, mdir, epochs=mnrl_epochs)
                model_path = str(mdir)
        elif condition == "tsdae_then_mnrl":
            mdir = model_dir("tsdae_then_mnrl")
            if args.skip_train and (mdir / "modules.json").exists():
                model_path = str(mdir)
            else:
                if tsdae_model_path is None:
                    tsdae_dir = model_dir("tsdae_wiki")
                    if wiki_texts is None:
                        wiki_texts = load_wiki_texts(max_per_lang=wiki_samples)
                    run_tsdae_training(BASE_MODEL, wiki_texts, tsdae_dir, epochs=tsdae_epochs)
                    tsdae_model_path = str(tsdae_dir)
                if oldi_pairs is None:
                    oldi_pairs = load_all_oldi_pairs()
                run_mnrl_training(tsdae_model_path, oldi_pairs, mdir, epochs=mnrl_epochs)
                model_path = str(mdir)

        report, elapsed = embed_and_evaluate(model_path, flores_data, condition, args.device)
        results[condition] = {
            "silhouette_family": report["silhouette_family"],
            "silhouette_romance_vs_rest": report["silhouette_romance_vs_rest"],
            "elapsed_s": round(elapsed),
        }
        
    print("\n" + "=" * 60)
    print("Summary")
    print(f"  {'Condition':<18} {'sil_family':>12} {'sil_romance':>12} {'time(s)':>9}")
    print("  " + "-" * 55)
    for cond, r in results.items():
        print(
            f"  {cond:<18} {r['silhouette_family']:>+12.4f} "
            f"{r['silhouette_romance_vs_rest']:>+12.4f} "
            f"{r['elapsed_s']:>9d}"
        )

    summary_path = outputs_subdir() / "condition_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(results, fh, indent=2)

if __name__ == "__main__":
    main()
