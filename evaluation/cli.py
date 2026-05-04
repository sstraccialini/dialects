#!/usr/bin/env python3
"""
Standalone CLI for running the evaluation suite on saved variety vectors.

Sub-commands
------------
run          Variety-level evaluation from a .npz or .csv vector file.
             This is the default when --vectors is given without a sub-command.
sentence     Sentence-level evaluation from a sentence_vectors.npz file.
compare      Cross-method comparison from multiple evaluation_results/ dirs.

Examples
--------
# Variety-level evaluation — NPZ (Word2Vec, XLM-R, FastText, etc.)
python evaluation/cli.py run \\
    --vectors  analysis/word2vec/flores/method_outputs/variety_vectors.npz \\
    --method   "Word2Vec (FLORES+)" \\
    --out-dir  analysis/word2vec/flores/evaluation_results

# Same, CSV format
python evaluation/cli.py run \\
    --vectors  analysis/multilingual_xlmr/flores/method_outputs/variety_vectors.csv \\
    --method   "XLM-R (FLORES+)" \\
    --out-dir  analysis/multilingual_xlmr/flores/evaluation_results

# Sentence-level evaluation
python evaluation/cli.py sentence \\
    --vectors  analysis/multilingual_xlmr/flores/method_outputs/sentence_vectors.npz \\
    --method   "XLM-R sentences (FLORES+)" \\
    --out-dir  analysis/multilingual_xlmr/flores/evaluation_results/sentence_level

# Cross-method comparison (distance matrices must exist in each eval dir)
python evaluation/cli.py compare \\
    --method-dirs "Word2Vec:analysis/word2vec/flores/evaluation_results,XLM-R:analysis/multilingual_xlmr/flores/evaluation_results" \\
    --vector-paths "Word2Vec:analysis/word2vec/flores/method_outputs/variety_vectors.npz,XLM-R:analysis/multilingual_xlmr/flores/method_outputs/variety_vectors.npz" \\
    --out-dir analysis/comparison/flores
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Standard taxonomy  (mirrors all analysis/*/src/config.py files)
# --------------------------------------------------------------------------- #

FAMILY_GROUPS: dict[str, str] = {
    "veneto":     "italo_romance",
    "siciliano":  "italo_romance",
    "lombardo":   "italo_romance",
    "sardo":      "italo_romance",
    "ligure":     "italo_romance",
    "friulano":   "italo_romance",
    "ladino":     "italo_romance",
    "napolitano": "italo_romance",
    "italiano":   "italian",
    "spagnolo":   "romance",
    "francese":   "romance",
    "catalano":   "romance",
    "tedesco":    "germanic",
    "inglese":    "english",
    "greco":      "greek",
    "arabo":      "semitic",
    "sloveno":    "slavic",
}

FAMILY_COLORS: dict[str, str] = {
    "italo_romance": "#d62728",
    "italian":       "#ff7f0e",
    "romance":       "#2ca02c",
    "germanic":      "#1f77b4",
    "english":       "#17becf",
    "greek":         "#9467bd",
    "semitic":       "#8c564b",
    "slavic":        "#e377c2",
}

FAMILY_DISPLAY_NAMES: dict[str, str] = {
    "italo_romance": "Italo-Romance dialect",
    "italian":       "Standard Italian",
    "romance":       "Other Romance",
    "germanic":      "Germanic",
    "english":       "English",
    "greek":         "Greek",
    "semitic":       "Semitic",
    "slavic":        "Slavic",
}

DISPLAY_NAMES: dict[str, str] = {
    "veneto":     "Veneto",
    "siciliano":  "Siciliano",
    "lombardo":   "Lombardo",
    "sardo":      "Sardo",
    "ligure":     "Ligure",
    "friulano":   "Friulano",
    "ladino":     "Ladino",
    "napolitano": "Napolitano",
    "italiano":   "Italiano",
    "spagnolo":   "Spagnolo",
    "francese":   "Francese",
    "catalano":   "Catalano",
    "tedesco":    "Tedesco",
    "inglese":    "Inglese",
    "greco":      "Greco",
    "arabo":      "Arabo",
    "sloveno":    "Sloveno",
}

ROMANCE_FAMILIES: set[str] = {"italo_romance", "italian", "romance"}


# --------------------------------------------------------------------------- #
# Vector loaders
# --------------------------------------------------------------------------- #

def _load_variety_vectors(path: str):
    """Return (matrix: np.ndarray, codes: list[str]) from .npz or .csv."""
    import pandas as pd
    p = Path(path)
    if p.suffix == ".npz":
        data = np.load(p, allow_pickle=True)
        matrix = data["matrix"].astype(np.float32)
        codes = [str(x) for x in data["labels"]]
    elif p.suffix in (".csv", ".tsv"):
        sep = "\t" if p.suffix == ".tsv" else ","
        df = pd.read_csv(p, index_col=0, sep=sep)
        codes = list(df.index)
        matrix = df.values.astype(np.float32)
    else:
        raise ValueError(f"Unsupported format '{p.suffix}'. Use .npz or .csv")
    return matrix, codes


def _load_sentence_vectors(path: str):
    """Return (matrix, labels) from a sentence_vectors.npz file."""
    data = np.load(path, allow_pickle=True)
    if "matrix" not in data or "labels" not in data:
        raise ValueError(
            f"sentence_vectors.npz must contain keys 'matrix' and 'labels'.\n"
            f"Found keys: {list(data.keys())}"
        )
    matrix = data["matrix"].astype(np.float32)
    labels = [str(x) for x in data["labels"]]
    return matrix, labels


# --------------------------------------------------------------------------- #
# Sub-commands
# --------------------------------------------------------------------------- #

def _ensure_project_root_on_path():
    """Add the project root (parent of evaluation/) to sys.path."""
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def cmd_run(args: argparse.Namespace) -> None:
    """Variety-level evaluation from a saved vector file."""
    _ensure_project_root_on_path()
    from evaluation.evaluation import run_evaluation

    matrix, codes = _load_variety_vectors(args.vectors)
    method = args.method or Path(args.vectors).parent.parent.name
    print(f"Loaded {matrix.shape[0]} varieties × {matrix.shape[1]} dims  [{args.vectors}]")

    # Filter taxonomy to varieties present in the file
    fg = {c: FAMILY_GROUPS[c] for c in codes if c in FAMILY_GROUPS}
    dn = {c: DISPLAY_NAMES[c] for c in codes if c in DISPLAY_NAMES}

    results = run_evaluation(
        matrix, codes,
        out_dir=args.out_dir,
        method_label=method,
        family_groups=fg,
        family_colors=FAMILY_COLORS,
        family_display_names=FAMILY_DISPLAY_NAMES,
        display_names=dn,
        romance_families=ROMANCE_FAMILIES,
        linkage_method=args.linkage,
        nearest_k=args.nearest_k,
    )

    print(f"\nDone. Outputs written to: {args.out_dir}")
    for k, v in results.items():
        if isinstance(v, str) and Path(v).exists():
            print(f"  {k:<35} {v}")
        elif v is not None and not isinstance(v, (str, Path, list)):
            print(f"  {k:<35} {v}")


def cmd_sentence(args: argparse.Namespace) -> None:
    """Sentence-level evaluation from sentence_vectors.npz."""
    _ensure_project_root_on_path()
    from evaluation.evaluation import run_sentence_evaluation

    matrix, labels = _load_sentence_vectors(args.vectors)
    method = args.method or ""
    print(f"Loaded {matrix.shape[0]} sentences × {matrix.shape[1]} dims  [{args.vectors}]")

    present = set(labels)
    fg = {c: FAMILY_GROUPS[c] for c in present if c in FAMILY_GROUPS}
    dn = {c: DISPLAY_NAMES[c] for c in present if c in DISPLAY_NAMES}

    results = run_sentence_evaluation(
        matrix, labels,
        out_dir=args.out_dir,
        method_label=method,
        family_groups=fg,
        family_colors=FAMILY_COLORS,
        family_display_names=FAMILY_DISPLAY_NAMES,
        display_names=dn,
        romance_families=ROMANCE_FAMILIES,
        n_sample=args.n_sample,
    )

    print(f"\nDone. Outputs written to: {args.out_dir}")
    for k, v in results.items():
        if v:
            print(f"  {k:<35} {v}")


def cmd_compare(args: argparse.Namespace) -> None:
    """Cross-method comparison from evaluation_results directories."""
    _ensure_project_root_on_path()
    from evaluation.compare_methods import run_cross_method_comparison

    def _parse_colon_pairs(s: str) -> dict:
        result = {}
        for item in s.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(f"Expected 'Name:path', got '{item}'")
            name, path = item.split(":", 1)
            result[name.strip()] = path.strip()
        return result

    method_dirs = _parse_colon_pairs(args.method_dirs)
    vector_paths = _parse_colon_pairs(args.vector_paths) if args.vector_paths else None

    print(f"Comparing {len(method_dirs)} methods:")
    for name, d in method_dirs.items():
        print(f"  {name}: {d}")

    results = run_cross_method_comparison(
        method_dirs,
        out_dir=args.out_dir,
        variety_vector_paths=vector_paths,
        mantel_permutations=args.mantel_perms,
    )

    print(f"\nDone. Outputs written to: {args.out_dir}")
    for k, v in results.items():
        if v and isinstance(v, str) and Path(v).exists():
            print(f"  {k:<35} {v}")


# --------------------------------------------------------------------------- #
# Argument parser
# --------------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python evaluation/cli.py",
        description="Italian dialect embedding evaluation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # ---- run (default) ----------------------------------------------------
    run_p = sub.add_parser("run", help="Variety-level evaluation (default)")
    run_p.add_argument("--vectors", required=True,
                       help="Path to variety_vectors.npz or variety_vectors.csv")
    run_p.add_argument("--out-dir", required=True, dest="out_dir",
                       help="Output directory")
    run_p.add_argument("--method", default="",
                       help="Human-readable method label for plot titles")
    run_p.add_argument("--linkage", default="average",
                       choices=["average", "complete", "single", "ward"],
                       help="Hierarchical linkage method (default: average)")
    run_p.add_argument("--nearest-k", type=int, default=3, dest="nearest_k",
                       help="Nearest neighbours per variety (default: 3)")
    run_p.set_defaults(func=cmd_run)

    # ---- sentence ---------------------------------------------------------
    sent_p = sub.add_parser("sentence", help="Sentence-level evaluation")
    sent_p.add_argument("--vectors", required=True,
                        help="Path to sentence_vectors.npz (keys: matrix, labels)")
    sent_p.add_argument("--out-dir", required=True, dest="out_dir")
    sent_p.add_argument("--method", default="")
    sent_p.add_argument("--n-sample", type=int, default=5000, dest="n_sample",
                        help="Subsample for silhouette/UMAP (default: 5000; 0 = all)")
    sent_p.set_defaults(func=cmd_sentence)

    # ---- compare ----------------------------------------------------------
    cmp_p = sub.add_parser("compare", help="Cross-method comparison")
    cmp_p.add_argument(
        "--method-dirs", required=True, dest="method_dirs",
        help=(
            "Comma-separated 'Name:path' pairs pointing to each method's "
            "evaluation_results/ directory (must contain distances.csv)"
        ),
    )
    cmp_p.add_argument("--out-dir", required=True, dest="out_dir")
    cmp_p.add_argument(
        "--vector-paths", default="", dest="vector_paths",
        help="Optional 'Name:path' pairs of variety_vectors.npz for Procrustes/CKA",
    )
    cmp_p.add_argument("--mantel-perms", type=int, default=999, dest="mantel_perms",
                       help="Mantel test permutations (default 999; 0 = skip)")
    cmp_p.set_defaults(func=cmd_compare)

    return parser


def main() -> None:
    parser = _build_parser()
    # Allow omitting the 'run' sub-command when --vectors is the first flag
    if len(sys.argv) > 1 and sys.argv[1].startswith("--"):
        sys.argv.insert(1, "run")
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
