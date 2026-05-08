"""
Build every available gold reference matrix for a given varieties module.

Builders covered:
    uriel              — URIEL/lang2vec, 6 sub-types
    glottolog_tree     — hand-coded Glottolog 5.x tree (LCA distance)
    asjp_lexical       — ASJP LDND (lexical surface)
    asjp_genealogy     — ASJP automated classification tree
    grambank           — Grambank typology (CLDF)
    phoible            — PHOIBLE phoneme inventories (Jaccard)
    lexibank           — Lexibank lexicon-features (cosine)
    geographic_glottolog — Glottolog coordinates (Haversine)

Builders that need network access (Grambank/PHOIBLE/Lexibank/ASJP) cache
their downloads under ``~/.cache/edoardo_gold/`` (override via
``EDOARDO_GOLD_CACHE`` env var).

CLI:
    python -m edoardo._shared_gold_builders.build_all \\
        --varieties-module edoardo.exp1_uriel_native.varieties \\
        --out-dir edoardo/exp1_uriel_native/gold_references/matrices
    python -m edoardo._shared_gold_builders.build_all \\
        --only uriel,glottolog_tree,geographic_glottolog \\
        --varieties-module ...
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from edoardo._shared_gold_builders import (
    build_uriel,
    build_glottolog_tree,
    build_asjp_lexical,
    build_asjp_genealogy,
    build_grambank,
    build_phoible,
    build_lexibank,
    build_geographic_glottolog,
)


BUILDERS = {
    "uriel":               build_uriel.main,
    "glottolog_tree":      build_glottolog_tree.main,
    "asjp_lexical":        build_asjp_lexical.main,
    "asjp_genealogy":      build_asjp_genealogy.main,
    "grambank":            build_grambank.main,
    "phoible":             build_phoible.main,
    "lexibank":            build_lexibank.main,
    "geographic_glottolog": build_geographic_glottolog.main,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--varieties-module", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--only", default=None,
                    help="Comma-separated subset of builders to run")
    ap.add_argument("--skip", default=None,
                    help="Comma-separated builders to exclude")
    args = ap.parse_args(argv)

    targets = list(BUILDERS.keys())
    if args.only:
        only = set(args.only.split(","))
        targets = [t for t in targets if t in only]
    if args.skip:
        skip = set(args.skip.split(","))
        targets = [t for t in targets if t not in skip]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Varieties module : {args.varieties_module}")
    print(f"Output dir       : {args.out_dir}")
    print(f"Builders         : {targets}\n")

    errors = 0
    for name in targets:
        print(f"[{name}]")
        try:
            rc = BUILDERS[name](["--varieties-module", args.varieties_module,
                                 "--out-dir", str(args.out_dir)])
            if rc != 0:
                errors += 1
        except SystemExit as exc:
            if exc.code not in (0, None):
                errors += 1
        except Exception:
            traceback.print_exc()
            errors += 1
        print()

    print(f"Done.  {len(targets) - errors}/{len(targets)} builders succeeded.")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
