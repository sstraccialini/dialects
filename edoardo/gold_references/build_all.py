"""
Build every gold reference matrix in one go.

Each builder is run in turn; failures are reported but do not abort the
others.  Output: one ``.npz`` per gold under ``matrices/``.

Run from the repo root:
    python -m edoardo.gold_references.build_all
    python -m edoardo.gold_references.build_all --skip asjp
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from edoardo.gold_references import (
    build_uriel,
    build_glottolog,
    build_expert,
    build_asjp,
)


BUILDERS = {
    "uriel":     build_uriel.main,
    "glottolog": build_glottolog.main,
    "expert":    build_expert.main,
    "asjp":      build_asjp.main,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).parent / "matrices")
    ap.add_argument("--skip", nargs="+", choices=list(BUILDERS), default=[])
    ap.add_argument("--only", nargs="+", choices=list(BUILDERS), default=None)
    args = ap.parse_args(argv)

    targets = list(args.only or BUILDERS.keys())
    targets = [t for t in targets if t not in args.skip]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Building gold references → {args.out_dir}")
    print(f"  builders: {targets}\n")

    errors = 0
    for name in targets:
        print(f"[{name}]")
        try:
            rc = BUILDERS[name](["--out-dir", str(args.out_dir)])
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
