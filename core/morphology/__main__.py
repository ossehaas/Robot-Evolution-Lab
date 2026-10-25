"""Command line: build robots from genomes.

    python -m core.morphology build prototypes/crawler_q4/genome.json
    python -m core.morphology build genome.json --out some/dir --no-usd
    python -m core.morphology random --count 6 --seed 1 --out runs/random_robots
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from . import Genome, export_all, random_genome


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m core.morphology")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write USD, URDF, STL parts, BOM and a preview for a genome")
    b.add_argument("genome", type=Path)
    b.add_argument("--out", type=Path, default=None, help="output directory (default: <genome dir>/build)")
    for flag in ("usd", "urdf", "print", "preview"):
        b.add_argument(f"--no-{flag}", action="store_true")
    r = sub.add_parser("random", help="sample random genomes and build them (quick way to see the design space)")
    r.add_argument("--count", type=int, default=6)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", type=Path, default=Path("runs/random_robots"))
    args = parser.parse_args()

    if args.cmd == "build":
        genome = Genome.load(args.genome)
        out = args.out or args.genome.parent / "build"
        spec = export_all(
            genome, out, usd=not args.no_usd, urdf=not args.no_urdf, print_files=not args.no_print, preview=not args.no_preview
        )
        print(spec.summary())
        print(f"written to {out}")
    else:
        rng = random.Random(args.seed)
        for i in range(args.count):
            g = random_genome(rng, name=f"random_{args.seed}_{i}")
            spec = export_all(g, args.out / g.name, usd=False, urdf=False, print_files=False)
            print(spec.summary().splitlines()[0])
        print(f"previews in {args.out}")


if __name__ == "__main__":
    main()
