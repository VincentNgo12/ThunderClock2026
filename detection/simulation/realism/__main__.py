# File: realism/__main__.py
# Purpose: Command line for the realism scorecard.
# Contributors: Unassigned
"""Usage (from detection/simulation/):
    .venv/bin/python -m realism score --out <scratch dir> [--events 40] [--act 20] [--jobs 4]
    .venv/bin/python -m realism compare <before dir> <after dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="realism", description="ThunderSim realism scorecard against real recordings")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="build the scorecard (writes scorecard.json/.md and plots to --out)")
    s.add_argument("--out", required=True)
    s.add_argument("--events", type=int, default=40, help="sim runs per event category")
    s.add_argument("--act", type=int, default=20, help="sim clips per activity category")
    s.add_argument("--jobs", type=int, default=4)
    s.add_argument("--no-falldet", action="store_true")
    s.add_argument("--only", default="", help="comma-separated categories")
    s.add_argument("--cache", default=None, help="real-side cache dir (default: <out>/../cache)")
    s.add_argument("--set", action="append", default=[], metavar="MODULE.NAME=VALUE",
                   help="override a module constant in the sim workers, e.g. thundersim.falls.PIVOT_RESIST=0.0")
    s.add_argument("--seed-offset", type=int, default=0,
                   help="shift the dev seeds (e.g. 400 for a second seed set; stays below 1000000)")
    r = sub.add_parser("rescore", help="recompute scorecard.json/.md from a run's saved sim_side.pkl")
    r.add_argument("out")
    c = sub.add_parser("compare", help="before/after table for two scorecard dirs")
    c.add_argument("before")
    c.add_argument("after")
    args = p.parse_args(argv)
    if args.cmd == "score":
        from .score import run_score, tables

        import ast

        only = {x for x in args.only.split(",") if x} or None
        overrides = {}
        for item in args.set:
            key, value = item.split("=", 1)
            overrides[key.strip()] = ast.literal_eval(value.strip())
        card = run_score(Path(args.out), args.events, args.act, args.jobs, not args.no_falldet, only,
                         Path(args.cache) if args.cache else None, overrides, args.seed_offset)
        print(tables(card, top=20))
        print("meta", card["meta"])
    elif args.cmd == "rescore":
        from .score import rescore

        rescore(Path(args.out))
    else:
        from .score import compare

        print(compare(Path(args.before), Path(args.after)))


if __name__ == "__main__":
    main()
