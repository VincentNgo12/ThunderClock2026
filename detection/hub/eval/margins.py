# File: margins.py
# Purpose: Threshold margins from harness outputs: T3 rotation peaks at the fire (positives),
#          C3/T4 still fractions at the decision, and which hard negatives got a T3 fire at all.
# Contributors: Unassigned
"""PYTHONPATH=.:simulation simulation/.venv/bin/python -m hub.eval.margins DIR [--arm D]"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--arm", default="D")
    args = ap.parse_args(argv)
    rot = defaultdict(list)
    still = defaultdict(list)
    t3_runs = defaultdict(lambda: [0, 0])
    for d in args.dirs:
        for line in (Path(d) / "runs.jsonl").read_text().splitlines():
            r = json.loads(line)
            if "error" in r or args.arm not in r["arms"]:
                continue
            evs = r["arms"][args.arm]
            sid = r["scenario"]
            t3 = [e for e in evs if e["kind"] == "suspected_fall" and e.get("path") == "topple"]
            fires = (r.get("status", {}).get(args.arm) or {}).get("t3_fires", 0)
            t3_runs[sid][0] += fires > 0
            t3_runs[sid][1] += 1
            for e in t3:
                peak = (e.get("evidence") or {}).get("rot_peak_rad_s")
                if peak is not None:
                    rot[sid].append(peak)
            for e in evs:
                ev = e.get("evidence") or {}
                if e["kind"] in ("fall_confirmed", "man_down_critical") and e.get("path") in ("topple", "man_down") \
                        and "still_frac" in ev:
                    still[(sid, e["kind"], e.get("rule"))].append(ev["still_frac"])
    print("T3 rotation peak at the suspected_fall (topple path), rad/s; threshold 3.0")
    for sid, v in sorted(rot.items()):
        v = np.array(v)
        print(f"  {sid:12s} n={len(v):3d} min={v.min():.2f} p10={np.quantile(v, 0.1):.2f} median={np.median(v):.2f} "
              f"below 3.3={int((v < 3.3).sum())} below 3.6={int((v < 3.6).sum())}")
    print("Runs with >= 1 T3 fire (any scenario with a fire):")
    for sid, (k, n) in sorted(t3_runs.items()):
        if k:
            print(f"  {sid:12s} {k}/{n}")
    print("Still fraction at C3/T4 decisions (thresholds: fast 0.90, slow 0.70, T4 0.70)")
    for key, v in sorted(still.items()):
        v = np.array(v)
        print(f"  {key[0]:12s} {key[1]:18s} {key[2]:24s} n={len(v):3d} min={v.min():.3f} median={np.median(v):.3f}")


if __name__ == "__main__":
    main()
