# File: paths.py
# Purpose: Which path catches each collapse in arm D: falldet impact (T1/C1) vs topple (T3/C3) vs
#          man-down only. Uses arm B (falldet alone) as the impact path's own verdict.
# Contributors: Unassigned
"""PYTHONPATH=.:simulation simulation/.venv/bin/python -m hub.eval.paths DIR [DIR ...]"""
import argparse
from collections import defaultdict

from hub.eval.summarize import load, run_metrics, wilson


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--scenarios", default="SYNC,SYNC-KNEEL,X-DISMISS")
    args = ap.parse_args(argv)
    wanted = args.scenarios.split(",")
    agg = defaultdict(lambda: defaultdict(int))
    for r in load(args.dirs, "runs.jsonl"):
        if "error" in r or r["scenario"] not in wanted or "D" not in r["arms"]:
            continue
        key = (r["variant"], r["scenario"])
        a = agg[key]
        a["n"] += 1
        evs = r["arms"]["D"]
        t_min = r["params"]["collapse_onset"] - 1.0 if r["scenario"] == "X-DISMISS" else 5.0
        evs = [e for e in evs if e["t"] > t_min]
        # The arbiter merges suspected_fall events within 10 s, so read T1 from falldet alone (arm B).
        b_evs = [e for e in r["arms"].get("B", []) if e["t"] > t_min]
        t1 = any(e["kind"] == "suspected_fall" for e in b_evs)
        t3 = (r.get("status", {}).get("D") or {}).get("t3_fires", 0) > 0 if r["scenario"] != "X-DISMISS" \
            else any(e["kind"] == "suspected_fall" and e.get("path") == "topple" for e in evs)
        b_conf = run_metrics(r, "B")["confirmed"] if "B" in r["arms"] else None
        m = run_metrics(r, "D")
        a["T1 (Stage-1) fired"] += t1
        a["T3 (topple) fired"] += t3
        a["both fired"] += t1 and t3
        if b_conf is not None:
            a["impact path confirms alone (arm B)"] += b_conf
        a["D confirmed"] += m["confirmed"]
        a[f"D first confirmation: {m['first_path'] or 'none'}"] += 1
        a["D man_down_critical <= 90 s"] += m["crit90"]
    for key in sorted(agg):
        a = agg[key]
        n = a.pop("n")
        print(f"{key[0]} {key[1]} (n = {n})")
        for name in sorted(a):
            lo, hi = wilson(a[name], n)
            print(f"  {name:40s} {a[name]:3d}/{n} [{lo:.2f}-{hi:.2f}]")


if __name__ == "__main__":
    main()
