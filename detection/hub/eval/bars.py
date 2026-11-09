# File: bars.py
# Purpose: Check the acceptance bars and regression guard of DETECTION_ARCHITECTURE.md 7 (and the
#          collapse-detection agent file) on harness outputs: arm D against arm B. Prints PASS/FAIL.
# Contributors: Unassigned
"""PYTHONPATH=.:simulation simulation/.venv/bin/python -m hub.eval.bars --base DIR [--apnea DIR]
   [--slowstart DIR] [--shifts DIR]"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from hub.eval.summarize import ALERTS, CAL_S, load, pct, run_metrics

FALLS = ("H-LAD", "H-SCAF", "H-ROOF", "H-HOLE", "H-EXC", "L-SLIP", "L-TRIP", "STAIR", "STRUCK-KD", "ARR", "NET",
         "FALL-GETUP", "SLIP-FRICTION")
HEIGHT = ("H-LAD", "H-SCAF", "H-ROOF", "H-HOLE", "H-EXC")
LEVEL = ("L-SLIP", "L-TRIP", "STAIR", "STRUCK-KD")
NF = ("NF-STUMBLE", "NF-RAIL", "STRUCK-NF")
NO_WORSE = ("A-SIT", "A-EQUIP", "A-JUMP")
NEG = ("A-LIE", "A-NAP", "A-LIEWORK", "A-KNEEL", "A-REBAR", "A-JACK", "A-RUN", "A-CARRY", "A-LADDER",
       "DEVICE-OFF", "HAT-OFF", "STRAP-SLIP", "A-HAMMER", "A-DRILL")


def table(rows, arm):
    out = defaultdict(list)
    for r in rows:
        if "error" not in r and arm in r["arms"]:
            out[r["scenario"]].append(run_metrics(r, arm))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--apnea", default=None)
    ap.add_argument("--slowstart", default=None)
    ap.add_argument("--shifts", default=None)
    args = ap.parse_args(argv)
    rows = load([args.base], "runs.jsonl")
    errors = [r for r in rows if "error" in r]
    D, B = table(rows, "D"), table(rows, "B")
    checks = [("no run errors", f"{len(errors)} errors", not errors)]

    def k(t, sid, key):
        return sum(m[key] for m in t.get(sid, [])), len(t.get(sid, []))

    for sid in ("SYNC", "SYNC-KNEEL"):
        c, n = k(D, sid, "confirmed")
        checks.append((f"{sid} fall_confirmed >= 18/20 (D)", f"{c}/{n}", n > 0 and c >= 18 * n / 20))
        c, n = k(D, sid, "crit90")
        checks.append((f"{sid} man_down_critical <= 90 s >= 19/20 (D)", f"{c}/{n}", n > 0 and c >= 19 * n / 20))
    c, n = k(D, "ARR", "cleared")
    checks.append(("ARR never cleared (D)", f"{c}/{n}", c == 0))
    for sid in FALLS:
        d, n = k(D, sid, "confirmed")
        b, _ = k(B, sid, "confirmed")
        checks.append((f"{sid} D >= B - 1", f"D {d} B {b} /{n}", d >= b - 1))
    for sid in NF:
        d, n = k(D, sid, "confirmed")
        checks.append((f"{sid} fall_confirmed <= 10 % (D)", f"{d}/{n}", d <= 0.1 * n))
    for sid in NO_WORSE:
        d, n = k(D, sid, "alert")
        b, _ = k(B, sid, "alert")
        checks.append((f"{sid} alerts D <= B + 1", f"D {d} B {b} /{n}", d <= b + 1))
    for sid in NEG:
        d, n = k(D, sid, "alert")
        checks.append((f"{sid} alerts <= 1/20 (D)", f"{d}/{n}", d <= max(1, n / 20)))
    h = [m["confirmed"] for s in HEIGHT for m in D.get(s, [])]
    lv = [m["confirmed"] for s in LEVEL for m in D.get(s, [])]
    checks.append(("SIM_SPEC height >= 90 % (D)", f"{sum(h)}/{len(h)}", h and sum(h) >= 0.9 * len(h)))
    checks.append(("SIM_SPEC level >= 80 % (D)", f"{sum(lv)}/{len(lv)}", lv and sum(lv) >= 0.8 * len(lv)))
    arr = D.get("ARR", [])
    checks.append(("SIM_SPEC ARR confirmed >= 80 % (D)", f"{sum(m['confirmed'] for m in arr)}/{len(arr)}",
                   arr and sum(m["confirmed"] for m in arr) >= 0.8 * len(arr)))
    lat = [m["latency"] for s in FALLS if s not in ("FALL-GETUP", "SLIP-FRICTION") for m in D.get(s, [])
           if m["latency"] is not None]
    p95 = pct(lat, 0.95)
    checks.append(("SIM_SPEC latency p95 <= 30 s (D)", f"{p95:.1f} s" if p95 is not None else "-",
                   p95 is not None and p95 <= 30))
    if args.apnea:
        A = table(load([args.apnea], "runs.jsonl"), "D")
        for sid in ("SYNC", "SYNC-KNEEL"):
            c, n = k(A, sid, "crit90")
            checks.append((f"apnea {sid} escalates >= 19/20 (D)", f"{c}/{n}", n > 0 and c >= 19 * n / 20))
    if args.slowstart:
        srows = [r for r in load([args.slowstart], "runs.jsonl") if "error" not in r]
        rec = sum((r["status"].get("D") or {}).get("rate_recoveries", 0) >= 1 for r in srows)
        checks.append(("slow start: falldet latch recovered in every run (D)", f"{rec}/{len(srows)}",
                       srows and rec == len(srows)))
    if args.shifts:
        srows = [r for r in load([args.shifts], "shifts.jsonl") if "error" not in r]
        hours = sum(r["hours"] for r in srows)
        hub_eps = set()
        for r in srows:
            for e in r["arms"]["D"]:
                if e["kind"] in ALERTS and e["t"] > CAL_S and e.get("path") in ("topple", "man_down"):
                    eps = [x for x in r["arms"]["D"] if x.get("episode") == e.get("episode") and x["kind"] in ALERTS]
                    first = min(eps, key=lambda x: x["t"]) if eps else e
                    if first.get("path") in ("topple", "man_down"):
                        hub_eps.add((r["seed"], e.get("episode")))
        checks.append(("shifts: hub-path episodes (opened by topple or man_down) = 0 (D)",
                       f"{len(hub_eps)} in {hours:.1f} h", len(hub_eps) == 0))
    width = max(len(c[0]) for c in checks)
    for name, value, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name.ljust(width)}  {value}")
    print("ALL PASS" if all(c[2] for c in checks) else "SOME BARS FAIL")


if __name__ == "__main__":
    main()
