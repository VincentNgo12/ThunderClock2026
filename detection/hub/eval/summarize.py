# File: summarize.py
# Purpose: Tables for harness outputs: per scenario x arm, by event kind, with Wilson 95 % CIs,
#          first-alert latency and path; shifts with alerts before check-in per 8 h (exact Poisson).
# Contributors: Unassigned
"""PYTHONPATH=.:simulation simulation/.venv/bin/python -m hub.eval.summarize DIR [DIR ...] [--md OUT.md]"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

ALERTS = ("fall_confirmed", "man_down_warning", "man_down_critical")
CAL_S = 5.0


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def poisson_ci(k, alpha=0.05):
    from scipy.stats import chi2
    lo = 0.0 if k == 0 else chi2.ppf(alpha / 2, 2 * k) / 2
    hi = chi2.ppf(1 - alpha / 2, 2 * k + 2) / 2
    return lo, hi


def frac(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} [{lo:.2f}-{hi:.2f}]"


def pct(values, q):
    if not values:
        return None
    v = sorted(values)
    i = min(len(v) - 1, max(0, int(math.ceil(q * len(v))) - 1))
    return v[i]


def load(dirs, name):
    rows = []
    for d in dirs:
        p = Path(d) / name
        if p.exists():
            rows += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return rows


def run_metrics(r, arm):
    evs = r["arms"].get(arm, [])
    t_min = CAL_S
    if r["scenario"] == "X-DISMISS":
        t_min = r["params"]["collapse_onset"] - 1.0
    after = [e for e in evs if e["t"] > t_min]
    alerts = [e for e in after if e["kind"] in ALERTS]
    ref = r.get("ref_time")
    m = {"confirmed": any(e["kind"] == "fall_confirmed" for e in after),
         "alert": bool(alerts), "n_alerts": len(alerts),
         "episodes": len({e["episode"] for e in alerts if e.get("episode") is not None}) or len(alerts),
         "md_warn": any(e["kind"] == "man_down_warning" for e in after),
         "md_crit": any(e["kind"] == "man_down_critical" for e in after),
         "cleared": any(e["kind"] == "cleared" for e in after),
         "near_fall": any(e["kind"] == "near_fall" for e in after),
         "first_path": None, "latency": None, "crit_latency": None, "paths": sorted({e.get("path") or "?" for e in alerts})}
    if alerts:
        first = min(alerts, key=lambda e: e["t_laptop"])
        m["first_path"] = first.get("path") or "falldet"
        if ref is not None:
            m["latency"] = first["t_laptop"] - ref
    crit = [e for e in after if e["kind"] == "man_down_critical"]
    if crit and ref is not None:
        m["crit_latency"] = min(e["t_laptop"] for e in crit) - ref
    m["crit90"] = m["crit_latency"] is not None and m["crit_latency"] <= 90.0
    return m


def scenario_table(rows):
    groups = defaultdict(list)
    for r in rows:
        if "error" in r:
            continue
        for arm in r["arms"]:
            key = (r["variant"], r.get("errors"), r.get("dead_slot"), r.get("fast"), r["scenario"], arm)
            groups[key].append(run_metrics(r, arm))
    lines = ["| variant | errors | dead | scenario | arm | fall_confirmed | any alert | md_critical <= 90 s | cleared | "
             "latency med / p95 s | first path |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        variant, errors, dead, fast, sid, arm = key
        ms = groups[key]
        n = len(ms)
        lat = [m["latency"] for m in ms if m["latency"] is not None]
        paths = defaultdict(int)
        for m in ms:
            if m["first_path"]:
                paths[m["first_path"]] += 1
        lat_s = "-" if not lat else f"{pct(lat, 0.5):.1f} / {pct(lat, 0.95):.1f}"
        lines.append(f"| {variant}{'' if fast else ' (protobuf)'} | {errors} | {'-' if dead is None else dead} | {sid} | {arm} | "
                     f"{frac(sum(m['confirmed'] for m in ms), n)} | {frac(sum(m['alert'] for m in ms), n)} | "
                     f"{frac(sum(m['crit90'] for m in ms), n)} | {sum(m['cleared'] for m in ms)} | {lat_s} | "
                     f"{', '.join(f'{p} {c}' for p, c in sorted(paths.items()))} |")
    return "\n".join(lines)


def shift_table(rows):
    agg = defaultdict(lambda: {"hours": 0.0, "alerts": 0, "episodes": 0, "by_path": defaultdict(int),
                               "by_activity": defaultdict(int), "t3": 0, "cpu": 0.0, "n": 0, "suspected": 0})
    for r in rows:
        if "error" in r:
            continue
        for arm, evs in r["arms"].items():
            a = agg[arm]
            a["n"] += 1
            a["hours"] += r["hours"]
            alerts = [e for e in evs if e["kind"] in ALERTS and e["t"] > CAL_S]
            a["alerts"] += len(alerts)
            eps = {e["episode"] for e in alerts if e.get("episode") is not None}
            a["episodes"] += len(eps) if eps else len(alerts)
            a["suspected"] += sum(e["kind"] == "suspected_fall" for e in evs)
            for e in alerts:
                a["by_path"][f"{e.get('path') or 'falldet'}:{e['kind']}"] += 1
                a["by_activity"][e.get("activity", "?")] += 1
            a["t3"] += (r.get("status", {}).get(arm) or {}).get("t3_fires", 0)
            a["cpu"] += r["cpu_s"].get(arm, 0.0)
    lines = ["| arm | shifts | hours | alert events | per 8 h [95 % CI] | episodes (check-ins) | per 8 h [95 % CI] | "
             "by path:kind | by activity | T3 fires/h | CPU s/h |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in sorted(agg):
        a = agg[arm]
        h = a["hours"] or 1e-9
        lo, hi = poisson_ci(a["alerts"])
        elo, ehi = poisson_ci(a["episodes"])
        lines.append(f"| {arm} | {a['n']} | {a['hours']:.1f} | {a['alerts']} | {a['alerts'] / h * 8:.2f} "
                     f"[{lo / h * 8:.2f}-{hi / h * 8:.2f}] | {a['episodes']} | {a['episodes'] / h * 8:.2f} "
                     f"[{elo / h * 8:.2f}-{ehi / h * 8:.2f}] | "
                     f"{', '.join(f'{k} {v}' for k, v in sorted(a['by_path'].items())) or '-'} | "
                     f"{', '.join(f'{k} {v}' for k, v in sorted(a['by_activity'].items())) or '-'} | "
                     f"{a['t3'] / h:.2f} | {a['cpu'] / h:.1f} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--md", default=None)
    args = ap.parse_args(argv)
    runs, shifts = load(args.dirs, "runs.jsonl"), load(args.dirs, "shifts.jsonl")
    parts = []
    errors = [r for r in runs + shifts if "error" in r]
    if errors:
        parts.append(f"**{len(errors)} runs raised errors**, first: {errors[0]['error']}")
    if runs:
        parts.append(scenario_table(runs))
    if shifts:
        parts.append(shift_table(shifts))
    text = "\n\n".join(parts)
    print(text)
    if args.md:
        Path(args.md).write_text(text + "\n")


if __name__ == "__main__":
    main()
