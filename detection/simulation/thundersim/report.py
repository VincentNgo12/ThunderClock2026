# File: report.py
# Purpose: Aggregate run results into SIM_SPEC 5.3 metrics, check the 5.4 pass bars and
#          write a Markdown report. Every number is labelled as simulated.
# Contributors: Unassigned

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from .scenarios import CATALOGUE

PASS_BARS = {  # SIM_SPEC 5.4 starting bars [design], applied to the target profile
    "height_sensitivity": 0.90,       # H-*
    "level_sensitivity": 0.80,        # L-*, STAIR, STRUCK-KD
    "arr_non_normal": 0.90,
    "arr_confirmed": 0.80,
    "near_fall_false_alarm_max": 0.10,
    "false_alarms_per_8h_max": 1.0,
    "false_alarms_per_8h_ci_upper_max": 2.0,
    "latency_p95_max_s": 30.0,
}
HEIGHT = {"H-LAD", "H-SCAF", "H-ROOF", "H-HOLE", "H-EXC"}
KNOWN_GAPS = {"NF-STUMBLE": "stumble with no impact; near-fall signal is short", "SYNC": "slow collapse, no free fall, Stage 1 may stay silent"}
LEVEL = {"L-SLIP", "L-TRIP", "STAIR", "STRUCK-KD"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def poisson_rate_ci(count: int, exposure: float) -> tuple[float, float]:
    """Exact (Garwood) 95 % CI for a count over an exposure, as a rate."""
    lo = stats.chi2.ppf(0.025, 2 * count) / 2 if count > 0 else 0.0
    hi = stats.chi2.ppf(0.975, 2 * count + 2) / 2
    return (lo / exposure, hi / exposure)


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def scenario_table(runs: list[dict]) -> dict:
    """Per (model, profile, scenario) metrics."""
    groups = defaultdict(list)
    for r in runs:
        if "error" in r:
            continue
        groups[(r["model"], r["profile"], r["scenario"], r.get("dead_slot"))].append(r)
    table = {}
    for (model, profile, sid, dead), rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        n = len(rs)
        outputs = [r["output"] for r in rs]
        passed = sum(r["passed"] for r in rs)
        confirmed = sum(o == "fall_confirmed" for o in outputs)
        near = sum(o == "near_fall" for o in outputs)
        lat = [r["latency_s"] for r in rs if r["latency_s"] is not None and r["expected"] == "fall_confirmed"]
        sup = [r["supervisor_latency_s"] for r in rs if r.get("supervisor_latency_s") is not None and r["expected"] == "fall_confirmed"]
        table[(model, profile, sid, dead)] = {
            "model": model, "profile": profile, "scenario": sid, "dead_slot": dead, "n": n,
            "expected": rs[0]["expected"] if len({r["expected"] for r in rs}) == 1 else "mixed",
            "report_only": rs[0]["report_only"], "pass_rate": passed / n, "pass_ci": wilson(passed, n),
            "confirmed_rate": confirmed / n, "confirmed_ci": wilson(confirmed, n), "near_fall_rate": near / n,
            "normal_rate": sum(o == "normal" for o in outputs) / n,
            "partial": sum(r.get("partial", False) for r in rs),
            "latency_p50": float(np.median(lat)) if lat else None,
            "latency_p95": float(np.percentile(lat, 95)) if lat else None,
            "sup_latency_p50": float(np.median(sup)) if sup else None,
            "sup_latency_p95": float(np.percentile(sup, 95)) if sup else None,
            "supervisor_alert_rate": sum(r["supervisor_alerts"] > 0 for r in rs) / n,
            "stage1_mean": float(np.mean([r["stage1_triggers"] for r in rs])),
            "loss_pct": float(np.mean([r["link"].get("loss_pct", 0) for r in rs])),
            "saturated_runs": sum(r["saturated_frames"] > 0 for r in rs),
            "errors": 0,
        }
    return table


def shift_summary(shifts: list[dict]) -> dict:
    groups = defaultdict(list)
    for s in shifts:
        if "error" not in s:
            groups[(s["model"], s["profile"])].append(s)
    out = {}
    for (model, profile), ss in groups.items():
        hours = sum(s["hours"] for s in ss)
        fa = sum(s["false_alarms"] for s in ss)
        sup = sum(s["supervisor_alerts"] for s in ss)
        trig = sum(s["stage1_triggers"] for s in ss)
        lo, hi = poisson_rate_ci(fa, hours / 8.0)
        causes = defaultdict(int)
        by_activity = defaultdict(int)
        for s in ss:
            for a in s["alerts"]:
                causes[a["detail"].get("rule", a["detail"].get("state", a["kind"]))] += 1
                if "activity" in a:
                    by_activity[a["activity"]] += 1
        out[(model, profile)] = {"model": model, "profile": profile, "shifts": len(ss), "hours": round(hours, 2),
                                 "false_alarms": fa, "fa_per_8h": fa / (hours / 8) if hours else None,
                                 "fa_per_8h_ci": (lo, hi), "supervisor_per_8h": sup / (hours / 8) if hours else None,
                                 "stage1_per_h": trig / hours if hours else None, "causes": dict(causes),
                                 "by_activity": dict(by_activity)}
    return out


def pass_bars(table: dict, shifts: dict, model: str, profile: str = "target") -> list[tuple[str, str, bool | None]]:
    rows = [v for v in table.values() if v["model"] == model and v["profile"] == profile and v["dead_slot"] is None]
    by = {v["scenario"]: v for v in rows}

    def pooled(ids):
        sel = [by[i] for i in ids if i in by]
        n = sum(v["n"] for v in sel)
        k = sum(round(v["confirmed_rate"] * v["n"]) for v in sel)
        return (k / n if n else None), n

    checks = []
    h, n = pooled(HEIGHT)
    checks.append(("Falls from height: fall_confirmed >= 90 %", f"{_pct(h)} (n={n})", None if h is None else h >= PASS_BARS["height_sensitivity"]))
    level, n = pooled(LEVEL)
    checks.append(("Same-level, stairs, struck: fall_confirmed >= 80 %", f"{_pct(level)} (n={n})", None if level is None else level >= PASS_BARS["level_sensitivity"]))
    if "ARR" in by:
        a = by["ARR"]
        non_normal = 1 - a["normal_rate"]
        checks.append(("ARR: non-normal >= 90 %", _pct(non_normal), non_normal >= PASS_BARS["arr_non_normal"]))
        checks.append(("ARR: fall_confirmed >= 80 %", _pct(a["confirmed_rate"]), a["confirmed_rate"] >= PASS_BARS["arr_confirmed"]))
    nf = [by[i] for i in by if CATALOGUE[i].group == "near_fall" and by[i]["expected"] == "near_fall"]
    if nf:
        worst = max(v["confirmed_rate"] for v in nf)
        checks.append(("NF-*: fall_confirmed <= 10 % (worst scenario)", _pct(worst), worst <= PASS_BARS["near_fall_false_alarm_max"]))
    lat = [v["latency_p95"] for v in rows if v["latency_p95"] is not None and not v["report_only"]]
    if lat:
        checks.append(("Alert latency p95 <= 30 s (worst scenario)", f"{max(lat):.1f} s", max(lat) <= PASS_BARS["latency_p95_max_s"]))
    sh = shifts.get((model, profile))
    if sh and sh["fa_per_8h"] is not None:
        ok = sh["fa_per_8h"] <= PASS_BARS["false_alarms_per_8h_max"] and sh["fa_per_8h_ci"][1] <= PASS_BARS["false_alarms_per_8h_ci_upper_max"]
        checks.append(("False alarms <= 1 per 8 h (95 % CI upper <= 2)",
                       f"{sh['fa_per_8h']:.2f} per 8 h (CI {sh['fa_per_8h_ci'][0]:.2f}-{sh['fa_per_8h_ci'][1]:.2f}, {sh['hours']:.0f} h)", ok))
    return checks


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.0f} %"


def write_markdown(path: Path, table: dict, shifts: dict, title: str, notes: list[str]) -> None:
    lines = [f"# {title}", "", "**Simulated data only.** These numbers describe behaviour on ThunderSim scenarios; "
             "they are a necessary, not sufficient, check before real-world pilot data (SIM_SPEC 5.4).", ""]
    lines += [f"- {n}" for n in notes] + [""]
    combos = sorted({(v["model"], v["profile"]) for v in table.values()} | set(shifts))
    for model, profile in combos:
        lines += [f"## {model} on `{profile}` firmware", ""]
        if profile in ("target", "target_8g"):
            lines += ["**SIM_SPEC 5.4 pass bars**", "", "| Check | Result | Pass |", "|---|---|---|"]
            for name, value, ok in pass_bars(table, shifts, model, profile):
                lines.append(f"| {name} | {value} | {'n/a' if ok is None else ('yes' if ok else '**no**')} |")
            lines.append("")
        lines += ["| Scenario | Expected | n | Pass (95 % Wilson CI) | fall_confirmed | near_fall | normal | Alert latency p50 / p95 s | With 15 s check-in p50 / p95 s | Supervisor alert | Stage-1 / run |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for v in sorted((v for v in table.values() if v["model"] == model and v["profile"] == profile),
                        key=lambda v: (CATALOGUE[v["scenario"]].group, v["scenario"], str(v["dead_slot"]))):
            name = v["scenario"] + (f" (slot {v['dead_slot']} dead)" if v["dead_slot"] is not None else "")
            name += " (report only)" if v["report_only"] else ""
            lo, hi = v["pass_ci"]
            lat = "-" if v["latency_p50"] is None else f"{v['latency_p50']:.1f} / {v['latency_p95']:.1f}"
            sup = "-" if v.get("sup_latency_p50") is None else f"{v['sup_latency_p50']:.1f} / {v['sup_latency_p95']:.1f}"
            lines.append(f"| {name} | {v['expected']} | {v['n']} | {100 * v['pass_rate']:.0f} % ({100 * lo:.0f}-{100 * hi:.0f}) | "
                         f"{100 * v['confirmed_rate']:.0f} % | {100 * v['near_fall_rate']:.0f} % | {100 * v['normal_rate']:.0f} % | {lat} | {sup} | "
                         f"{100 * v['supervisor_alert_rate']:.0f} % | {v['stage1_mean']:.1f} |")
        gaps = [v for v in table.values() if v["model"] == model and v["profile"] == profile and v["dead_slot"] is None
                and v["scenario"] in KNOWN_GAPS]
        if gaps:
            lines += ["", "**Known gaps** (listed separately, as agreed with research-95):", ""]
            for v in sorted(gaps, key=lambda v: v["scenario"]):
                lo, hi = v["pass_ci"]
                lines.append(f"- {v['scenario']} ({KNOWN_GAPS[v['scenario']]}): expected {v['expected']}, pass "
                             f"{100 * v['pass_rate']:.0f} % (95 % CI {100 * lo:.0f}-{100 * hi:.0f}, n={v['n']}); outputs "
                             f"fall_confirmed {100 * v['confirmed_rate']:.0f} %, near_fall {100 * v['near_fall_rate']:.0f} %, "
                             f"normal {100 * v['normal_rate']:.0f} %; Stage-1 triggers per run {v['stage1_mean']:.1f}.")
        sh = shifts.get((model, profile))
        if sh:
            lo, hi = sh["fa_per_8h_ci"]
            lines += ["", f"**Fall-free shifts:** {sh['shifts']} shifts, {sh['hours']:.1f} h, {sh['false_alarms']} false alarms = "
                      f"**{sh['fa_per_8h']:.2f} per 8 h** (95 % CI {lo:.2f}-{hi:.2f}); after worker check-in "
                      f"{sh['supervisor_per_8h']:.2f} per 8 h; Stage-1 triggers {sh['stage1_per_h']:.1f} per hour. "
                      f"Causes: {sh['causes'] or 'none'}." + (f" During: {sh['by_activity']}." if sh.get('by_activity') else "")]
        lines.append("")
    path.write_text("\n".join(lines))
