# File: __main__.py
# Purpose: Command line for ThunderSim: list, run, suite, shifts, tune, export, calibrate, plot.
# Contributors: Unassigned
"""Usage (from simulator/):  .venv/bin/python -m thundersim <command> --help"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .device import PROFILES
from .report import load_jsonl, scenario_table, shift_summary, write_markdown
from .runner import StreamOptions, run_scenario, run_scenario_task, run_shift_task
from .scenarios import CATALOGUE

OUTPUT = Path(__file__).resolve().parents[1] / "output"
GROUPS = ("height", "level", "arrest", "collapse", "near_fall", "negative")


def pick_scenarios(spec: str) -> list[str]:
    if spec == "all":
        return list(CATALOGUE)
    chosen = []
    for item in spec.split(","):
        item = item.strip()
        if item in GROUPS:
            chosen += [s.id for s in CATALOGUE.values() if s.group == item]
        elif item in CATALOGUE:
            chosen.append(item)
        else:
            raise SystemExit(f"Unknown scenario or group {item!r}; see `list`")
    return chosen


def cmd_list(args) -> None:
    for group in GROUPS:
        print(f"\n[{group}]")
        for s in CATALOGUE.values():
            if s.group == group:
                flag = " (report only)" if s.report_only else ""
                print(f"  {s.id:14s} expect {s.expected:14s} {s.description}{flag}")
    print("\nDevice profiles:")
    for p in PROFILES.values():
        print(f"  {p.name:14s} {p.description}")


def cmd_run(args) -> None:
    result = run_scenario(args.scenario, args.seed, args.model, StreamOptions(profile=args.profile, fast=args.fast,
                          radio=args.radio), obs_s=args.obs, dead_slot=args.dead_slot)
    print(f"{result.scenario} seed {result.seed} on {result.profile} with {result.model}: expected {result.expected}, "
          f"got {result.output} -> {'PASS' if result.passed else 'FAIL'}; latency {result.latency_s} s; link {result.link}")
    print("ground truth:", [(e["t"], e["name"]) for e in result.truth])
    for e in result.events:
        print(f"  t={e['t']:7.2f}s  {e['kind']:18s} conf={e['confidence']:.2f} {json.dumps(e['detail'], default=str)[:160]}")
    if args.plot:
        from .plot import plot_run

        print("plot:", plot_run(args.scenario, args.seed, Path(args.plot), result.events, args.obs))


def cmd_suite(args) -> None:
    out = Path(args.out or OUTPUT / f"suite_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    scenarios = pick_scenarios(args.scenarios)
    seeds = [args.seed_base + i for i in range(args.runs)]
    dead = [None] if args.dead_slot is None else [int(x) for x in args.dead_slot.split(",")]
    tasks = [(sid, seed, model, StreamOptions(profile=profile, fast=args.fast), args.obs, d, None)
             for model in args.models.split(",") for profile in args.profiles.split(",")
             for sid in scenarios for seed in seeds for d in dead]
    print(f"{len(tasks)} runs -> {out} ({args.jobs} processes)")
    started = time.time()
    runs_path = out / "runs.jsonl"
    with open(runs_path, "w") as handle, ProcessPoolExecutor(args.jobs) as pool:
        for i, result in enumerate(pool.map(run_scenario_task, tasks, chunksize=2), 1):
            handle.write(json.dumps(result, default=str) + "\n")
            if i % 50 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} runs, {time.time() - started:.0f} s")
    finish_report(out, notes=[f"{args.runs} randomised runs per scenario (seeds {seeds[0]}..{seeds[-1]}), "
                              f"{'fast transport' if args.fast else 'protobuf + backend decoder'}, observation {args.obs:.0f} s."])


def cmd_shifts(args) -> None:
    out = Path(args.out or OUTPUT / f"shifts_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(args.seed_base + i, args.hours, model, StreamOptions(profile=profile, fast=not args.protobuf))
             for model in args.models.split(",") for profile in args.profiles.split(",") for i in range(args.count)]
    print(f"{len(tasks)} shifts of {args.hours} h -> {out}")
    started = time.time()
    with open(out / "shifts.jsonl", "w") as handle, ProcessPoolExecutor(args.jobs) as pool:
        for i, result in enumerate(pool.map(run_shift_task, tasks), 1):
            handle.write(json.dumps(result, default=str) + "\n")
            print(f"  {i}/{len(tasks)} shifts, {time.time() - started:.0f} s: {result.get('model')} {result.get('profile')} "
                  f"FA={result.get('false_alarms')} stage1={result.get('stage1_triggers')} {result.get('error', '')}")
    finish_report(out, notes=[f"{args.count} fall-free shifts of {args.hours} h per model/profile (SIM_SPEC 5.2 mix)."])


def finish_report(out: Path, notes: list[str], extra_dirs: list[Path] = ()) -> None:
    runs, shifts = [], []
    for d in [out, *extra_dirs]:
        if (d / "runs.jsonl").exists():
            runs += load_jsonl(d / "runs.jsonl")
        if (d / "shifts.jsonl").exists():
            shifts += load_jsonl(d / "shifts.jsonl")
    errors = [r for r in runs + shifts if "error" in r]
    table = scenario_table(runs)
    summary = shift_summary(shifts)
    if errors:
        notes = notes + [f"{len(errors)} runs raised errors (see runs.jsonl 'error'): {errors[0]['error'][:200]}"]
    write_markdown(out / "report.md", table, summary, "ThunderSim results", notes)
    (out / "summary.json").write_text(json.dumps({"scenarios": [v for v in table.values()],
                                                  "shifts": [v for v in summary.values()]}, indent=1, default=str))
    print((out / "report.md").read_text())


def cmd_report(args) -> None:
    dirs = [Path(d) for d in args.dirs]
    out = Path(args.out) if args.out else dirs[0]
    out.mkdir(parents=True, exist_ok=True)
    finish_report(out, notes=[f"Combined from {', '.join(str(d) for d in dirs)}."], extra_dirs=[d for d in dirs if d != out])


def cmd_tune(args) -> None:
    from .tune import tune

    result = tune(Path(args.out or OUTPUT / "tune"), args.trials, args.seeds, args.profile, args.obs, args.shift_hours, args.jobs)
    for label in ("default", "best"):
        r = result[label]
        print(f"{label:8s} score {r['score']:.3f} sens {r['sensitivity']:.2f} near-fall {r['near_fall_acc']:.2f} "
              f"neg FA {r['negative_fa']:.2f} shift FA/8h {r['shift_fa_per_8h']:.2f}")
    print("tuned config written to", Path(args.out or OUTPUT / "tune") / "tuned_baseline.json",
          "- judge it on TEST seeds: suite --models baseline:<that file> --seed-base 1000000")


def cmd_export(args) -> None:
    from .tune import export

    n = export(Path(args.out), pick_scenarios(args.scenarios), [args.seed_base + i for i in range(args.runs)],
               args.profile, args.obs, args.jobs)
    print(f"exported {n} runs to {args.out} (simulated data; see manifest.json)")


def cmd_calibrate(args) -> None:
    from .calibrate import calibrate

    result = calibrate(runs=args.runs)
    print(json.dumps({k: v for k, v in result.items() if k != "candidates"}, indent=1))
    for c in result["candidates"]:
        print(c)


def cmd_posture(args) -> None:
    from .posture import POSTURES, plot_posture_example, posture_suite, write_posture_report

    out = Path(args.out or OUTPUT / f"posture_{time.strftime('%Y%m%d_%H%M%S')}")
    scenarios = list(POSTURES) if args.scenarios == "all" else args.scenarios.split(",")
    seeds = [args.seed_base + i for i in range(args.runs)]
    rows = posture_suite(out, scenarios, seeds, args.profiles.split(","), args.jobs)
    write_posture_report(rows, out, [f"{args.runs} seeds per scenario (seed base {args.seed_base}); every run starts with the "
                                     f"5 s neutral stand and three functional-calibration bends; metrics exclude those."])
    for sid in ("P-STOOP-LIFT", "P-SLOUCH"):
        if sid in scenarios:
            plot_posture_example(out, sid, seeds[0])
    print((out / "posture_report.md").read_text())


def cmd_plot(args) -> None:
    from .plot import plot_run

    print(plot_run(args.scenario, args.seed, Path(args.out), None, args.obs))


def main(argv=None) -> None:
    jobs = max((os.cpu_count() or 2) - 1, 1)
    p = argparse.ArgumentParser(prog="thundersim", description="ThunderClock fall/man-down simulator (simulated data only)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="scenarios and device profiles").set_defaults(fn=cmd_list)

    r = sub.add_parser("run", help="one scenario run, with events")
    r.add_argument("scenario", choices=list(CATALOGUE))
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--model", default="baseline", help="baseline | baseline:<config.json> | falldet | pkg.module:Class")
    r.add_argument("--profile", default="target", choices=list(PROFILES))
    r.add_argument("--radio", default=None)
    r.add_argument("--obs", type=float, default=120.0)
    r.add_argument("--dead-slot", type=int, default=None)
    r.add_argument("--fast", action="store_true", help="skip protobuf encode/decode")
    r.add_argument("--plot", default=None, help="write a debug PNG here")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("suite", help="randomised scenario suite -> runs.jsonl + report.md")
    s.add_argument("--models", default="baseline")
    s.add_argument("--profiles", default="target,current")
    s.add_argument("--scenarios", default="all", help="all, group names or IDs, comma separated")
    s.add_argument("--runs", type=int, default=20)
    s.add_argument("--seed-base", type=int, default=0, help="use 1000000 for held-out TEST seeds")
    s.add_argument("--obs", type=float, default=120.0)
    s.add_argument("--dead-slot", default=None, help="e.g. 0 or 0,3: run with that anchor dead")
    s.add_argument("--fast", action="store_true")
    s.add_argument("--jobs", type=int, default=jobs)
    s.add_argument("--out", default=None)
    s.set_defaults(fn=cmd_suite)

    sh = sub.add_parser("shifts", help="fall-free day-in-the-life shifts -> false alarms per 8 h")
    sh.add_argument("--models", default="baseline")
    sh.add_argument("--profiles", default="target")
    sh.add_argument("--count", type=int, default=10)
    sh.add_argument("--hours", type=float, default=8.0)
    sh.add_argument("--seed-base", type=int, default=500_000)
    sh.add_argument("--protobuf", action="store_true", help="also run the protobuf decoder (slower)")
    sh.add_argument("--jobs", type=int, default=jobs)
    sh.add_argument("--out", default=None)
    sh.set_defaults(fn=cmd_shifts)

    rp = sub.add_parser("report", help="rebuild report.md from one or more output folders")
    rp.add_argument("dirs", nargs="+")
    rp.add_argument("--out", default=None)
    rp.set_defaults(fn=cmd_report)

    t = sub.add_parser("tune", help="random-search baseline thresholds on DEV seeds")
    t.add_argument("--trials", type=int, default=40)
    t.add_argument("--seeds", type=int, default=6, help="dev seeds per scenario")
    t.add_argument("--profile", default="target")
    t.add_argument("--obs", type=float, default=90.0)
    t.add_argument("--shift-hours", type=float, default=2.0)
    t.add_argument("--jobs", type=int, default=jobs)
    t.add_argument("--out", default=None)
    t.set_defaults(fn=cmd_tune)

    e = sub.add_parser("export", help="labelled synthetic dataset (.npz per run) for model developers")
    e.add_argument("--out", required=True)
    e.add_argument("--scenarios", default="all")
    e.add_argument("--runs", type=int, default=5)
    e.add_argument("--seed-base", type=int, default=2_000_000)
    e.add_argument("--profile", default="target")
    e.add_argument("--obs", type=float, default=60.0)
    e.add_argument("--jobs", type=int, default=jobs)
    e.set_defaults(fn=cmd_export)

    c = sub.add_parser("calibrate", help="fit soft-tissue coupling to Erciyes waist impact peaks")
    c.add_argument("--runs", type=int, default=60)
    c.set_defaults(fn=cmd_calibrate)

    po = sub.add_parser("posture", help="can the strip detect bad spine posture? (truth vs estimated angles)")
    po.add_argument("--scenarios", default="all")
    po.add_argument("--runs", type=int, default=10)
    po.add_argument("--seed-base", type=int, default=0)
    po.add_argument("--profiles", default="target,current_100hz,current")
    po.add_argument("--jobs", type=int, default=jobs)
    po.add_argument("--out", default=None)
    po.set_defaults(fn=cmd_posture)

    pl = sub.add_parser("plot", help="spine snapshots + signals for one run")
    pl.add_argument("scenario", choices=list(CATALOGUE))
    pl.add_argument("--seed", type=int, default=0)
    pl.add_argument("--obs", type=float, default=60.0)
    pl.add_argument("--out", required=True)
    pl.set_defaults(fn=cmd_plot)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
