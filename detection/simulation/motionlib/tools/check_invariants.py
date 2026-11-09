# File: check_invariants.py
# Purpose: Gate for every viewer mocap change: the simulated ground truth in the viewer payload
#          must be byte-identical with mocap on and off on the regression set.
# Contributors: Unassigned
"""Run from detection/simulation:

    .venv/bin/python motionlib/tools/check_invariants.py            # full regression set, seeds 0-2
    .venv/bin/python motionlib/tools/check_invariants.py --quick    # seed 0 only
    .venv/bin/python motionlib/tools/check_invariants.py --jobs 6 --physics 0
    .venv/bin/python motionlib/tools/check_invariants.py --switch contact   # contact pass off vs on

Each (scenario, seed, mocap) payload is built in its own subprocess, so the mocap switch
(THUNDERSIM_MOCAP=0/1, or a `mocap=` argument of run_payload once it exists) cannot leak between
runs through module state. --switch contact compares the contact pass (viz/contact.py) off and on
the same way (THUNDERSIM_CONTACT=0/1 and run_payload(contact=...)), with mocap off. The child prints one sha256 per checked field; the parent compares
them. Checked fields: frames.spine, frames.segq, frames.anchors, frames.chipq, frames.phase,
frames.worn and every signals.* field. Exit code 0 only if every pair matches.
Before stage S6 adds thundersim/viz/mocap.py the switch does nothing, so this checks determinism.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parents[2]          # detection/simulation
REGRESSION = ("A-HAMMER", "A-CARRY", "A-LADDER", "A-KNEEL", "A-REBAR", "A-RUN", "L-TRIP", "H-SCAF", "ARR")
SEEDS = (0, 1, 2)
FRAME_FIELDS = ("spine", "segq", "anchors", "chipq", "phase", "worn")


def field_digests(scenario: str, seed: int, on: bool, physics: bool, switch: str = "mocap") -> dict[str, str]:
    """Child side: build one payload (the switch `mocap` or `contact` set to `on`) and hash the
    invariant fields."""
    sys.path.insert(0, str(SIM_DIR))
    mocap = on if switch == "mocap" else False
    os.environ["THUNDERSIM_MOCAP"] = "1" if mocap else "0"
    if switch == "contact":
        os.environ["THUNDERSIM_CONTACT"] = "1" if on else "0"
    from thundersim.viz.export import run_payload

    kwargs = {"model": "none", "physics": physics}
    params = inspect.signature(run_payload).parameters
    if "mocap" in params:
        kwargs["mocap"] = mocap
    if switch == "contact" and "contact" in params:
        kwargs["contact"] = on
    payload = run_payload(scenario, seed, **kwargs)
    digests = {}
    for name in FRAME_FIELDS:
        digests[f"frames.{name}"] = sha(payload["frames"][name])
    for name, value in sorted(payload["signals"].items()):
        digests[f"signals.{name}"] = sha(value)
    return digests


def sha(value) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def run_child(scenario: str, seed: int, on: bool, physics: bool, switch: str = "mocap") -> dict[str, str]:
    command = [sys.executable, str(Path(__file__).resolve()), "--child", scenario, str(seed), str(int(on)),
               str(int(physics)), "--switch", switch]
    done = subprocess.run(command, cwd=SIM_DIR, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"{scenario} seed {seed} {switch}={int(on)} failed:\n{done.stderr[-2000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def check_pair(scenario: str, seed: int, physics: bool, switch: str = "mocap") -> list[str]:
    """Problems found for one (scenario, seed); empty when the invariants hold."""
    off = run_child(scenario, seed, False, physics, switch)
    on = run_child(scenario, seed, True, physics, switch)
    problems = [f"{scenario} seed {seed}: {name} differs" for name in sorted(off) if on.get(name) != off[name]]
    problems += [f"{scenario} seed {seed}: {name} only with {switch} on" for name in sorted(set(on) - set(off))]
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Mocap on/off invariants of the viewer payload.")
    parser.add_argument("--child", nargs=4, metavar=("SCENARIO", "SEED", "MOCAP", "PHYSICS"), help=argparse.SUPPRESS)
    parser.add_argument("--quick", action="store_true", help="seed 0 only")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--physics", type=int, default=1, choices=(0, 1))
    parser.add_argument("--scenarios", default=",".join(REGRESSION))
    parser.add_argument("--switch", default="mocap", choices=("mocap", "contact"), help="which pass to turn off and on")
    args = parser.parse_args()
    if args.child:
        scenario, seed, on, physics = args.child
        print(json.dumps(field_digests(scenario, int(seed), on == "1", physics == "1", args.switch)))
        return

    mocap_present = (SIM_DIR / "thundersim" / "viz" / "mocap.py").exists()
    seeds = (0,) if args.quick else SEEDS
    jobs = [(s, seed) for s in args.scenarios.split(",") for seed in seeds]
    print(f"check_invariants: {len(jobs)} runs x {args.switch} off/on, physics={args.physics}, "
          f"mocap module {'present' if mocap_present else 'absent (determinism check only)'}")
    problems: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {pool.submit(check_pair, s, seed, bool(args.physics), args.switch): (s, seed) for s, seed in jobs}
        for future, (s, seed) in futures.items():
            try:
                found = future.result()
            except Exception as exc:  # a crash is a failure, not a pass
                found = [f"{s} seed {seed}: ERROR {exc}"]
            print(f"  {s:9s} seed {seed}: {'PASS' if not found else 'FAIL'}")
            problems += found
    for line in problems:
        print("  -", line)
    print("check_invariants:", "PASS" if not problems else f"FAIL ({len(problems)} problem(s))")
    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()
