# File: harness.py
# Purpose: Scratch evaluation harness for the hub (DETECTION_ARCHITECTURE.md 7). Runs several
#          arms on IDENTICAL ThunderSim streams, adds the fw_main device profile in memory,
#          robustness arms (jerks, apnea, damping, curl, gaps at the event, restart, slow start,
#          50/20 Hz, dismiss-then-collapse) and long shifts. The simulator itself is not modified.
# Contributors: Unassigned
"""Run from detection/ (outputs always go to --out, a scratch directory):

  PYTHONPATH=.:simulation simulation/.venv/bin/python -m hub.eval.harness scenarios \
      --scenarios SYNC,SYNC-KNEEL --runs 20 --seed-base 0 --variant base --out $TMPDIR/hub/run1
  ... -m hub.eval.harness shifts --count 15 --hours 8 --seed-base 500000 --out $TMPDIR/hub/shifts
  ... -m hub.eval.summarize $TMPDIR/hub/run1

Variants: base | jerks | apnea | damped | curl | gap0.3 | gap0.7 | gap1.5 | restart | slowstart |
          rate50 | rate20 | dismiss (scenario X-DISMISS only)
Seeds: dev = 0.. (scenarios) and 500000.. (shifts). 1000000-1999999 is refused (burned block).
Blocks >= 2000000 run only if the block number is already in hub/SEED_LEDGER.md.
"""
import argparse
import dataclasses
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
HUB = Path(__file__).resolve().parents[1]
DETECTION = HUB.parent
for p in (str(DETECTION), str(DETECTION / "simulation")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from thundersim import falls  # noqa: E402
from thundersim.device import DeviceProfile, PROFILES  # noqa: E402
from thundersim.model_api import ALERT_KINDS  # noqa: E402
from thundersim.motions import Builder  # noqa: E402
from thundersim.runner import Pipeline, StreamOptions  # noqa: E402
from thundersim.scenarios import (CALIBRATION_S, CATALOGUE, begin, build_run, lead_in, scenario,  # noqa: E402
                                  shift_blocks, soft_tissue)
from thundersim.spine import random_mounts  # noqa: E402

# fw_main: ImuManager::begin at f3337e3 (+-8 g, +-1000 dps, ~10 Hz DLPF, 10 ms period).
if "fw_main" not in PROFILES:
    PROFILES["fw_main"] = DeviceProfile("fw_main", 100.0, 8, 1000, 10,
                                        "Firmware main f3337e3: 10 ms period, +/-8 g, +/-1000 dps, DLPF ~10 Hz")

ARMS = {
    "A": "hub.sim_models:FalldetShipped",
    "B": "hub.sim_models:FalldetCurrent",
    "C": "hub.sim_models:HubNoT4",
    "D": "hub.sim_models:HubFull",
    "D1": "hub.sim_models:HubPassThrough",
    "E": "baseline",
    "L": "hub.sim_models:HubLogreg",
}


def make_arm(key, spec):
    """'D' style keys load ARMS specs; 'D:section.key=value[;...]' builds a hub with config overrides
    (threshold sweeps on identical streams)."""
    from thundersim.model_api import load_model
    if spec is not None:
        return load_model(spec)
    from hub.main_model import MainModel
    overrides = {}
    for item in key.split(":", 1)[1].split(";"):
        path, value = item.split("=")
        section, name = path.split(".")
        overrides.setdefault(section, {})[name] = json.loads(value)
    return MainModel(overrides=overrides, name=key)


# ------------------------------------------------------------------ extra harness scenario
if "X-DISMISS" not in CATALOGUE:
    @scenario("X-DISMISS", "collapse", "fall_confirmed",
              "Harness: trip and get up (alert dismissed), then a collapse without impact", report_only=True)
    def _x_dismiss(b, rng, ctx):
        begin(b)
        lead_in(b, rng, keep_moving=True)
        falls.same_level_fall(b, "forward")
        b.hold(float(rng.uniform(3, 8)), sway=0.0)
        b.get_up("recovery")
        b.walk(float(rng.uniform(20, 40)), phase="recovery")
        b.stand(float(rng.uniform(10, 20)))
        falls.slow_collapse(b, "standing")
        ctx["ref_time"] = [t for t, n, _ in b.events if n == "impact"][-1]
        ctx["params"]["collapse_onset"] = round([t for t, n, _ in b.events if n == "fall_onset"][-1], 3)
        b.conscious = False
        b.hold(ctx["obs_s"], "lying", sway=0.0, fidget=0.0)


# ------------------------------------------------------------------ body variants (builder)
def builder_for(variant, seed):
    if variant not in ("jerks", "apnea", "damped", "curl"):
        return Builder
    vrng = np.random.default_rng(seed + 424242)   # own generator: the scenario's draws are unchanged

    class VariantBuilder(Builder):
        _done = False

        def hold(self, duration, phase="activity", sway=1.0, fidget=None):
            if self._done or phase != "lying" or self.conscious:
                return super().hold(duration, phase, sway, fidget)
            self._done = True
            if variant in ("apnea", "damped"):
                keep = self.breath_amp
                self.breath_amp = 0.0 if variant == "apnea" else keep * 0.2
                s = super().hold(duration, phase, sway, fidget)
                self.breath_amp = keep
                return s
            if variant == "curl":
                self.move(1.5, phase, voluntary=False, flex=self.pose.flex + np.deg2rad([25.0, 35.0, 45.0]))
                return super().hold(max(duration - 1.5, 1.0), phase, sway, fidget)
            # jerks: five 0.2 s jerks (10-20 deg) in the first 30 s
            times = np.sort(vrng.uniform(1.0, 29.0, 5))
            done = 0.0
            for tj in times:
                if tj > done + 0.05:
                    super().hold(tj - done, phase, sway, fidget)
                seg = self.seg(0.2, phase, fidget=0.0)
                shape = falls.bump(seg.k)
                seg.flex = seg.flex + np.outer(shape, np.deg2rad(vrng.uniform(10, 20, 3) * vrng.choice([-1, 1], 3)))
                seg.lat = seg.lat + np.outer(shape, np.deg2rad(vrng.uniform(-10, 10, 3)))
                self.commit(seg)
                done = tj + 0.2
            return super().hold(max(duration - done, 1.0), phase, sway, fidget)

    return VariantBuilder


# ------------------------------------------------------------------ multi-arm + stream perturbation
class Multi:
    """Feeds every arm the same batches; records events (original device time) per arm."""
    name = "multi"

    def __init__(self, arms, dismiss=False):
        self.models = {k: make_arm(k, spec) for k, spec in arms.items()}
        self.dismiss = dismiss
        self.shift = 0.0

    def reset(self, context):
        self.events = {k: [] for k in self.models}
        self.cpu = {k: 0.0 for k in self.models}
        self.pending_dismiss = {k: [] for k in self.models}
        self.dismissed = {k: 0 for k in self.models}
        for m in self.models.values():
            m.reset(context)

    def update(self, batch):
        end = float(batch.t[-1])
        for k, m in self.models.items():
            if self.dismiss and hasattr(m, "dismiss"):
                due = [d for d in self.pending_dismiss[k] if d <= float(batch.t[0])]
                if due:
                    self.pending_dismiss[k] = [d for d in self.pending_dismiss[k] if d > float(batch.t[0])]
                    self.dismissed[k] += int(bool(m.dismiss()))
            c0 = time.process_time()
            evs = m.update(batch) or []
            self.cpu[k] += time.process_time() - c0
            for e in evs:
                if e.t > end + 1e-6:
                    raise ValueError(f"{k}: event after its batch")
                self.events[k].append(self._rec(e, e.t + self.shift))
                if self.dismiss and e.kind == "fall_confirmed" and not self.dismissed[k] and not self.pending_dismiss[k]:
                    self.pending_dismiss[k].append(e.t + 10.0)      # supervisor dismisses after 10 s
        return []

    def finish(self):
        for k, m in self.models.items():
            if hasattr(m, "finish"):
                for e in m.finish() or []:
                    self.events[k].append(self._rec(e, e.t + self.shift))
        return []

    @staticmethod
    def _rec(e, t):
        d = e.detail or {}
        return {"t_dev": float(t), "kind": e.kind, "confidence": round(float(e.confidence), 3),
                "path": d.get("path"), "rule": d.get("rule"), "episode": d.get("episode_id"),
                "tags": d.get("tags"), "trigger_t": d.get("trigger_t"), "evidence": d.get("evidence")}


class Perturb:
    """Stream perturbations in front of Multi: gaps at the event, restart, slow start, decimation."""
    name = "perturb"

    def __init__(self, inner, variant):
        self.inner, self.variant = inner, variant
        self.ref_dev = None
        self.first_t = None
        self.count = 0
        self.restarted = False
        self.shift = 0.0

    def reset(self, context):
        self.inner.reset(context)

    def update(self, batch):
        t = np.asarray(batch.t, float)
        if self.first_t is None:
            self.first_t = float(t[0])
        keep = np.ones(len(t), bool)
        v = self.variant
        if v.startswith("gap") and self.ref_dev is not None:
            g = float(v[3:])
            keep &= ~((t >= self.ref_dev - g / 2) & (t < self.ref_dev + g / 2))
        elif v == "slowstart":
            idx = self.count + np.arange(len(t))
            keep &= ~((t < self.first_t + 1.5) & (idx % 5 != 0))
        elif v in ("rate50", "rate20"):
            k = 2 if v == "rate50" else 5
            keep &= (self.count + np.arange(len(t))) % k == 0
        self.count += len(t)
        if v == "restart" and self.ref_dev is not None:
            boot = self.ref_dev + 3.0          # reboot 3 s after the event; it takes 0.5 s
            pre = keep & (t < boot)
            post = keep & (t >= boot + 0.5)
            out = []
            for part, shift in ((pre, 0.0), (post, boot)):
                if part.any():
                    self.inner.shift = shift
                    out += self.inner.update(self._sub(batch, part, t[part] - shift))
            return out
        if not keep.any():
            return []
        self.inner.shift = 0.0
        return self.inner.update(self._sub(batch, keep, t[keep]))

    @staticmethod
    def _sub(batch, mask, t):
        return dataclasses.replace(batch, t=t, acc=batch.acc[mask], gyro=batch.gyro[mask],
                                   slot_mask=batch.slot_mask[mask], sat_mask=batch.sat_mask[mask])

    def finish(self):
        return self.inner.finish()


# ------------------------------------------------------------------ one scenario run
def run_one(task):
    scenario_id, seed, variant, arms, profile, errors, fast, dead_slot, obs_s = task
    try:
        from thundersim.calibrate import load_fn_scale
        run = build_run(scenario_id, seed, obs_s=obs_s, fn_scale=load_fn_scale(), slot_failure=dead_slot,
                        builder=builder_for(variant, seed))
        multi = Multi({k: ARMS.get(k) for k in arms}, dismiss=(variant == "dismiss"))
        model = Perturb(multi, variant)
        mounts = random_mounts(run.worker, np.random.default_rng(seed + 3))
        options = StreamOptions(profile=profile, fast=fast, errors=errors)
        pipe = Pipeline(model, run.worker, run.site, mounts, run.soft_fn, run.soft_zeta, options, seed, run.slot_failures)
        if run.ref_time is not None:
            model.ref_dev = run.ref_time + pipe.boot_s
        total = len(run.track) / run.track.fs
        pipe.feed(run.track, run.work_modes, run.site.ambient_c, total)
        pipe.finish()
        dev = np.array(pipe.arrivals_dev, float) / 1000.0
        at = np.array(pipe.arrivals_at, float)
        out = {}
        for k, evs in multi.events.items():
            for e in evs:
                e["t"] = round(e["t_dev"] - pipe.boot_s, 3)
                e["t_laptop"] = round(float(np.interp(e["t_dev"], dev, at)) if len(dev) else e["t"], 3)
            out[k] = evs
        status = {k: m.status() for k, m in multi.models.items() if hasattr(m, "status")}
        return {"scenario": scenario_id, "seed": seed, "variant": variant, "profile": profile, "errors": errors,
                "fast": fast, "dead_slot": dead_slot, "expected": run.expected, "ref_time": run.ref_time,
                "checkin_s": run.checkin_s, "params": run.params, "duration_s": round(total, 1),
                "truth": [(round(t, 3), n) for t, n, _ in run.events], "arms": out, "status": status,
                "cpu_s": {k: round(v, 3) for k, v in multi.cpu.items()}, "link": pipe.link.stats.summary()}
    except Exception as error:  # keep the batch running; report the failure
        import traceback
        return {"scenario": scenario_id, "seed": seed, "variant": variant, "error": repr(error),
                "trace": traceback.format_exc()[-2000:]}


def run_shift_one(task):
    seed, hours, arms, profile, errors, fast = task
    try:
        from thundersim.calibrate import load_fn_scale
        multi = Multi({k: ARMS.get(k) for k in arms})
        pipe = None
        timeline = []
        for activity, track, events, modes, worker, site in shift_blocks(seed, hours):
            if pipe is None:
                mounts = random_mounts(worker, np.random.default_rng(seed + 3))
                fn, zeta = soft_tissue(np.random.default_rng(seed + 4), site.clothing, load_fn_scale())
                pipe = Pipeline(multi, worker, site, mounts, fn, zeta, StreamOptions(profile=profile, fast=fast, errors=errors), seed)
            start = pipe.t_offset
            pipe.feed(track, modes, site.ambient_c, hours * 3600)
            timeline.append((start, pipe.t_offset, activity))
        pipe.finish()
        out = {}
        for k, evs in multi.events.items():
            for e in evs:
                e["t"] = round(e["t_dev"] - pipe.boot_s, 3)
                e["activity"] = next((a for s0, s1, a in timeline if s0 <= e["t"] < s1), "?")
            out[k] = evs
        status = {k: m.status() for k, m in multi.models.items() if hasattr(m, "status")}
        return {"seed": seed, "hours": round(pipe.t_offset / 3600, 3), "profile": profile, "errors": errors,
                "arms": out, "status": status, "cpu_s": {k: round(v, 2) for k, v in multi.cpu.items()}}
    except Exception as error:
        import traceback
        return {"seed": seed, "error": repr(error), "trace": traceback.format_exc()[-2000:]}


# ------------------------------------------------------------------ seed discipline
def check_seed_block(base):
    if 1_000_000 <= base < 2_000_000:
        raise SystemExit("seed block 1000000-1999999 is burned; use dev seeds or a ledgered block >= 2000000")
    if base >= 2_000_000:
        ledger = (HUB / "SEED_LEDGER.md").read_text() if (HUB / "SEED_LEDGER.md").exists() else ""
        if f"| {base} |" not in ledger:
            raise SystemExit(f"held-out block {base} is not in hub/SEED_LEDGER.md; add the row before running")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hub.eval.harness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scenarios")
    s.add_argument("--scenarios", required=True)
    s.add_argument("--runs", type=int, default=20)
    s.add_argument("--seed-base", type=int, default=0)
    s.add_argument("--variant", default="base")
    s.add_argument("--arms", default="A,B,C,D,D1,E,L")
    s.add_argument("--profile", default="fw_main")
    s.add_argument("--errors", default="calibrated")
    s.add_argument("--protobuf", action="store_true", help="run the protobuf decoder (default: --fast path)")
    s.add_argument("--dead-slot", type=int, default=None)
    s.add_argument("--obs", type=float, default=120.0)
    s.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    s.add_argument("--out", required=True)
    h = sub.add_parser("shifts")
    h.add_argument("--count", type=int, default=15)
    h.add_argument("--hours", type=float, default=8.0)
    h.add_argument("--seed-base", type=int, default=500_000)
    h.add_argument("--arms", default="A,B,C,D,E")
    h.add_argument("--profile", default="fw_main")
    h.add_argument("--errors", default="calibrated")
    h.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    h.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    check_seed_block(args.seed_base)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = args.arms.split(",") if "|" not in args.arms else args.arms.split("|")
    started = time.time()
    if args.cmd == "scenarios":
        tasks = [(sid, args.seed_base + i, args.variant, arms, args.profile, args.errors, not args.protobuf,
                  args.dead_slot, args.obs) for sid in args.scenarios.split(",") for i in range(args.runs)]
        fn, name = run_one, "runs.jsonl"
    else:
        tasks = [(args.seed_base + i, args.hours, arms, args.profile, args.errors, True) for i in range(args.count)]
        fn, name = run_shift_one, "shifts.jsonl"
    meta = {"argv": sys.argv[1:] if argv is None else argv, "cmd": args.cmd, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / f"meta_{name.split('.')[0]}.json").write_text(json.dumps(meta, indent=1))
    with open(out / name, "a") as handle, ProcessPoolExecutor(args.jobs) as pool:
        for i, result in enumerate(pool.map(fn, tasks, chunksize=1), 1):
            handle.write(json.dumps(result, default=str) + "\n")
            handle.flush()
            if "error" in result:
                print("ERROR", result.get("scenario"), result.get("seed"), result["error"], result.get("trace", "")[-600:])
            if i % 20 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
