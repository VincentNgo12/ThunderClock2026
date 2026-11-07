# File: runner.py
# Purpose: Drive a scenario or shift through the full chain (body -> soft tissue -> MPU-6050 ->
#          protobuf -> BLE -> backend decoder -> canonicalize -> model -> check-in) and score it.
# Contributors: Unassigned

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .canonical import Canonicalizer
from .device import PROFILES, DeviceErrors, ImuStrip
from .model_api import ALERT_KINDS, FallModel, StreamContext, load_model
from .scenarios import CALIBRATION_S, ScenarioRun, build_run, random_worker, shift_blocks, soft_tissue
from .site import device_temperature
from .spine import apply_soft_tissue, random_mounts, strip_kinematics, virtual_imu
from .transport import BleLink, RawCodec

CHECKIN_WINDOW_S = 15.0   # worker has this long to cancel before the supervisor is alerted [design]
DEVICE_CHUNK_S = 10.0


@dataclass
class StreamOptions:
    profile: str = "target"
    fast: bool = False              # skip protobuf encode/decode (values still rounded to float32)
    radio: str | None = None        # override the scenario's radio environment
    errors: str = "calibrated"      # calibrated | uncalibrated
    batch_s: float = 0.5


@dataclass
class StreamOutput:
    events: list = field(default_factory=list)      # dicts with sim-time t and laptop arrival time
    frames: int = 0
    saturated_frames: int = 0
    link: dict = field(default_factory=dict)
    duration_s: float = 0.0


class Pipeline:
    """One worker's strip, radio link, decoder, canonicalizer and model, fed block by block."""

    def __init__(self, model: FallModel, worker, site, mounts, soft_fn, soft_zeta, options: StreamOptions,
                 seed: int, slot_failures=()):
        self.model, self.worker, self.site, self.options = model, worker, site, options
        self.mounts, self.soft_fn, self.soft_zeta = mounts, soft_fn, soft_zeta
        self.profile = PROFILES[options.profile]
        errors = DeviceErrors() if options.errors == "calibrated" else DeviceErrors.uncalibrated()
        rng = np.random.default_rng(seed + 31)
        errors = DeviceErrors(errors.accel_bias_g, errors.accel_scale, errors.gyro_bias_dps, errors.gyro_tempco_dps_per_c,
                              float(rng.uniform(0.0008, 0.0015)), errors.timing_jitter_s)  # SIM_SPEC: 0.8-1.5 ms per read
        self.strip = ImuStrip(self.profile, 4, errors, seed=seed + 1)
        self.link = BleLink(options.radio or site.radio, seed=seed + 2)
        self.codec = None if options.fast else RawCodec()
        self.canon = Canonicalizer(self.profile.odr_hz, self.profile.accel_range_g, self.profile.gyro_range_dps,
                                   CALIBRATION_S, options.batch_s)
        self.slot_failures = list(slot_failures)
        self.boot_s = self.strip.boot_offset_ms / 1000.0
        self.arrivals_dev, self.arrivals_at = [], []
        self.out = StreamOutput()
        model.reset(StreamContext(self.profile.odr_hz, self.profile.name, self.profile.accel_range_g,
                                  self.profile.gyro_range_dps, seed))
        self.t_offset = 0.0
        self.total_s = None

    def feed(self, track, work_modes, ambient_span: tuple[float, float], total_s: float) -> None:
        """Push one PoseTrack (starting at the current stream time) through the chain."""
        kin = strip_kinematics(track, self.worker, self.mounts)
        acc, gyro = virtual_imu(kin, track.fs)
        acc, gyro = apply_soft_tissue(acc, gyro, track.fs, self.soft_fn, self.soft_zeta)
        n = len(track)
        chunk = int(DEVICE_CHUNK_S * track.fs)
        modes_t = np.array([t for t, _ in work_modes]) + self.t_offset
        modes_v = [m for _, m in work_modes]
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            t = self.t_offset + np.arange(start, stop) / track.fs
            frac = np.clip(t / max(total_s, 1e-6), 0, 1)
            ambient = ambient_span[0] + (ambient_span[1] - ambient_span[0]) * frac
            samples = self.strip.process(t, acc[start:stop], gyro[start:stop], device_temperature(ambient, self.site.sun))
            for k in range(len(samples.t_true)):
                self._deliver(samples.t_true[k], samples.time_ms[k], samples.acc[k], samples.gyro[k], modes_t, modes_v)
        self.t_offset += n / track.fs

    def _deliver(self, t_true, time_ms, acc, gyro, modes_t, modes_v) -> None:
        acc = acc.copy()
        gyro = gyro.copy()
        for slot, t0, t1 in self.slot_failures:
            if t0 <= t_true < t1:
                acc[slot] = 0.0  # a dead sensor reads all-zero records on the raw path
                gyro[slot] = 0.0
        if self.codec is not None:
            payload = self.codec.encode(time_ms, acc, gyro)
            nbytes = len(payload)
        else:
            payload, nbytes = None, 147
        arrival = self.link.send(float(t_true), nbytes)
        if arrival is None:
            return
        if self.codec is not None:
            try:
                time_ms, acc, gyro = self.codec.decode(payload)
            except Exception:  # decoder rejection is counted, never crashes the run
                self.link.stats.rejected += 1
                return
        else:
            acc, gyro = acc.astype(np.float32).astype(float), gyro.astype(np.float32).astype(float)
        idx = np.searchsorted(modes_t, t_true, side="right") - 1
        mode = modes_v[max(idx, 0)] if len(modes_v) else "normal"
        self.arrivals_dev.append(int(time_ms))
        self.arrivals_at.append(float(arrival))
        self.canon.push(int(time_ms), acc, gyro, mode)
        self._run_model(final=False)

    def _run_model(self, final: bool) -> None:
        for batch in self.canon.pop_batches(final=final):
            self.out.frames += len(batch)
            self.out.saturated_frames += int(batch.sat_mask.any(axis=1).sum())
            for ev in self.model.update(batch) or []:
                if ev.t > float(batch.t[-1]) + 1e-6:
                    raise ValueError(f"{self.model.name} returned an event after its batch (not causal)")
                self.out.events.append({"t_dev": float(ev.t), "kind": ev.kind, "confidence": float(ev.confidence),
                                        "detail": ev.detail})

    def finish(self) -> StreamOutput:
        self._run_model(final=True)
        last_t = self.canon._times[-1] if self.canon._times else 0.0
        for ev in (self.model.finish() if hasattr(self.model, "finish") else []) or []:
            # Optional hook: decide anything still pending at the end of the stream.
            self.out.events.append({"t_dev": min(float(ev.t), last_t), "kind": ev.kind, "confidence": float(ev.confidence),
                                    "detail": ev.detail})
        dev = np.array(self.arrivals_dev, float) / 1000.0
        at = np.array(self.arrivals_at, float)
        for ev in self.out.events:
            ev["t"] = ev["t_dev"] - self.boot_s  # simulation time of the frame that triggered it
            ev["t_laptop"] = float(np.interp(ev["t_dev"], dev, at)) if len(dev) else ev["t"]
        self.out.link = self.link.stats.summary()
        self.out.duration_s = self.t_offset
        return self.out


# --- scoring --------------------------------------------------------------------------------
def classify(events: list) -> str:
    kinds = {e["kind"] for e in events if e["t"] > CALIBRATION_S}
    if kinds & set(ALERT_KINDS):
        return "fall_confirmed"
    if "near_fall" in kinds:
        return "near_fall"
    return "normal"


def apply_checkin(events: list, checkin_s: float | None) -> tuple[list, int]:
    """Worker check-in: each alert prompts the worker; a fine worker cancels within checkin_s.

    Returns supervisor alert times (laptop clock) and the number of cancelled alerts. One
    cancel covers the rest of that episode (until a 'cleared' or 10 min).
    """
    supervisor, cancelled, snooze_until = [], 0, -1.0
    for e in sorted(events, key=lambda e: e["t_laptop"]):
        if e["kind"] == "cleared":
            snooze_until = -1.0
        if e["kind"] not in ALERT_KINDS or e["t"] <= CALIBRATION_S:
            continue
        if e["t_laptop"] < snooze_until:
            cancelled += 1
            continue
        if checkin_s is not None and checkin_s < CHECKIN_WINDOW_S:
            cancelled += 1
            snooze_until = e["t_laptop"] + 600.0
            continue
        supervisor.append(e["t_laptop"] + CHECKIN_WINDOW_S)
    return supervisor, cancelled


@dataclass
class RunResult:
    scenario: str
    seed: int
    profile: str
    model: str
    expected: str
    output: str
    passed: bool
    partial: bool
    report_only: bool
    ref_time: float | None
    latency_s: float | None
    supervisor_latency_s: float | None
    supervisor_alerts: int
    cancelled_alerts: int
    stage1_triggers: int
    params: dict
    events: list
    truth: list
    link: dict
    frames: int
    saturated_frames: int
    duration_s: float
    dead_slot: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def score(run: ScenarioRun, out: StreamOutput, profile: str, model_name: str, dead_slot: int | None) -> RunResult:
    output = classify(out.events)
    alerts = [e for e in out.events if e["kind"] in ALERT_KINDS and e["t"] > CALIBRATION_S]
    supervisor, cancelled = apply_checkin(out.events, run.checkin_s)
    latency = sup_latency = None
    if run.ref_time is not None and alerts:
        latency = round(min(e["t_laptop"] for e in alerts) - run.ref_time, 3)
    if run.ref_time is not None and supervisor:
        sup_latency = round(min(supervisor) - run.ref_time, 3)
    if run.expected == "fall_confirmed":
        passed = output == "fall_confirmed" and latency is not None and latency > -2.0
    else:
        passed = output == run.expected
    partial = run.scenario == "ARR" and output == "near_fall"  # SIM_SPEC: near_fall is a partial pass for ARR
    truth = [{"t": round(t, 3), "name": n, **{k: v for k, v in info.items() if isinstance(v, (int, float, str, bool))}}
             for t, n, info in run.events]
    return RunResult(run.scenario, run.seed, profile, model_name, run.expected, output, bool(passed), partial,
                     run.report_only, run.ref_time, latency, sup_latency, len(supervisor), cancelled,
                     sum(e["kind"] == "suspected_fall" for e in out.events), run.params,
                     [{k: v for k, v in e.items()} for e in out.events], truth, out.link, out.frames,
                     out.saturated_frames, round(out.duration_s, 2), dead_slot)


def run_scenario(scenario_id: str, seed: int, model_spec: "str | FallModel" = "baseline", options: StreamOptions | None = None,
                 obs_s: float = 120.0, dead_slot: int | None = None, fn_scale: float | None = None) -> RunResult:
    options = options or StreamOptions()
    if fn_scale is None:
        from .calibrate import load_fn_scale

        fn_scale = load_fn_scale()
    run = build_run(scenario_id, seed, obs_s=obs_s, fn_scale=fn_scale, slot_failure=dead_slot)
    model = load_model(model_spec) if isinstance(model_spec, str) else model_spec
    mounts = random_mounts(run.worker, np.random.default_rng(seed + 3))
    pipe = Pipeline(model, run.worker, run.site, mounts, run.soft_fn, run.soft_zeta, options, seed, run.slot_failures)
    total = len(run.track) / run.track.fs
    pipe.feed(run.track, run.work_modes, run.site.ambient_c, total)
    out = pipe.finish()
    return score(run, out, options.profile, getattr(model, "name", model_spec), dead_slot)


def run_scenario_task(args: tuple) -> dict:
    """Process-pool entry point."""
    scenario_id, seed, model_spec, options, obs_s, dead_slot, fn_scale = args
    try:
        return run_scenario(scenario_id, seed, model_spec, options, obs_s, dead_slot, fn_scale).to_dict()
    except Exception as error:  # keep the suite running; report the failure
        return {"scenario": scenario_id, "seed": seed, "profile": options.profile, "error": repr(error)}


def run_shift(seed: int, hours: float, model_spec: str = "baseline", options: StreamOptions | None = None) -> dict:
    """Fall-free day-in-the-life stream: false alarms and Stage-1 triggers per hour."""
    options = options or StreamOptions(fast=True)
    model = load_model(model_spec) if isinstance(model_spec, str) else model_spec
    pipe = None
    activities: dict[str, float] = {}
    timeline: list[tuple[float, float, str]] = []  # (start, end, activity) in simulation time
    from .calibrate import load_fn_scale

    for activity, track, events, modes, worker, site in shift_blocks(seed, hours):
        if pipe is None:
            mounts = random_mounts(worker, np.random.default_rng(seed + 3))
            fn, zeta = soft_tissue(np.random.default_rng(seed + 4), site.clothing, load_fn_scale())
            pipe = Pipeline(model, worker, site, mounts, fn, zeta, options, seed)
        start = pipe.t_offset
        pipe.feed(track, modes, site.ambient_c, hours * 3600)
        timeline.append((start, pipe.t_offset, activity))
        activities[activity] = activities.get(activity, 0.0) + len(track) / track.fs
    out = pipe.finish()
    alerts = [e for e in out.events if e["kind"] in ALERT_KINDS and e["t"] > CALIBRATION_S]
    supervisor, cancelled = apply_checkin(out.events, checkin_s=5.0)
    hrs = out.duration_s / 3600
    return {"seed": seed, "profile": options.profile, "model": getattr(model, "name", model_spec), "hours": round(hrs, 3),
            "false_alarms": len(alerts), "supervisor_alerts": len(supervisor), "cancelled": cancelled,
            "stage1_triggers": sum(e["kind"] == "suspected_fall" for e in out.events),
            "near_falls": sum(e["kind"] == "near_fall" for e in out.events),
            "alerts": [{"t": round(e["t"], 1), "kind": e["kind"], "detail": e["detail"],
                        "activity": next((a for s0, s1, a in timeline if s0 <= e["t"] < s1), "?")} for e in alerts],
            "activity_s": {k: round(v) for k, v in activities.items()}, "link": out.link}


def run_shift_task(args: tuple) -> dict:
    seed, hours, model_spec, options = args
    try:
        return run_shift(seed, hours, model_spec, options)
    except Exception as error:
        return {"seed": seed, "profile": options.profile, "error": repr(error)}


__all__ = ["StreamOptions", "Pipeline", "run_scenario", "run_shift", "classify", "apply_checkin", "random_worker"]
