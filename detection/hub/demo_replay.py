# File: demo_replay.py
# Purpose: Precompute one ThunderSim scenario as a replayable device stream for the spine demo:
#          raw 4-IMU samples as the radio delivered them (what the hub consumes) and the matching
#          segment orientations as quaternions (what the spine view draws). No backend or hub imports.
# Contributors: Unassigned
"""A Replay is the simulated wearer's strip, sample by sample, in device time.

* raw:  time_ms (N,), acc (N,4,3) m/s^2, gyro (N,4,3) rad/s, in the SEGMENT frame (+X right,
        +Y up, +Z posterior) as the device profile `fw_main` (100 Hz, +-8 g, +-1000 dps, 10 Hz DLPF)
        would measure them. Samples the BLE link lost are removed, so gaps are real.
        A dead sensor reads an all-zero record, as on the raw path.
* quat: (N,4,4) w,x,y,z per sensor: q0^-1 * q(t) of the simulated segment, i.e. the rotation away
        from the calibration stand expressed in the wearer frame. Rotating about +X by a negative
        angle is a forward bend (same convention as backend/app/simulated.py).

The quaternions are ground truth segment orientations, not a fusion of the noisy raw data; the
spine view therefore shows what the wearer really does, and the hub reacts to what the sensors
would feel. Needs detection/simulation on sys.path (thundersim) and numpy/scipy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PROFILE = "fw_main"
GRAVITY = 9.80665


@dataclass
class Replay:
    scenario: str
    seed: int
    description: str
    expected: str
    duration_s: float
    boot_s: float
    t: np.ndarray                  # (N,) true simulation seconds of each delivered sample
    time_ms: np.ndarray            # (N,) device uptime ms
    acc: np.ndarray                # (N,4,3)
    gyro: np.ndarray               # (N,4,3)
    quat: np.ndarray               # (N,4,4) w,x,y,z
    truth: list = field(default_factory=list)   # [{"t": s, "name": str}]
    link: dict = field(default_factory=dict)


def ensure_profile():
    """The firmware profile of origin/main (ImuManager::begin); the stock CLI does not have it."""
    from thundersim.device import PROFILES, DeviceProfile

    if PROFILE not in PROFILES:
        PROFILES[PROFILE] = DeviceProfile(PROFILE, 100.0, 8, 1000, 10,
                                          "ImuManager::begin: 10 ms period, +-8 g, +-1000 dps, ~10 Hz DLPF")
    return PROFILES[PROFILE]


def scenario_catalogue() -> list[dict]:
    from thundersim.scenarios import CATALOGUE

    return [{"id": key, "group": spec.group, "description": spec.description,
             "expected": getattr(spec, "expected", None)} for key, spec in CATALOGUE.items()]


class _NullModel:
    name = "none"

    def reset(self, context):
        pass

    def update(self, batch):
        return []


def build_replay(scenario: str, seed: int = 1, obs_s: float = 60.0, dead_slot: int | None = None) -> Replay:
    from thundersim.calibrate import load_fn_scale
    from thundersim.runner import Pipeline, StreamOptions
    from thundersim.scenarios import build_run
    from thundersim.spine import random_mounts, strip_kinematics

    ensure_profile()
    run = build_run(scenario, seed, obs_s=obs_s, fn_scale=load_fn_scale(), slot_failure=dead_slot)
    track = run.track
    fs = track.fs
    mounts = random_mounts(run.worker, np.random.default_rng(seed + 3))   # same draw as thundersim run
    kin = strip_kinematics(track, run.worker, mounts)

    class Capture(Pipeline):
        """Pipeline that keeps every sample the radio delivered, with its device time."""

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.cap = []

        def _deliver(self, t_true, time_ms, acc, gyro, modes_t, modes_v):
            a, g = acc.copy(), gyro.copy()
            for slot, t0, t1 in self.slot_failures:
                if t0 <= t_true < t1:
                    a[slot] = 0.0
                    g[slot] = 0.0
            before = len(self.arrivals_at)
            super()._deliver(t_true, time_ms, acc, gyro, modes_t, modes_v)
            if len(self.arrivals_at) > before:
                self.cap.append((float(t_true), int(time_ms), a, g))

    pipe = Capture(_NullModel(), run.worker, run.site, mounts, run.soft_fn, run.soft_zeta,
                   StreamOptions(profile=PROFILE, fast=True), seed, run.slot_failures)
    pipe.feed(track, run.work_modes, run.site.ambient_c, len(track) / fs)
    out = pipe.finish()
    if not pipe.cap:
        raise RuntimeError(f"{scenario}: the link delivered no samples")

    t = np.array([c[0] for c in pipe.cap])
    time_ms = np.array([c[1] for c in pipe.cap], dtype=np.int64)
    acc = np.stack([c[2] for c in pipe.cap]).astype(np.float64)
    gyro = np.stack([c[3] for c in pipe.cap]).astype(np.float64)

    idx = np.clip(np.round(t * fs).astype(int), 0, len(track) - 1)
    quat = np.empty((len(t), 4, 4))
    for i in range(4):
        rel = kin.seg_rot[i][0].inv() * kin.seg_rot[i][idx]
        x, y, z, w = rel.as_quat().T
        sign = np.where(w < 0, -1.0, 1.0)          # keep w >= 0: shortest representation
        quat[:, i] = np.stack([w * sign, x * sign, y * sign, z * sign], axis=1)

    truth = [{"t": round(float(tt), 3), "name": name} for tt, name, _ in run.events]
    return Replay(scenario, seed, run.spec.description if hasattr(run, "spec") else "", run.expected,
                  len(track) / fs, pipe.boot_s, t, time_ms, acc, gyro, quat, truth, out.link)
