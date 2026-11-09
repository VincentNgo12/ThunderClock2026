# File: realism/emulate.py
# Purpose: Generate simulator runs for the realism scorecard and emulate each REAL sensor on the
#          simulated body (placement, lever arm, soft tissue, bandwidth, rate, range, noise).
# Contributors: Unassigned
"""Sim side of the comparison. Only dev seeds are used (500000-999999).

Harmonisation, like for like (the real data is never altered):
- frame: the simulator's segment frame (+X right, +Y up, +Z posterior), which is what the
  real data's per-subject calibration (harmonize.calibrate) produces; mount error is therefore
  removed on both sides;
- placement: Erciyes waist <-> pelvis anchor (sacral skin, 4 cm behind the spine line);
  Erciyes chest (sternum) <-> upper-thoracic segment point 0.20 m anterior and 0.08 m below the
  upper-thoracic anchor (the inverse of harmonize.CHEST_TO_UT_M), so it sees the lever-arm terms
  a real sternum sensor sees; VTT hip <-> pelvis anchor, VTT back <-> upper-thoracic anchor;
- soft tissue: the simulator's own per-slot mount filter (pelvis slot for waist/hip, upper
  thoracic slot for chest/back);
- bandwidth and rate: 4th-order zero-phase low-pass at SensorSpec.lowpass_hz, then decimation
  to the dataset rate (Erciyes 25 Hz, VTT 100 Hz);
- range and noise: per-axis clipping at the dataset's full scale (VTT +/-2 g, +/-250 deg/s) and
  small white noise;
- conditions (Erciyes events only): falls onto a crash mat, dry, light clothing, and an
  Erciyes-like population (1.57-1.84 m, 47-92 kg, no tool belt) [verify against the dataset
  description]. Construction activities (VTT) keep the simulator's own site and worker draw.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace

import numpy as np
from scipy import signal
from scipy.spatial.transform import Rotation

from thundersim import scenarios as sc
from thundersim.motions import Builder
from thundersim.site import random_site
from thundersim.spine import GRAVITY_WORLD, Worker, apply_soft_tissue, strip_kinematics

from .paths import SPECS, SensorSpec

CHEST_TO_UT_M = np.array([0.0, 0.08, 0.20])  # harmonize.py: sternum sensor -> upper-thoracic anchor
SEED_BASE = 500_000                           # dev seeds only (500000-999999); never >= 1000000
OBS_S = 15.0                                  # post-event observation kept in event runs


def dev_seed(block: int, i: int) -> int:
    assert 0 <= i < 1000, "seed index must stay inside its 1000-seed block"
    seed = SEED_BASE + 1000 * block + i
    assert seed < 1_000_000, "held-out seeds are never used here"
    return seed


def erciyes_worker(rng: np.random.Generator, clothing: str) -> Worker:
    """Same number of draws as scenarios.random_worker, so the rest of the run is unchanged."""
    del clothing
    return Worker(height_m=float(rng.uniform(1.57, 1.84)), mass_kg=float(rng.uniform(47, 92)),
                  load_kg=float(rng.uniform(0.0, 0.0)), mount_error_deg=6.0)


@contextmanager
def lab_conditions():
    """Patch the scenario module's site and worker draws to Erciyes lab conditions."""
    orig_site, orig_worker = sc.random_site, sc.random_worker
    sc.random_site = lambda rng, surfaces=None: replace(orig_site(rng), surface="crash_mat", condition="dry",
                                                        clothing="light")
    sc.random_worker = erciyes_worker
    try:
        yield
    finally:
        sc.random_site, sc.random_worker = orig_site, orig_worker


# --- body signals -------------------------------------------------------------------------------
def _point_imu(pos: np.ndarray, rot: Rotation, fs: float) -> tuple[np.ndarray, np.ndarray]:
    dt = 1.0 / fs
    accel_world = np.gradient(np.gradient(pos, dt, axis=0), dt, axis=0)
    acc = rot.inv().apply(accel_world - GRAVITY_WORLD)
    gyro = np.zeros_like(acc)
    rel = rot[:-2].inv() * rot[2:]
    gyro[1:-1] = rel.as_rotvec() / (2 * dt)
    gyro[0], gyro[-1] = gyro[1], gyro[-2]
    return acc, gyro


def body_signals(track, worker, soft_fn, soft_zeta, dataset: str, four: bool = False):
    """Segment-frame specific force and rate at the real sensor sites, 1 kHz, after soft tissue.

    four=False: (N,2,3) [waist/hip, chest/back]; four=True: (N,4,3) the strip anchors."""
    kin = strip_kinematics(track, worker, [Rotation.identity()] * 4)
    if four:
        pts = [(kin.anchors[:, i], kin.seg_rot[i]) for i in range(4)]
        slots = [0, 1, 2, 3]
    elif dataset == "univr":  # one lower-back sensor (L1-L2) ~ the lumbar anchor, in both slots
        pts = [(kin.anchors[:, 1], kin.seg_rot[1])] * 2
        slots = [1, 1]
    else:
        upper = kin.anchors[:, 3]
        if dataset == "erciyes":
            upper = upper - kin.seg_rot[3].apply(CHEST_TO_UT_M)
        pts = [(kin.anchors[:, 0], kin.seg_rot[0]), (upper, kin.seg_rot[3])]
        slots = [0, 3]
    acc = np.empty((len(track), len(pts), 3))
    gyro = np.empty_like(acc)
    for k, (pos, rot) in enumerate(pts):
        acc[:, k], gyro[:, k] = _point_imu(pos, rot, track.fs)
    acc, gyro = apply_soft_tissue(acc, gyro, track.fs, np.asarray(soft_fn)[slots], np.asarray(soft_zeta)[slots])
    return acc, gyro


def sample_like(acc: np.ndarray, gyro: np.ndarray, fs: float, spec: SensorSpec, rng: np.random.Generator):
    """Band-limit, decimate, add noise and clip like the real sensor."""
    sos = signal.butter(4, spec.lowpass_hz, fs=fs, output="sos")
    step = int(round(fs / spec.fs))
    acc = signal.sosfiltfilt(sos, acc, axis=0)[::step]
    gyro = signal.sosfiltfilt(sos, gyro, axis=0)[::step]
    acc = acc + rng.normal(0, spec.acc_noise, acc.shape)
    gyro = gyro + rng.normal(0, spec.gyro_noise, gyro.shape)
    return np.clip(acc, -spec.acc_clip, spec.acc_clip), np.clip(gyro, -spec.gyro_clip, spec.gyro_clip)


def falldet_like_erciyes(track, worker, soft_fn, soft_zeta, rng) -> tuple[np.ndarray, np.ndarray]:
    """4-anchor 100 Hz stream with Erciyes bandwidth (25 Hz samples upsampled 4x), the way
    harmonize.py made falldet's Erciyes training data."""
    acc, gyro = body_signals(track, worker, soft_fn, soft_zeta, "erciyes", four=True)
    spec = SPECS["erciyes"]
    acc, gyro = sample_like(acc, gyro, track.fs, spec, rng)
    return signal.resample_poly(acc, 4, 1, axis=0), signal.resample_poly(gyro, 4, 1, axis=0)


# --- event runs --------------------------------------------------------------------------------
def _first_after(events, name, t0=0.0):
    for t, n, info in events:
        if n == name and t >= t0:
            return t, info
    return None, None


def event_run(cat, seed: int):
    """One simulated event for an EventCategory. Returns None if the run does not belong to the
    category (e.g. a forward blow for the side-fall category).

    Returns dict(track, worker, soft_fn, soft_zeta, window=(t0, t1), params)."""
    rng = np.random.default_rng(seed)
    if cat.onset_event in ("first_sit", "first_lie"):
        with lab_conditions():
            site = sc.random_site(np.random.default_rng(seed + 7_919))
            worker = sc.random_worker(rng, site.clothing)
        b = Builder(worker, site, rng)
        sc.begin(b)
        b.stand(float(rng.uniform(2, 4)))
        t0 = b.t
        if cat.onset_event == "first_sit":
            b.sit(float(rng.uniform(0.35, 0.5)), hard=True)  # A-SIT's sit
            b.hold(float(rng.uniform(6, 10)), sway=0.3)
            t1 = b.t
        else:
            b.lie("supine", float(rng.uniform(2.0, 4.0)))   # A-LIE's lie-down
            t1 = b.t + 0.5   # search the lie-down move only, not the lying activity after it
            b.lying_activity(float(rng.uniform(6, 10)))
        track = b.build()
        fn, zeta = sc.soft_tissue(rng, site.clothing)
        return dict(track=track, worker=worker, soft_fn=fn, soft_zeta=zeta, window=(t0, t1), params={})
    if cat.scenario == "H-LOW":  # low-platform fall (UNIVRFall 39/40), same physics as H-SCAF
        with lab_conditions():
            site = sc.random_site(np.random.default_rng(seed + 7_919))
            worker = sc.random_worker(rng, site.clothing)
        b = Builder(worker, site, rng)
        height = float(rng.uniform(0.4, 0.8))
        sc.begin(b, height)
        b.stand(float(rng.uniform(2, 4)))
        from thundersim import falls
        falls.height_fall(b, height, start="edge", surface=site.surface)
        falls.post_fall(b, "still_then_move", OBS_S)
        track = b.build()
        fn, zeta = sc.soft_tissue(rng, site.clothing)
        onset, _ = _first_after(b.events, "fall_onset")
        return dict(track=track, worker=worker, soft_fn=fn, soft_zeta=zeta, window=(onset - 0.5, onset + 3.0),
                    params={"height_m": height}, events=b.events)
    with lab_conditions():
        run = sc.build_run(cat.scenario, seed, obs_s=OBS_S)
    if cat.scenario == "STRUCK-KD":
        _, info = _first_after(run.events, "struck")
        if info is None or info.get("direction") == "forward":
            return None
    onset, _ = _first_after(run.events, cat.onset_event)
    if onset is None:
        return None
    return dict(track=run.track, worker=run.worker, soft_fn=run.soft_fn, soft_zeta=run.soft_zeta,
                window=(onset - 0.5, onset + 3.0), params=run.params, events=run.events)


# --- activity clips ----------------------------------------------------------------------------
def _act_body(b: Builder, rng: np.random.Generator, activity: str) -> None:
    if activity == "walk":
        b.walk(float(rng.uniform(25, 45)), turn_rate=float(rng.uniform(-0.1, 0.1)))
    elif activity == "walk_load":
        b.walk(float(rng.uniform(25, 45)), load_kg=20.0)
    elif activity == "run":
        b.run(float(rng.uniform(10, 20)), speed=float(rng.uniform(2.5, 3.5)))
    elif activity == "bend":
        for _ in range(3):
            b.stoop(float(rng.uniform(60, 90)), hold_s=float(rng.uniform(1, 3)))
            b.stand(float(rng.uniform(1, 3)))
    elif activity == "squat":
        for _ in range(3):
            b.squat(float(rng.uniform(1, 3)))
            b.stand(float(rng.uniform(1, 3)))
    elif activity == "pick":
        for _ in range(3):
            b.carry(float(rng.uniform(4, 8)), float(rng.uniform(2, 10)))
    elif activity == "kneel":
        b.kneel_work(float(rng.uniform(40, 60)))
    elif activity == "overhead":
        b.drill_overhead(float(rng.uniform(30, 50)))
    elif activity == "lie_work":
        b.lie("supine", 3.0)
        b.lying_activity(float(rng.uniform(40, 60)))
        b.get_up("activity")
    elif activity == "jump":
        for _ in range(3):
            b.ladder(float(rng.uniform(0.3, 0.6)), up=True)
            b.stand(float(rng.uniform(1, 3)))
            b.jump_down(b.ground_y)
            b.walk(float(rng.uniform(2, 4)))
    elif activity == "ladder":
        for _ in range(2):
            height = float(rng.uniform(1.5, 3.0))
            b.ladder(height, up=True)
            b.stand(float(rng.uniform(1, 2)))
            b.ladder(height, up=False)
    else:
        raise KeyError(activity)


def shift_run(seed: int, hours: float = 0.15):
    """Fall-free shift blocks (scenarios.shift_blocks, SIM_SPEC 5.2 mix) for the worksite comparison.
    Returns a list of dict(track, worker, soft_fn, soft_zeta, window) per block."""
    out = []
    rng = np.random.default_rng(seed + 31)
    for activity, track, _events, _modes, worker, site in sc.shift_blocks(seed, hours, block_s=(60, 180)):
        fn, zeta = sc.soft_tissue(rng, site.clothing)
        out.append(dict(track=track, worker=worker, soft_fn=fn, soft_zeta=zeta, window=(2.0, len(track) / track.fs - 1.0),
                        params={"activity": activity}))
    return out


def activity_run(activity: str, seed: int, dataset: str):
    """One activity clip after the 5 s calibration stand. Returns dict(track, worker, ..., window)."""
    rng = np.random.default_rng(seed)
    if dataset == "erciyes":
        with lab_conditions():
            site = sc.random_site(np.random.default_rng(seed + 7_919))
            worker = sc.random_worker(rng, site.clothing)
    else:
        site = random_site(np.random.default_rng(seed + 7_919))
        worker = sc.random_worker(rng, site.clothing)
    b = Builder(worker, site, rng)
    sc.begin(b)
    b.stand(1.0)
    t0 = b.t
    _act_body(b, rng, activity)
    t1 = b.t
    b.stand(1.0)
    track = b.build()
    fn, zeta = sc.soft_tissue(rng, site.clothing)
    return dict(track=track, worker=worker, soft_fn=fn, soft_zeta=zeta, window=(t0, t1), params={})
