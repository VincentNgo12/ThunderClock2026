# File: posture.py
# Purpose: Can the 4-IMU strip detect bad spine posture? Posture scenarios with ground-truth
#          segment angles, posture estimators run on the decoded/canonical frames, ergonomic
#          bad-posture rules, and accuracy / detection metrics.
# Contributors: Unassigned
"""Posture evaluation (spine-model research note, Q1-Q5).

Ground truth: the simulator's true segment orientations, as sagittal / lateral inclination of
each segment's long axis in the worker's heading frame (forward +, right +), relative to the
neutral stand, plus axial twist between pelvis and upper thoracic.

Estimators (per slot, from canonical frames only, i.e. what the laptop would receive):
- accel:  gravity = low-passed accelerometer (tau 0.5 s); works at any rate, fooled by motion.
- fusion: gravity-vector complementary filter (gyro propagation, accel correction gated to
          quasi-static samples) at >= 20 Hz; falls back to accel at lower rates. Also integrates
          relative axial rate for twist, reset during quiet upright periods (note Q2 step 4).
- +cal:   functional calibration from three forward bends removes each sensor's mounting
          rotation about the spine axis (note Q2 step 3).

Bad-posture rules (applied identically to truth and estimate):
- EN 1005-4 trunk zones 0-20 / 20-60 / > 60 deg (upper-thoracic inclination, note Q5);
- severe_flexion: trunk > 60 deg held >= 4 s (ISO 11226 treats > 4 s as static);
- iso_sustained: trunk in 20-60 deg (enter 20, exit 15) longer than T_max = 4 - 3(theta-20)/40 min;
- lateral_hold: |trunk lateral| > 20 deg held >= 4 s;
- flexed_curve: spine flexion (upper thoracic minus pelvis) > 25 deg held >= 30 s (slumped or
  stooped spinal curve);
- twist_hold: |axial twist| > 20 deg held >= 3 s (no magnetometer: expected to be unreliable).
Thresholds are the note's design defaults, not validated limits.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .model_api import GRAVITY
from .motions import Builder
from .runner import Pipeline, StreamOptions
from .scenarios import begin, random_worker, soft_tissue
from .site import random_site
from .spine import random_mounts, strip_kinematics
from .tune import Recorder

BOOT_S = 5.0  # device millis() starts 5 s before simulation time (ImuStrip.boot_offset_ms)
EPISODE_RULES = ("severe_flexion", "iso_sustained", "lateral_hold", "flexed_curve", "twist_hold")


# --- scenarios -----------------------------------------------------------------------------------
@dataclass(frozen=True)
class PostureScenario:
    id: str
    description: str
    build: Callable


POSTURES: dict[str, PostureScenario] = {}


def posture_scenario(id: str, description: str):
    def register(fn):
        POSTURES[id] = PostureScenario(id, description, fn)
        return fn
    return register


@posture_scenario("P-UPRIGHT", "Upright work: walking, standing, hammering at chest height (no bad posture)")
def _p_upright(b, rng):
    for _ in range(int(rng.integers(3, 6))):
        b.walk(float(rng.uniform(10, 30)))
        b.hammer(float(rng.uniform(10, 25)))
        b.stand(float(rng.uniform(5, 15)))


@posture_scenario("P-BENCH", "Working at a low bench: 25-50 deg trunk flexion held for minutes (ISO 11226)")
def _p_bench(b, rng):
    for _ in range(int(rng.integers(1, 3))):
        b.posture_hold(float(rng.uniform(25, 50)), float(rng.uniform(90, 240)), drift_deg=4.0)
        b.walk(float(rng.uniform(5, 15)))


@posture_scenario("P-STOOP-LIFT", "Repeated stoop lifts: trunk 65-95 deg, held 2-8 s")
def _p_stoop(b, rng):
    for _ in range(int(rng.integers(6, 12))):
        b.posture_hold(float(rng.uniform(65, 95)), float(rng.uniform(2, 8)), pelvis_share=0.5, drift_deg=2.0)
        b.walk(float(rng.uniform(3, 8)), load_kg=float(rng.uniform(10, 25)))


@posture_scenario("P-SQUAT-LIFT", "Squat-style lifts: trunk kept at 20-35 deg (good technique)")
def _p_squat(b, rng):
    for _ in range(int(rng.integers(6, 12))):
        b.move(1.2, y=b.ground_y + 0.32 * b.worker.height_m, pitch=np.deg2rad(rng.uniform(15, 25)),
               flex=np.deg2rad(rng.uniform(4, 10)) * np.array([0.4, 0.35, 0.25]))
        b.hold(float(rng.uniform(2, 6)), sway=0.4)
        b.stand_up(1.3)
        b.walk(float(rng.uniform(3, 8)), load_kg=float(rng.uniform(10, 25)))


@posture_scenario("P-REBAR", "Rebar tying: sustained 70-100 deg stoop for minutes")
def _p_rebar(b, rng):
    b.rebar_tie(float(rng.uniform(120, 240)))


@posture_scenario("P-LATERAL", "Reaching sideways: lateral bend 15-35 deg held 5-20 s")
def _p_lateral(b, rng):
    for _ in range(int(rng.integers(4, 8))):
        side = 1 if rng.random() < 0.5 else -1
        b.posture_hold(float(rng.uniform(5, 15)), float(rng.uniform(5, 20)), lat_deg=side * float(rng.uniform(15, 35)))
        b.stand(float(rng.uniform(3, 8)))


@posture_scenario("P-TWIST", "Twisted lifting: 30-45 deg flexion with 20-40 deg axial twist, held 3-10 s")
def _p_twist(b, rng):
    for _ in range(int(rng.integers(4, 8))):
        side = 1 if rng.random() < 0.5 else -1
        b.posture_hold(float(rng.uniform(30, 45)), float(rng.uniform(3, 10)), twist_deg=side * float(rng.uniform(20, 40)))
        b.walk(float(rng.uniform(3, 6)))


@posture_scenario("P-OVERHEAD", "Overhead drilling: trunk extended 10-30 deg")
def _p_overhead(b, rng):
    for _ in range(int(rng.integers(2, 4))):
        b.drill_overhead(float(rng.uniform(30, 60)))
        b.stand(float(rng.uniform(5, 10)))


@posture_scenario("P-SLOUCH", "Seated slouch: pelvis tilted back, lumbar/thoracic flexion 25-40 deg, trunk near upright")
def _p_slouch(b, rng):
    b.walk(float(rng.uniform(5, 10)))
    b.sit_posture(float(rng.uniform(120, 240)), slouch_deg=float(rng.uniform(25, 40)))


@posture_scenario("P-SIT-UPRIGHT", "Seated upright on a bench (no bad posture)")
def _p_sit_upright(b, rng):
    b.walk(float(rng.uniform(5, 10)))
    b.sit_posture(float(rng.uniform(120, 240)), slouch_deg=float(rng.uniform(0, 5)))


@posture_scenario("P-JACK", "Jackhammer: ~18 deg flexion under 20-40 Hz vibration (dynamic accuracy)")
def _p_jack(b, rng):
    for _ in range(int(rng.integers(1, 3))):
        b.jackhammer(float(rng.uniform(45, 90)))
        b.stand(float(rng.uniform(5, 15)))


@posture_scenario("P-CARRY", "Walking with a 15-30 kg load (leaning ~10 deg), dynamic accelerations")
def _p_carry(b, rng):
    for _ in range(int(rng.integers(2, 4))):
        b.walk(float(rng.uniform(30, 60)), load_kg=float(rng.uniform(15, 30)))
        b.stand(float(rng.uniform(3, 8)))


@posture_scenario("P-STRAP-SLIP", "Bench work, then the strap slips 10-20 deg on one anchor, then more bench work")
def _p_strap(b, rng):
    b.posture_hold(float(rng.uniform(30, 45)), float(rng.uniform(60, 120)), drift_deg=3.0)
    b.strap_slip(int(rng.integers(0, 4)), float(rng.uniform(10, 20)))
    b.walk(float(rng.uniform(5, 10)))
    b.posture_hold(float(rng.uniform(30, 45)), float(rng.uniform(60, 120)), drift_deg=3.0)


@dataclass
class PostureRun:
    scenario: str
    seed: int
    track: object
    worker: object
    site: object
    events: list
    soft_fn: np.ndarray
    soft_zeta: np.ndarray


def build_posture_run(scenario_id: str, seed: int) -> PostureRun:
    rng = np.random.default_rng(seed)
    site = random_site(np.random.default_rng(seed + 7_919))
    worker = random_worker(rng, site.clothing)
    b = Builder(worker, site, rng)
    begin(b)
    b.functional_calibration()
    POSTURES[scenario_id].build(b, rng)
    b.stand(5.0)
    from .calibrate import load_fn_scale

    fn, zeta = soft_tissue(rng, site.clothing, load_fn_scale())
    return PostureRun(scenario_id, seed, b.build(), worker, site, b.events, fn, zeta)


# --- ground truth ----------------------------------------------------------------------------------
def true_angles(run: PostureRun) -> dict:
    """Per-sample (1 kHz) truth: sagittal and lateral inclination of each segment, axial twist."""
    track = run.track
    kin = strip_kinematics(track, run.worker, random_mounts(run.worker, np.random.default_rng(0)))
    h = track.heading
    fwd = np.stack([-np.sin(h), np.zeros_like(h), -np.cos(h)], axis=1)
    right = np.stack([np.cos(h), np.zeros_like(h), -np.sin(h)], axis=1)
    sag = np.empty((len(track), 4))
    lat = np.empty((len(track), 4))
    for i in range(4):
        u = kin.seg_rot[i].apply([0.0, 1.0, 0.0])
        a, r = np.einsum("ij,ij->i", u, fwd), np.einsum("ij,ij->i", u, right)
        sag[:, i] = np.degrees(np.arctan2(a, u[:, 1]))
        lat[:, i] = np.degrees(np.arctan2(r, np.hypot(u[:, 1], a)))
    twist = np.degrees(track.twist.sum(axis=1))
    return {"t": track.t, "sag": sag, "lat": lat, "twist": twist}


# --- estimators ------------------------------------------------------------------------------------
def frames_for(run: PostureRun, profile: str) -> dict:
    rec = Recorder()
    options = StreamOptions(profile=profile, fast=True)
    mounts = random_mounts(run.worker, np.random.default_rng(run.seed + 3))
    pipe = Pipeline(rec, run.worker, run.site, mounts, run.soft_fn, run.soft_zeta, options, run.seed)
    pipe.feed(run.track, [(0.0, "normal")], run.site.ambient_c, len(run.track) / run.track.fs)
    pipe.finish()
    cat = lambda name: np.concatenate([getattr(b, name) for b in rec.batches])
    return {"t": cat("t") - BOOT_S, "acc": cat("acc").astype(float), "gyro": cat("gyro").astype(float),
            "mask": cat("slot_mask")}


def _angles_from_gravity(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sag = np.degrees(np.arctan2(g[..., 2], g[..., 1]))
    lat = np.degrees(np.arctan2(-g[..., 0], np.hypot(g[..., 1], g[..., 2])))
    return sag, lat


def estimate(frames: dict, method: str, cal_window: tuple[float, float] | None) -> dict:
    """Run one estimator over the frames; returns sag/lat (N,4) and twist (N,) in degrees."""
    t, acc, gyro, mask = frames["t"], frames["acc"], frames["gyro"], frames["mask"]
    n = len(t)
    g_out = np.empty((n, 4, 3))
    twist = np.full(n, np.nan)
    g = np.where(mask[0][:, None], acc[0], [0.0, GRAVITY, 0.0])
    tw, quiet_since = 0.0, None
    for k in range(n):
        dt = t[k] - t[k - 1] if k else 0.01
        a = np.where(mask[k][:, None], acc[k], g)
        w = np.where(mask[k][:, None], gyro[k], 0.0)
        if method == "fusion" and dt <= 0.05:
            # Propagate gravity with the gyro (dg/dt = -w x g), then pull towards the accelerometer,
            # only while the sample is quasi-static (|a| within 0.15 g of 1 g).
            g = g - np.cross(w, g) * dt
            static = np.abs(np.linalg.norm(a, axis=1) - GRAVITY) < 0.15 * GRAVITY
            gain = np.where(static, 1 - np.exp(-dt / 0.6), 1 - np.exp(-dt / 6.0))
            g = g + gain[:, None] * (a - g)
            g = g / np.linalg.norm(g, axis=1, keepdims=True) * GRAVITY
            # Relative axial rate (upper thoracic minus pelvis) about each sensor's long axis.
            tw += np.degrees(w[3, 1] - w[0, 1]) * dt
            sag_now, lat_now = _angles_from_gravity(g)
            upright = np.all(np.abs(sag_now) < 10) and np.all(np.abs(lat_now) < 10) and np.linalg.norm(w) < 0.15
            quiet_since = (quiet_since or t[k]) if upright else None
            if quiet_since is not None and t[k] - quiet_since > 2.0:
                tw = 0.0  # standing still and upright: assume no twist (note Q2 step 4)
            twist[k] = tw
        else:
            alpha = 1 - np.exp(-dt / 0.5)
            g = g + alpha * (a - g)
        g_out[k] = g
    if cal_window is not None:
        # Functional calibration: the forward bends tilt gravity into each sensor's sagittal plane.
        in_cal = (t >= cal_window[0]) & (t <= cal_window[1])
        for slot in range(4):
            gs = g_out[in_cal, slot]
            tilt = np.degrees(np.arctan2(np.hypot(gs[:, 0], gs[:, 2]), gs[:, 1])) if len(gs) else np.zeros(0)
            sel = gs[tilt > 15]
            if len(sel) == 0:
                continue
            phi = np.arctan2(sel[:, 0].mean(), sel[:, 2].mean())  # bend direction, ideally +Z
            c, s = np.cos(-phi), np.sin(-phi)
            x, z = g_out[:, slot, 0].copy(), g_out[:, slot, 2].copy()
            g_out[:, slot, 0], g_out[:, slot, 2] = x * c + z * s, -x * s + z * c
    sag, lat = _angles_from_gravity(g_out)
    return {"t": t, "sag": sag, "lat": lat, "twist": twist}


# --- bad-posture rules -----------------------------------------------------------------------------
def runs_above(t: np.ndarray, x: np.ndarray, enter: float, exit_: float, min_s: float) -> list[tuple[float, float]]:
    """Intervals where x rises above `enter` and stays above `exit_` for at least min_s."""
    out, start, on = [], None, False
    valid = np.isfinite(x)
    for k in range(len(t)):
        if not valid[k]:
            continue
        if not on and x[k] > enter:
            on, start = True, t[k]
        elif on and x[k] < exit_:
            on = False
            if t[k] - start >= min_s:
                out.append((start, t[k]))
    if on and t[-1] - start >= min_s:
        out.append((start, t[-1]))
    return out


def iso_limit_min(theta: float) -> float:
    """ISO 11226 acceptable holding time, linear reading of the summary (note Q5)."""
    return float(np.clip(4 - 3 * (theta - 20) / 40, 1.0, 4.0))


def episodes(series: dict, after: float) -> dict[str, list]:
    t = series["t"]
    keep = t >= after
    t = t[keep]
    trunk = series["sag"][keep, 3]
    lateral = np.abs(series["lat"][keep, 3])
    curve = series["sag"][keep, 3] - series["sag"][keep, 0]
    out = {
        "severe_flexion": runs_above(t, trunk, 60, 55, 4.0),
        "lateral_hold": runs_above(t, lateral, 20, 15, 4.0),
        "flexed_curve": runs_above(t, curve, 25, 20, 30.0),
        "twist_hold": runs_above(t, np.abs(series["twist"][keep]), 20, 15, 3.0) if np.isfinite(series["twist"]).any() else None,
    }
    iso = []
    for start, end in runs_above(t, trunk, 20, 15, 60.0):
        sel = (t >= start) & (t <= end)
        theta = float(np.median(trunk[sel]))
        if theta <= 60 and end - start > 60 * iso_limit_min(theta):
            iso.append((start, end))
    out["iso_sustained"] = iso
    return out


def match(truth: list, est: list | None) -> dict:
    if est is None:
        return {"truth": len(truth), "hit": None, "false": None, "latency": []}
    hit, latency, used = 0, [], set()
    for ts, te in truth:
        for j, (es, ee) in enumerate(est):
            if es <= te and ee >= ts:
                hit += 1
                latency.append(es - ts)
                used.add(j)
                break
    false = sum(1 for j, (es, ee) in enumerate(est) if j not in used and
                not any(es <= te and ee >= ts for ts, te in truth))
    return {"truth": len(truth), "hit": hit, "false": false, "latency": latency}


def zone(x: np.ndarray) -> np.ndarray:
    return np.digitize(x, [20.0, 60.0])  # 0: <20, 1: 20-60, 2: >60 (EN 1005-4)


def kappa(a: np.ndarray, b: np.ndarray, k: int = 3) -> float:
    cm = np.zeros((k, k))
    np.add.at(cm, (a, b), 1)
    n = cm.sum()
    po = np.trace(cm) / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / n / n
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def spine_shape_error_cm(true_sag: np.ndarray, est_sag: np.ndarray, spacing: np.ndarray) -> float:
    """RMS distance between reconstructed and true anchors (sagittal plane, relative to the
    pelvis), using the tape-measured spacings: the input to the Hermite spine curve."""
    def anchors(sag):
        rad = np.radians(sag)
        steps = np.stack([np.sin(rad[:, :3]), np.cos(rad[:, :3])], axis=2) * spacing[None, :, None]
        return np.cumsum(steps, axis=1)
    diff = anchors(true_sag) - anchors(est_sag)
    return float(100 * np.sqrt(np.nanmean(np.sum(diff**2, axis=2))))


# --- evaluation ------------------------------------------------------------------------------------
VARIANTS = (("accel", False), ("accel", True), ("fusion", False), ("fusion", True))


def evaluate_run(scenario_id: str, seed: int, profile: str) -> list[dict]:
    run = build_posture_run(scenario_id, seed)
    truth = true_angles(run)
    frames = frames_for(run, profile)
    times = {n: t for t, n, _ in run.events}
    cal = (times["functional_cal_start"], times["functional_cal_end"])
    after = cal[1] + 2.0
    # Truth at 50 Hz for episodes; at frame times for sample-level comparison.
    step = 20
    truth50 = {k: (v[::step] if k != "t" else v[::step]) for k, v in truth.items()}
    idx = np.clip(np.round(frames["t"] * run.track.fs).astype(int), 0, len(run.track) - 1)
    truth_f = {"t": frames["t"], "sag": truth["sag"][idx], "lat": truth["lat"][idx], "twist": truth["twist"][idx]}
    true_eps = episodes(truth50, after)
    results = []
    for method, use_cal in VARIANTS:
        est = estimate(frames, method, cal if use_cal else None)
        sel = est["t"] >= after
        err = est["sag"][sel] - truth_f["sag"][sel]
        trunk_err = np.abs(err[:, 3])
        lat_err = np.abs(est["lat"][sel, 3] - truth_f["lat"][sel, 3])
        curve_err = np.abs((est["sag"][sel, 3] - est["sag"][sel, 0]) - (truth_f["sag"][sel, 3] - truth_f["sag"][sel, 0]))
        lumbar_err = np.abs((est["sag"][sel, 1] - est["sag"][sel, 0]) - (truth_f["sag"][sel, 1] - truth_f["sag"][sel, 0]))
        tz, ez = zone(truth_f["sag"][sel, 3]), zone(est["sag"][sel, 3])
        twist_err = np.abs(est["twist"][sel] - truth_f["twist"][sel]) if np.isfinite(est["twist"][sel]).any() else None
        est_eps = episodes(est, after)
        row = {"scenario": scenario_id, "seed": seed, "profile": profile, "method": method, "functional_cal": use_cal,
               "frames": int(sel.sum()), "duration_s": round(float(est["t"][-1] - after), 1),
               "trunk_mae": float(np.nanmean(trunk_err)), "trunk_p95": float(np.nanpercentile(trunk_err, 95)),
               "trunk_bias": float(np.nanmean(err[:, 3])), "lateral_mae": float(np.nanmean(lat_err)),
               "curve_mae": float(np.nanmean(curve_err)), "lumbar_mae": float(np.nanmean(lumbar_err)),
               "twist_mae": None if twist_err is None else float(np.nanmean(twist_err)),
               "true_twist_rms": float(np.sqrt(np.mean(truth_f["twist"][sel] ** 2))),
               "zone_agreement": float(np.mean(tz == ez)), "zone_kappa": kappa(tz, ez),
               "time_gt60_true": float(np.mean(tz == 2)), "time_gt60_est": float(np.mean(ez == 2)),
               "time_20_60_true": float(np.mean(tz == 1)), "time_20_60_est": float(np.mean(ez == 1)),
               "shape_rms_cm": spine_shape_error_cm(truth_f["sag"][sel], est["sag"][sel], run.worker.spacings()),
               "episodes": {rule: match(true_eps[rule] or [], est_eps.get(rule)) for rule in EPISODE_RULES}}
        results.append(row)
    return results


def _task(args):
    try:
        return evaluate_run(*args)
    except Exception as error:  # keep the batch running
        return [{"scenario": args[0], "seed": args[1], "profile": args[2], "error": repr(error)}]


def posture_suite(out: Path, scenarios: list[str], seeds: list[int], profiles: list[str], jobs: int) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(s, seed, p) for p in profiles for s in scenarios for seed in seeds]
    rows = []
    with ProcessPoolExecutor(jobs) as pool, open(out / "posture_runs.jsonl", "w") as handle:
        for i, result in enumerate(pool.map(_task, tasks), 1):
            for row in result:
                handle.write(json.dumps(row) + "\n")
                rows.append(row)
            if i % 20 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} posture runs")
    return rows


# --- report ----------------------------------------------------------------------------------------
RULE_SCENARIOS = {  # which scenarios carry true episodes of each rule (for the per-rule tables)
    "severe_flexion": "P-STOOP-LIFT, P-REBAR", "iso_sustained": "P-BENCH, P-STRAP-SLIP",
    "lateral_hold": "P-LATERAL", "flexed_curve": "P-SLOUCH, P-REBAR, P-STOOP-LIFT, P-BENCH",
    "twist_hold": "P-TWIST",
}


def _variant_name(row: dict) -> str:
    return f"{row['method']}{' + cal' if row['functional_cal'] else ''}"


def summarise(rows: list[dict]) -> dict:
    from collections import defaultdict

    from .report import wilson

    groups = defaultdict(list)
    for r in rows:
        if "error" not in r:
            groups[(r["profile"], _variant_name(r))].append(r)
    out = {}
    for key, rs in groups.items():
        hours = sum(r["duration_s"] for r in rs) / 3600
        rules = {}
        for rule in EPISODE_RULES:
            ms = [r["episodes"][rule] for r in rs]
            if all(m["hit"] is None for m in ms):
                rules[rule] = None
                continue
            truth = sum(m["truth"] for m in ms if m["hit"] is not None)
            hit = sum(m["hit"] or 0 for m in ms)
            false = sum(m["false"] or 0 for m in ms)
            lat = [x for m in ms for x in m["latency"]]
            rules[rule] = {"truth": truth, "hit": hit, "sens": hit / truth if truth else None,
                           "ci": wilson(hit, truth) if truth else None, "false": false, "false_per_h": false / hours,
                           "latency_med": float(np.median(lat)) if lat else None}
        mean = lambda k: float(np.mean([r[k] for r in rs if r[k] is not None])) if any(r[k] is not None for r in rs) else None
        out[key] = {"profile": key[0], "variant": key[1], "runs": len(rs), "hours": hours,
                    "trunk_mae": mean("trunk_mae"), "trunk_p95": float(np.median([r["trunk_p95"] for r in rs])),
                    "lateral_mae": mean("lateral_mae"), "curve_mae": mean("curve_mae"), "lumbar_mae": mean("lumbar_mae"),
                    "shape_rms_cm": mean("shape_rms_cm"), "twist_mae": mean("twist_mae"), "zone_agreement": mean("zone_agreement"),
                    "gt60_err_pp": float(np.mean([abs(r["time_gt60_est"] - r["time_gt60_true"]) for r in rs]) * 100),
                    "mid_err_pp": float(np.mean([abs(r["time_20_60_est"] - r["time_20_60_true"]) for r in rs]) * 100),
                    "rules": rules}
    return out


def per_scenario(rows: list[dict], profile: str, variant: str) -> list[dict]:
    from collections import defaultdict

    g = defaultdict(list)
    for r in rows:
        if "error" not in r and r["profile"] == profile and _variant_name(r) == variant:
            g[r["scenario"]].append(r)
    table = []
    for sid in POSTURES:
        rs = g.get(sid, [])
        if not rs:
            continue
        eps = {rule: (sum(r["episodes"][rule]["hit"] or 0 for r in rs), sum(r["episodes"][rule]["truth"] for r in rs),
                      sum(r["episodes"][rule]["false"] or 0 for r in rs)) for rule in EPISODE_RULES
               if any(r["episodes"][rule]["hit"] is not None for r in rs)}
        table.append({"scenario": sid, "n": len(rs), "trunk_mae": np.mean([r["trunk_mae"] for r in rs]),
                      "trunk_p95": np.median([r["trunk_p95"] for r in rs]), "lateral_mae": np.mean([r["lateral_mae"] for r in rs]),
                      "curve_mae": np.mean([r["curve_mae"] for r in rs]), "shape": np.mean([r["shape_rms_cm"] for r in rs]),
                      "eps": eps})
    return table


def write_posture_report(rows: list[dict], out: Path, notes: list[str]) -> None:
    summary = summarise(rows)
    order = [(p, v) for p in ("target", "current_100hz", "current") for v in ("accel", "accel + cal", "fusion", "fusion + cal")]
    keys = [k for k in order if k in summary]
    lines = ["# Can the spine strip detect bad posture?", "",
             "**Simulated data only.** Ground truth is the simulator's true segment orientation; estimates use only the decoded "
             "sensor frames, exactly what the laptop would receive. Thresholds are the spine-model note's design defaults "
             "(EN 1005-4 zones, ISO 11226 holding times), not validated limits.", ""] + [f"- {n}" for n in notes] + [""]
    lines += ["## Angle accuracy (mean over runs)", "",
              "| Firmware | Estimator | Runs | Trunk MAE deg | Trunk p95 deg | Lateral MAE deg | Spine-curve MAE deg | Lumbar MAE deg | Spine shape RMS cm | Twist MAE deg | EN zone agreement | Time >60 deg error pp | Time 20-60 deg error pp |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda x, d=1: "-" if x is None else f"{x:.{d}f}"
    for k in keys:
        s = summary[k]
        lines.append(f"| {s['profile']} | {s['variant']} | {s['runs']} | {fmt(s['trunk_mae'])} | {fmt(s['trunk_p95'])} | {fmt(s['lateral_mae'])} | "
                     f"{fmt(s['curve_mae'])} | {fmt(s['lumbar_mae'])} | {fmt(s['shape_rms_cm'])} | {fmt(s['twist_mae'])} | "
                     f"{100 * s['zone_agreement']:.1f} % | {fmt(s['gt60_err_pp'])} | {fmt(s['mid_err_pp'])} |")
    lines += ["", "## Bad-posture episode detection (pooled over all posture scenarios)", "",
              "Sensitivity = true episodes overlapped by a detected episode of the same rule (95 % Wilson CI). False = detected "
              "episodes overlapping no true one, per hour of observed work.", ""]
    lines += ["| Firmware | Estimator | " + " | ".join(EPISODE_RULES) + " |", "|---|---|" + "---|" * len(EPISODE_RULES)]
    for k in keys:
        s = summary[k]
        cells = []
        for rule in EPISODE_RULES:
            r = s["rules"][rule]
            if r is None:
                cells.append("not measurable")
            elif r["sens"] is None:
                cells.append(f"no true episodes; {r['false_per_h']:.2f} false/h")
            else:
                lo, hi = r["ci"]
                cells.append(f"{r['hit']}/{r['truth']} = {100 * r['sens']:.0f} % ({100 * lo:.0f}-{100 * hi:.0f}); {r['false_per_h']:.2f} false/h")
        lines.append(f"| {s['profile']} | {s['variant']} | " + " | ".join(cells) + " |")
    lines += ["", "Rules: " + "; ".join(f"**{r}** ({RULE_SCENARIOS[r]})" for r in EPISODE_RULES) + ".", ""]
    for profile, variant in (("target", "fusion + cal"), ("current", "accel + cal")):
        table = per_scenario(rows, profile, variant)
        if not table:
            continue
        lines += [f"## Per scenario: `{profile}` firmware, {variant}", "",
                  "| Scenario | Runs | Trunk MAE | Trunk p95 | Lateral MAE | Spine-curve MAE | Shape RMS cm | Episodes hit/true (false) |",
                  "|---|---|---|---|---|---|---|---|"]
        for row in table:
            eps = "; ".join(f"{rule} {h}/{t} ({f})" for rule, (h, t, f) in row["eps"].items() if t or f) or "none"
            lines.append(f"| {row['scenario']} | {row['n']} | {row['trunk_mae']:.1f} | {row['trunk_p95']:.1f} | {row['lateral_mae']:.1f} | "
                         f"{row['curve_mae']:.1f} | {row['shape']:.1f} | {eps} |")
        lines.append("")
    lines += ["## Scenarios", ""] + [f"- **{s.id}**: {s.description}" for s in POSTURES.values()]
    (out / "posture_report.md").write_text("\n".join(lines))
    (out / "posture_summary.json").write_text(json.dumps([{**v, "rules": v["rules"]} for v in summary.values()], indent=1, default=str))


def plot_posture_example(out: Path, scenario_id: str = "P-STOOP-LIFT", seed: int = 1) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import CubicHermiteSpline

    run = build_posture_run(scenario_id, seed)
    truth = true_angles(run)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    for ax, profile, method in ((axes[0], "target", "fusion"), (axes[1], "current", "accel")):
        frames = frames_for(run, profile)
        times = {n: t for t, n, _ in run.events}
        est = estimate(frames, method, (times["functional_cal_start"], times["functional_cal_end"]))
        ax.plot(truth["t"][::20], truth["sag"][::20, 3], color="#222", lw=1.2, label="true trunk inclination")
        ax.plot(est["t"], est["sag"][:, 3], color="#d62728", lw=0.9, marker="o" if profile == "current" else None,
                ms=3, label=f"estimated ({profile} firmware, {method} + cal)")
        ax.plot(truth["t"][::20], truth["sag"][::20, 3] - truth["sag"][::20, 0], color="#1f77b4", lw=0.8, alpha=0.7,
                label="true spine flexion (upper thoracic - pelvis)")
        for y in (20, 60):
            ax.axhline(y, color="#999", ls=":", lw=0.8)
        ax.set_ylabel("degrees")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title(f"{scenario_id} seed {seed}: {profile} firmware", fontsize=9)
    axes[1].set_xlabel("time (s)")
    fig.tight_layout()
    path = out / f"posture_{scenario_id}_{seed}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)

    # Spine curve: true vs estimated anchors (sagittal plane) through Hermite interpolation.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    frames = frames_for(run, "target")
    times = {n: t for t, n, _ in run.events}
    est = estimate(frames, "fusion", (times["functional_cal_start"], times["functional_cal_end"]))
    spacing = run.worker.spacings()
    after = int((times["functional_cal_end"] + 2.0) * run.track.fs)
    curve_peak = after + int(np.nanargmax(truth["sag"][after:, 3] - truth["sag"][after:, 0]))
    trunk_peak = after + int(np.nanargmax(truth["sag"][after:, 3]))
    moments = [("neutral", int(2.0 * run.track.fs)), ("max spine flexion", curve_peak), ("max trunk inclination", trunk_peak)]
    for ax, (label, i) in zip(axes, moments):
        j = int(np.argmin(np.abs(frames["t"] - truth["t"][i])))
        for sag, colour, name in ((truth["sag"][i], "#222", "true"), (est["sag"][j], "#d62728", "estimated")):
            rad = np.radians(sag)
            pts = np.vstack([[0.0, 0.0], np.cumsum(np.stack([np.sin(rad[:3]), np.cos(rad[:3])], 1) * spacing[:, None], 0)])
            tangents = np.stack([np.sin(rad), np.cos(rad)], 1)
            chord = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
            curve = CubicHermiteSpline(chord, pts, tangents, axis=0)(np.linspace(0, chord[-1], 50))
            ax.plot(curve[:, 0] * 100, curve[:, 1] * 100, color=colour, label=name)
            ax.scatter(pts[:, 0] * 100, pts[:, 1] * 100, color=colour, s=18)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{label} @ {truth['t'][i]:.1f} s", fontsize=9)
        ax.set_xlabel("forward (cm)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("up from pelvis sensor (cm)")
    fig.tight_layout()
    path2 = out / f"posture_curve_{scenario_id}_{seed}.png"
    fig.savefig(path2, dpi=110)
    plt.close(fig)
    return path
