# File: motion_validation.py
# Purpose: Compare the simulator's trunk motion with real human motion: the canonical mocap clips
#          (motionlib/canonical), the team's own vest recordings (backend JSONL) and cited literature
#          ranges, with the same per-activity metrics for every source. CLI: `thundersim motionval`.
# Contributors: Unassigned
"""Motion validation (simulator step 1).

Every source is cut into activity windows (walk, carry_walk, run, lift, kneel, ladder, idle). A
window holds, at 100 Hz:
  angles (N,12)  trunk angles in the simulator convention (heading, pitch, roll, flex x3, lat x3,
                 twist x3, rad), zero in the neutral stand
  pos (N,3)      pelvis-anchor position (m, +Y up), or None when the source has no floor
  acc (N,3)      pelvis IMU specific force (m/s^2), rotated so +Y is gravity in the neutral stand
                 (the product's neutral calibration, canonical.Canonicalizer), or None
The same metric functions then run on every window, so a difference between sources is a
difference in motion, not in the analysis.

Sources:
  sim    scenario runs through spine kinematics, soft tissue and the MPU-6050 emulator (target
         profile), the primitive windows from viz.recorder.RecordingBuilder
  mocap  motionlib canonical clips, windows found from the motion itself (speed, bends, climbing);
         the pelvis IMU is derived from the S1 anchor path (band-limited by the source frame rate)
  vest   the team's own recordings: backend `app.cli stream --record` JSONL (raw and/or
         quaternion) plus labels.csv and meta.json per session (validation/RECORDING_PROTOCOL.md)
  lit    LITERATURE below; each range names its source and what it measures

Simulated and recorded data only: no medical or clinical claim follows from any of this.
"""

from __future__ import annotations

import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.spatial.transform import Rotation, Slerp

from .canonical import neutral_alignment
from .device import PROFILES, DeviceErrors, ImuStrip
from .model_api import GRAVITY
from .scenarios import CALIBRATION_S, build_run
from .spine import (UP, apply_soft_tissue, joint_rotation, random_mounts, root_rotation, strip_kinematics,
                    virtual_imu)

FS = 100.0                       # analysis rate (Hz), the vest's target rate
SIM_ROOT = Path(__file__).resolve().parents[1]
MOCAP_DIR = SIM_ROOT / "motionlib" / "canonical"
RECORDINGS_DIR = SIM_ROOT / "validation" / "recordings"
ACTIVITIES = ("walk", "carry_walk", "run", "lift", "kneel", "ladder", "idle")
SEGMENTS = ("pelvis", "lumbar", "thoracic", "upper")
DEFAULT_SCENARIOS = ("A-CARRY", "A-RUN", "A-KNEEL", "A-LADDER")
ANGLE_LOWPASS_HZ = 6.0           # angle smoothing before rates and accelerations, the same for every source
MIN_SECONDS = {"walk": 1.5, "carry_walk": 1.5, "run": 0.8, "kneel": 3.0, "ladder": 2.0, "idle": 3.0}

# --- literature ranges -----------------------------------------------------------------------------
# (activity, metric) -> (low, high, source and what the numbers are). Only ranges whose numbers were
# read from the source (or its abstract) are listed. Indicative ranges are marked as such.
LITERATURE = {
    ("walk", "bob_cm"): (2.7, 4.8, "Orendurff 2004 (J Rehabil Res Dev 41:829), 10 adults: vertical COM excursion "
                                   "2.74 +- 0.52 cm at 0.7 m/s to 4.83 +- 0.92 cm at 1.6 m/s (COM, not the sacrum)"),
    ("walk", "ml_cm"): (3.9, 7.0, "Orendurff 2004: mediolateral COM excursion 6.99 +- 1.34 cm at 0.7 m/s to "
                                  "3.85 +- 1.41 cm at 1.6 m/s"),
    ("walk", "pelvis_roll_pp_deg"): (3.5, 10.1, "Latorre 2019 (J NeuroEng Rehabil 16:97), Kinect v2, 161 adults 20-49 y "
                                                "at 1.16-1.19 m/s: pelvic obliquity range 6.5-7.6 +- 2.0-3.1 deg "
                                                "(mean +- 1 SD across decades); indicative (markerless)"),
    ("walk", "pelvis_yaw_pp_deg"): (6.4, 16.0, "Latorre 2019: pelvic rotation range 10.2-13.1 +- 2.9-3.7 deg "
                                               "(mean +- 1 SD across decades); indicative (markerless)"),
    ("lift", "peak_rate_thoracic_dps"): (126.0, 150.0, "Saraceni 2021 (PLoS One 16:e0254241), 41 adults, self-selected "
                                                       "floor lifts: peak thorax velocity 138 deg/s, 95 % CI of the mean"),
    ("lift", "peak_thoracic_deg"): (97.3, 108.3, "Saraceni 2021: peak thorax inclination 102.8 deg, 95 % CI of the "
                                                 "mean; floor lifts only (higher grips bend less)"),
    ("lift", "pelvis_at_peak_deg"): (45.0, 57.6, "Saraceni 2021: pelvis inclination at lift-off 51.3 deg, 95 % CI of "
                                                 "the mean; floor lifts only"),
    ("lift", "knee_deg"): (73.2, 98.6, "Saraceni 2021: peak knee flexion 85.9 deg, 95 % CI of the mean (self-selected, "
                                       "mostly semi-squat); stoops bend less, squats more"),
    ("lift", "rise_s"): (1.2, 2.1, "von Arx 2021 (Front Bioeng Biotechnol 9:769117): lift-up time 4.6 +- 0.7 "
                                   "(freestyle) to 5.9 +- 1.1 (stoop) x sqrt(l0/g) ~ 0.30 s, mean +- 1 SD; their phase "
                                   "is defined by the box, ours by the thorax angle (10-90 %)"),
}


# (activity, metric) -> why the mocap reference cannot judge the simulator there. Such rows are shown
# but get the verdict "not judged".
REFERENCE_CAVEATS = {
    ("run", "pelvis_pitch_mean_deg"): "the CMU running clips have no standing neutral (zero = most upright running frames)",
    ("run", "upper_incl_mean_deg"): "the CMU running clips have no standing neutral (zero = most upright running frames)",
    ("kneel", "upper_incl_p10_deg"): "NIOSH zero is each clip's most upright kneeling moment, so its p10 is ~0 by construction",
    ("kneel", "pelvis_share"): "NIOSH segments come from three spine points (0.93) and CMU 23_03 disagrees (0.02)",
    ("lift", "peak_accel_thoracic_dps2"): "second derivative of 30-120 Hz mocap angles: noise-inflated",
}


# --- data containers ---------------------------------------------------------------------------------
@dataclass
class Window:
    """One activity window of one source, at FS. See the module docstring for the arrays."""

    source: str                 # sim | mocap | vest
    clip: str                   # scenario/seed, clip id or session id
    activity: str
    angles: np.ndarray
    pos: np.ndarray | None = None
    acc: np.ndarray | None = None
    height_m: float | None = None
    info: dict = field(default_factory=dict)   # load_kg, technique, kind, knee_deg, imu_band_hz, notes

    @property
    def seconds(self) -> float:
        return len(self.angles) / FS


# --- shared geometry and signal helpers ----------------------------------------------------------------
def segment_rotations(angles: np.ndarray) -> list[Rotation]:
    """Segment -> world rotations rebuilt with the simulator's own functions (spine.py)."""
    seg = [root_rotation(angles[:, 0], angles[:, 1], angles[:, 2])]
    for j in range(3):
        seg.append(seg[-1] * joint_rotation(angles[:, 3 + j], angles[:, 6 + j], angles[:, 9 + j]))
    return seg


def segment_inclinations(angles: np.ndarray) -> np.ndarray:
    """(N,4) sagittal inclination (deg) of the pelvis, lumbar, thoracic and upper-thoracic segments: the
    segment's long axis from vertical, forward positive, in the plane of the pelvis heading."""
    heading = angles[:, 0]
    forward = np.stack([-np.sin(heading), np.zeros(len(heading)), -np.cos(heading)], axis=1)
    out = np.empty((len(angles), 4))
    for i, rot in enumerate(segment_rotations(angles)):
        axis = rot.apply(UP)
        out[:, i] = np.rad2deg(np.arctan2(np.sum(axis * forward, axis=1), axis[:, 1]))
    return out


def lowpass(x: np.ndarray, hz: float, fs: float = FS) -> np.ndarray:
    """Zero-phase 2nd-order Butterworth (4th order effective); returns x unchanged if too short."""
    if len(x) < 30 or hz >= fs / 2:
        return x
    sos = signal.butter(2, hz, fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def highpass(x: np.ndarray, hz: float, fs: float = FS) -> np.ndarray:
    if len(x) < 30:
        return x - np.mean(x, axis=0)
    sos = signal.butter(2, hz, "high", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def dominant_hz(x: np.ndarray, band: tuple[float, float], fs: float = FS) -> float:
    """Frequency of the largest spectral peak of x inside band (Hz), zero-padded for resolution."""
    x = signal.detrend(np.asarray(x, float))
    if len(x) < 2 * fs / band[1]:
        return float("nan")
    n_fft = max(8192, len(x))
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    inside = (freqs >= band[0]) & (freqs <= band[1])
    return float(freqs[inside][np.argmax(spec[inside])])


def per_cycle_range(x: np.ndarray, cycle_hz: float, fs: float = FS) -> float:
    """Median peak-to-peak of x over consecutive whole cycles of length 1 / cycle_hz."""
    if not np.isfinite(cycle_hz) or cycle_hz <= 0:
        return float("nan")
    k = int(round(fs / cycle_hz))
    if k < 3 or len(x) < k:
        return float("nan")
    ranges = [np.ptp(x[i:i + k]) for i in range(0, len(x) - k + 1, k)]
    return float(np.median(ranges))


def runs_where(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """[start, end) index runs where mask is True for at least min_len samples."""
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= min_len]


def horizontal_speed(pos: np.ndarray, smooth_s: float = 1.0) -> np.ndarray:
    vel = np.gradient(pos[:, [0, 2]], 1.0 / FS, axis=0)
    return lowpass(np.linalg.norm(vel, axis=1), 1.0 / smooth_s)


def segment_rates(angles: np.ndarray) -> np.ndarray:
    """(N-1,4) angular speed (rad/s) of each trunk segment, 10 Hz low-passed (what a gyro on it reads)."""
    out = []
    for rot in segment_rotations(angles):
        rel = rot[:-1].inv() * rot[1:]
        out.append(np.linalg.norm(lowpass(rel.as_rotvec() * FS, 10.0), axis=1))
    return np.array(out).T


def knee_angle_deg(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> np.ndarray:
    """Knee flexion (deg): 0 with a straight leg."""
    a, b = hip - knee, ankle - knee
    cos = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return 180.0 - np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0)))


# --- metrics -------------------------------------------------------------------------------------------
def gait_metrics(w: Window) -> dict:
    """Walking, carrying and running: cadence, pelvis bob and sway, pelvis angles per stride, trunk lean,
    and the pelvis IMU's vertical specific force (per-step peak, RMS, high-frequency share)."""
    band = (2.0, 3.8) if w.activity == "run" else (1.2, 2.6)
    out = {}
    incl = segment_inclinations(w.angles)
    if w.pos is not None:
        out["speed_mps"] = float(np.mean(horizontal_speed(w.pos)))
        step_hz = dominant_hz(w.pos[:, 1], band)
        out["step_length_m"] = out["speed_mps"] / step_hz
    elif w.acc is not None:
        step_hz = dominant_hz(w.acc[:, 1], band)
    else:
        step_hz = float("nan")
    out["step_hz"] = step_hz
    stride_hz = step_hz / 2.0
    if w.pos is not None:
        bob = signal.detrend(lowpass(w.pos[:, 1], 8.0))
        out["bob_cm"] = 100 * per_cycle_range(bob, step_hz)
        path = lowpass(w.pos[:, [0, 2]], 0.5)
        direction = np.gradient(path, axis=0)
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
        right = np.stack([-direction[:, 1], direction[:, 0]], axis=1)
        lateral = np.sum((w.pos[:, [0, 2]] - path) * right, axis=1)
        out["ml_cm"] = 100 * per_cycle_range(lateral, stride_hz)
    heading = np.rad2deg(np.unwrap(w.angles[:, 0]))
    out["pelvis_yaw_pp_deg"] = per_cycle_range(highpass(heading, 0.4), stride_hz)
    out["pelvis_roll_pp_deg"] = per_cycle_range(highpass(np.rad2deg(w.angles[:, 2]), 0.4), stride_hz)
    out["pelvis_pitch_pp_deg"] = per_cycle_range(highpass(np.rad2deg(w.angles[:, 1]), 0.4), stride_hz)
    twist = np.rad2deg(w.angles[:, 9:12].sum(axis=1))
    out["thorax_pelvis_yaw_pp_deg"] = per_cycle_range(highpass(twist, 0.4), stride_hz)
    out["pelvis_pitch_mean_deg"] = float(np.mean(incl[:, 0]))
    out["upper_incl_mean_deg"] = float(np.mean(incl[:, 3]))
    out["trunk_rate_p95_rads"] = float(np.max(np.percentile(segment_rates(w.angles), 95, axis=0)))
    if w.acc is not None:
        vertical_g = w.acc[:, 1] / GRAVITY
        k = int(round(FS / step_hz)) if np.isfinite(step_hz) else 0
        if k >= 3 and len(vertical_g) >= k:
            peaks = [np.max(vertical_g[i:i + k]) for i in range(0, len(vertical_g) - k + 1, k)]
            out["imu_step_peak_g"] = float(np.median(peaks))
            # The same with every source limited to 10 Hz, below every mocap source's bandwidth.
            slow = lowpass(vertical_g, 10.0)
            out["imu_step_peak_10hz_g"] = float(np.median([np.max(slow[i:i + k]) for i in range(0, len(slow) - k + 1, k)]))
        out["imu_vert_rms_g"] = float(np.std(vertical_g))
        out["imu_peak_norm_g"] = float(np.percentile(np.linalg.norm(w.acc, axis=1) / GRAVITY, 99))
        if w.info.get("imu_band_hz", FS / 2) >= 20.0 and len(vertical_g) >= 256:
            freqs, psd = signal.welch(vertical_g - vertical_g.mean(), FS, nperseg=256)
            total = psd[freqs >= 0.5].sum()
            out["imu_share_above_5hz"] = float(psd[freqs >= 5.0].sum() / total)
            out["imu_share_above_10hz"] = float(psd[freqs >= 10.0].sum() / total)
    return out


def find_bends(incl_upper: np.ndarray, min_peak_deg: float = 30.0) -> list[tuple[int, int, int]]:
    """Forward bends in the upper-thoracic inclination: (start, peak, end) indices, where start and end
    are the last and first samples below 10 % of the bend's peak."""
    smooth = lowpass(incl_upper, ANGLE_LOWPASS_HZ)
    peaks, _ = signal.find_peaks(smooth, height=min_peak_deg, prominence=0.6 * min_peak_deg, distance=int(1.0 * FS))
    bends = []
    for p in peaks:
        level = 0.1 * smooth[p]
        below = np.where(smooth[:p] < level)[0]
        after = np.where(smooth[p:] < level)[0]
        start = int(below[-1]) if len(below) else 0
        end = int(p + after[0]) if len(after) else len(smooth) - 1
        bends.append((start, int(p), end))
    return bends


def lift_metrics(w: Window, bend: tuple[int, int, int], knee_deg: float | None = None) -> dict:
    """One bend: peak inclination per segment, pelvis share (lumbo-pelvic ratio), lumbar and whole-spine
    flexion, pelvis drop, thoracic peak rate and acceleration, and lower / hold / rise durations."""
    start, peak, end = bend
    incl = lowpass(segment_inclinations(w.angles), ANGLE_LOWPASS_HZ)
    out = {f"peak_{name}_deg": float(np.max(incl[start:end + 1, i])) for i, name in enumerate(SEGMENTS)}
    at = int(start + np.argmax(incl[start:end + 1, 3]))
    out["pelvis_at_peak_deg"] = float(incl[at, 0])
    out["pelvis_share"] = float(incl[at, 0] / max(incl[at, 3], 1e-6))
    out["spine_flex_at_peak_deg"] = float(np.rad2deg(w.angles[at, 3:6].sum()))
    out["lumbar_flex_at_peak_deg"] = float(incl[at, 1] - incl[at, 0]) + float(np.rad2deg(w.angles[at, 3]))
    thorax = incl[:, 2]
    rate = np.gradient(thorax, 1.0 / FS)
    out["peak_rate_thoracic_dps"] = float(np.max(np.abs(rate[start:end + 1])))
    out["peak_accel_thoracic_dps2"] = float(np.max(np.abs(np.gradient(rate, 1.0 / FS)[start:end + 1])))
    upper = incl[:, 3]
    top = upper[at]
    above = np.where(upper[start:end + 1] >= 0.9 * top)[0] + start
    lo10 = np.where(upper[start:at + 1] <= 0.1 * top)[0]
    hi10 = np.where(upper[at:end + 1] <= 0.1 * top)[0]
    t_down10 = (start + lo10[-1]) if len(lo10) else start
    t_up10 = (at + hi10[0]) if len(hi10) else end
    out["lower_s"] = float((above[0] - t_down10) / FS)
    out["hold_s"] = float((above[-1] - above[0]) / FS)
    out["rise_s"] = float((t_up10 - above[-1]) / FS)
    if w.pos is not None:
        stand = float(np.median(w.pos[max(start - int(0.3 * FS), 0):start + 1, 1]))
        drop = stand - float(np.min(w.pos[start:end + 1, 1]))
        out["pelvis_drop_m"] = drop
        if w.height_m:
            out["pelvis_drop_per_height"] = drop / w.height_m
    if knee_deg is not None:
        out["knee_deg"] = float(knee_deg)
    return out


def kneel_metrics(w: Window) -> dict:
    """Kneeling work: trunk inclination level and spread, pelvis share and pelvis height."""
    incl = lowpass(segment_inclinations(w.angles), ANGLE_LOWPASS_HZ)
    out = {"upper_incl_mean_deg": float(np.mean(incl[:, 3])),
           "upper_incl_p10_deg": float(np.percentile(incl[:, 3], 10)),
           "upper_incl_p90_deg": float(np.percentile(incl[:, 3], 90)),
           "pelvis_pitch_mean_deg": float(np.mean(incl[:, 0])),
           "pelvis_share": float(np.mean(incl[:, 0]) / max(abs(np.mean(incl[:, 3])), 1e-6)),
           "upper_incl_rate_p95_dps": float(np.percentile(np.abs(np.gradient(incl[:, 3], 1 / FS)), 95))}
    if w.pos is not None and w.info.get("floor", True) and w.height_m:
        out["pelvis_height_per_height"] = float(np.median(w.pos[:, 1]) / w.height_m)
    return out


def ladder_metrics(w: Window) -> dict:
    """Ladder climbing: rung rate, climb speed, trunk lean, and pelvis roll and sway per rung. A roll that
    alternates sides has half the rung frequency (roll_to_rung_freq ~ 0.5); one that repeats the same
    side every rung has roll_to_rung_freq ~ 1."""
    incl = segment_inclinations(w.angles)
    out = {"pitch_mean_deg": float(np.mean(incl[:, 0])), "upper_incl_mean_deg": float(np.mean(incl[:, 3]))}
    roll = highpass(np.rad2deg(w.angles[:, 2]), 0.15)
    if w.pos is not None:
        vy = np.gradient(lowpass(w.pos[:, 1], 4.0), 1.0 / FS)
        rung_hz = dominant_hz(np.abs(vy), (0.4, 2.0))
        out["rung_hz"] = rung_hz
        out["climb_speed_mps"] = float(np.mean(np.abs(vy)))
        out["rise_per_rung_m"] = out["climb_speed_mps"] / rung_hz
        out["roll_pp_deg"] = per_cycle_range(roll, rung_hz / 2.0)
        out["roll_to_rung_freq"] = dominant_hz(roll, (0.15, 2.0)) / rung_hz
        heading = w.angles[:, 0]
        right = np.stack([np.cos(heading), -np.sin(heading)], axis=1)      # pelvis right axis in (x, z)
        lateral = np.sum(highpass(w.pos[:, [0, 2]], 0.15) * right, axis=1)
        out["sway_pp_cm"] = 100 * per_cycle_range(lateral, rung_hz / 2.0)
    return out


def idle_metrics(w: Window) -> dict:
    """Quiet standing: pelvis sway and trunk angle RMS after removing slow drift (0.05 Hz)."""
    out = {"pitch_rms_deg": float(np.std(highpass(np.rad2deg(w.angles[:, 1]), 0.05))),
           "roll_rms_deg": float(np.std(highpass(np.rad2deg(w.angles[:, 2]), 0.05))),
           "spine_flex_rms_deg": float(np.std(highpass(np.rad2deg(w.angles[:, 3:6].sum(axis=1)), 0.05)))}
    if w.pos is not None:
        sway = highpass(w.pos[:, [0, 2]], 0.05)
        out["sway_rms_mm"] = float(1000 * np.sqrt(np.mean(np.sum(sway**2, axis=1))))
    return out


def window_metrics(w: Window) -> list[dict]:
    """Metrics of one window: one row per bend for lifts, one row otherwise."""
    if w.activity in ("walk", "carry_walk", "run"):
        rows = [gait_metrics(w)]
    elif w.activity == "lift":
        incl = segment_inclinations(w.angles)
        knee = w.info.get("knee_series")
        rows = []
        for bend in find_bends(incl[:, 3]):
            k = None
            if knee is not None:
                k = float(np.max(knee[bend[0]:bend[2] + 1]))
            elif w.info.get("knee_deg") is not None:
                k = w.info["knee_deg"]
            rows.append(lift_metrics(w, bend, k))
            rows[-1]["_series"] = incl[max(bend[0] - 20, 0):bend[2] + 20, 2]
    elif w.activity == "kneel":
        rows = [kneel_metrics(w)]
    elif w.activity == "ladder":
        rows = [ladder_metrics(w)]
    else:
        rows = [idle_metrics(w)]
    for row in rows:
        row.update({"_source": w.source, "_clip": w.clip, "_activity": w.activity, "_seconds": w.seconds,
                    "_technique": w.info.get("technique"), "_load_kg": w.info.get("load_kg"),
                    "_kind": w.info.get("kind")})
    return rows


# --- source (a): the simulator ---------------------------------------------------------------------------
def sim_imu(run, seed: int) -> np.ndarray:
    """Pelvis specific force at 100 Hz as the target-profile vest would deliver it, neutral-aligned."""
    track = run.track
    mounts = random_mounts(run.worker, np.random.default_rng(seed + 5))
    kin = strip_kinematics(track, run.worker, mounts)
    acc, gyro = virtual_imu(kin, track.fs)
    acc, gyro = apply_soft_tissue(acc, gyro, track.fs, run.soft_fn, run.soft_zeta)
    strip = ImuStrip(PROFILES["target"], 4, DeviceErrors(), seed=seed + 1)
    t = np.arange(len(track)) / track.fs
    samples = strip.process(t, acc, gyro, strip.calibration_temp_c)
    pelvis = samples.acc[:, 0]
    quiet = pelvis[samples.t_true <= CALIBRATION_S]
    aligned = neutral_alignment(quiet.mean(axis=0)).apply(pelvis)
    # Put the samples on the analysis grid (sample k at k / FS).
    grid = np.arange(int(len(track) / track.fs * FS)) / FS
    return np.stack([np.interp(grid, samples.t_true, aligned[:, i]) for i in range(3)], axis=1)


def sim_windows(scenario: str, seed: int, obs_s: float = 60.0) -> list[Window]:
    """Run one scenario and cut it into activity windows from the recorded primitive calls."""
    from .viz.recorder import RecordingBuilder

    sink: list = []
    run = build_run(scenario, seed, obs_s=obs_s, builder=lambda w, s, r: RecordingBuilder(w, s, r, sink=sink))
    rec = sink[0]
    track = run.track
    step = int(round(track.fs / FS))
    angles = np.column_stack([track.heading, track.pitch, track.roll, track.flex, track.lat, track.twist])[::step]
    pos = track.root_pos[::step]
    acc = sim_imu(run, seed)
    n = min(len(angles), len(acc))
    angles, pos, acc = angles[:n], pos[:n], acc[:n]
    height = run.worker.height_m
    clip = f"{scenario}/{seed}"
    lift_events = [(t, name, info) for t, name, info in run.events if name in ("lift_grasp", "put_down_touch")]
    windows = []

    def cut(t0: float, t1: float, activity: str, **info) -> None:
        a, b = int(max(t0, 0) * FS), int(min(t1 * FS, n))
        if b - a >= int(MIN_SECONDS.get(activity, 1.0) * FS):
            windows.append(Window("sim", clip, activity, angles[a:b], pos[a:b], acc[a:b], height,
                                  {"imu_band_hz": 44.0, **info}))

    for call in rec.calls:
        parents = rec.ancestors(call)
        if call.name == "walk":
            parent = rec.calls[call.parent] if call.parent is not None else None
            load = parent.args.get("load_kg") if parent is not None and parent.name == "carry" else None
            activity = "carry_walk" if load else "walk"
            cut(call.t0 + 1.0, call.t1 - 1.0, activity, load_kg=load)
        elif call.name == "run":
            cut(call.t0 + 1.0, call.t1 - 1.0, "run")
        elif call.name in ("lift", "put_down"):
            event = next((e for e in lift_events if call.t0 <= e[0] <= call.t1), None)
            info = event[2] if event else {}
            cut(call.t0 - 0.5, call.t1 + 0.5, "lift", technique=info.get("technique"), kind=call.name,
                load_kg=info.get("load_kg"), knee_deg=info.get("knee_deg"))
        elif call.name == "kneel_work":
            cut(call.t0 + 2.5, call.t1 - 1.5, "kneel")
        elif call.name == "ladder" and "cab_ride" not in parents and call.args.get("height", 0) >= 1.0:
            cut(call.t0 + 0.8, call.t1, "ladder", up=call.args.get("up"))
        elif call.name == "stand" and not parents:
            cut(call.t0 + 0.5, call.t1 - 0.5, "idle")
    return windows


def _sim_task(args) -> list[dict]:
    scenario, seed, obs_s = args
    rows = []
    for w in sim_windows(scenario, seed, obs_s):
        rows.extend(window_metrics(w))
    return rows


# --- source (b): canonical mocap clips -------------------------------------------------------------------
def load_mocap_clip(path: Path) -> dict:
    """A canonical clip plus its data card (motionlib/cards/<clip>.json)."""
    data = dict(np.load(path, allow_pickle=False))
    card_path = path.parents[1] / "cards" / (path.stem + ".json")
    card = json.loads(card_path.read_text()) if card_path.exists() else {}
    data["card"] = card
    return data


def mocap_imu(pos: np.ndarray, angles: np.ndarray, source_fps: float) -> tuple[np.ndarray, float]:
    """Pelvis specific force from the S1 anchor path, in the neutral-calibrated pelvis segment frame
    (+Y = gravity in the neutral stand, as on the vest). Returns (acc, bandwidth Hz): mocap is filtered
    and sampled at source_fps, so nothing above ~0.4 x source_fps is real."""
    band = min(0.4 * float(source_fps), 20.0)
    smooth = lowpass(pos, band)
    accel = np.gradient(np.gradient(smooth, 1.0 / FS, axis=0), 1.0 / FS, axis=0)
    specific = accel + np.array([0.0, GRAVITY, 0.0])
    pelvis = segment_rotations(angles)[0]
    return pelvis.inv().apply(specific), band


def mocap_windows(path: Path) -> list[Window]:
    """Cut one canonical clip into activity windows using its tags and its own motion."""
    clip = load_mocap_clip(path)
    card = clip["card"]
    name = path.stem
    tags = set(card.get("tags", []))
    angles = clip["trunk_angles"].astype(float)
    names = [str(x) for x in clip["joint_names"]]
    floor = not card.get("root_relative", False) and card.get("source") not in ("niosh", "cml")
    pos = clip["trunk_anchors"][:, 0].astype(float) if floor else None
    height = float(clip["height_est_m"]) if floor else None
    acc, band = (mocap_imu(pos, angles, float(clip["source_fps"])) if pos is not None else (None, 0.0))
    base = {"imu_band_hz": band, "neutral": card.get("trunk", {}).get("neutral_method"), "floor": floor}
    joints = clip["joints"].astype(float)
    windows = []

    def window(a: int, b: int, activity: str, **info) -> Window:
        return Window("mocap", name, activity, angles[a:b], None if pos is None else pos[a:b],
                      None if acc is None else acc[a:b], height, {**base, **info})

    if pos is not None and tags & {"walk", "run", "carry", "load"}:
        speed = horizontal_speed(pos)
        if "run" in tags:
            spans, activity = runs_where(speed > 2.0, int(MIN_SECONDS["run"] * FS)), "run"
        else:
            carrying = bool(tags & {"carry", "load"})
            activity = "carry_walk" if carrying else "walk"
            spans = runs_where((speed > 0.6) & (speed < 2.0), int(MIN_SECONDS[activity] * FS))
        for a, b in spans:
            trim = int((0.1 if activity == "run" else 0.2) * FS)
            if b - a - 2 * trim >= int(MIN_SECONDS[activity] * FS):
                windows.append(window(a + trim, b - trim, activity))
    if tags & {"pickup", "putdown", "stoop", "squat", "lift"}:
        knee = np.maximum(knee_angle_deg(joints[:, names.index("hip_l")], joints[:, names.index("knee_l")],
                                         joints[:, names.index("ankle_l")]),
                          knee_angle_deg(joints[:, names.index("hip_r")], joints[:, names.index("knee_r")],
                                         joints[:, names.index("ankle_r")]))
        technique = "squat" if "squat" in tags else ("stoop" if "stoop" in tags else None)
        windows.append(Window("mocap", name, "lift", angles, pos, acc, height,
                              {**base, "knee_series": knee, "technique": technique}))
    if "kneel_work" in tags or ("kneel" in tags and pos is not None):
        if pos is None:
            windows.append(window(0, len(angles), "kneel"))
        else:
            low = pos[:, 1] < 0.45 * float(clip["height_est_m"])
            for a, b in runs_where(low, int(MIN_SECONDS["kneel"] * FS)):
                windows.append(window(a, b, "kneel"))
    if "ladder" in tags and pos is not None:
        vy = np.gradient(lowpass(pos[:, 1], 0.5), 1.0 / FS)
        for a, b in runs_where(np.abs(vy) > 0.08, int(MIN_SECONDS["ladder"] * FS)):
            windows.append(window(a, b, "ladder", up=bool(np.mean(vy[a:b]) > 0)))
    if "idle" in tags and pos is not None:
        still = horizontal_speed(pos) < 0.05
        for a, b in runs_where(still, int(MIN_SECONDS["idle"] * FS)):
            windows.append(window(a, b, "idle"))
    return windows


# --- source (c): the team's vest recordings -------------------------------------------------------------
def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _device_times(rows: list[dict]) -> np.ndarray:
    """sample_time_ms -> seconds, unwrapping the uint32 millis() counter."""
    ms = np.array([r["sample_time_ms"] for r in rows], float)
    wraps = np.cumsum(np.concatenate([[0], np.diff(ms) < -2**31]))
    return (ms + wraps * 2**32) / 1000.0


def _ordered(rows: list[dict]) -> list[dict]:
    """Drop repeated and backwards device times (as the backend reader does)."""
    if not rows:
        return rows
    t = _device_times(rows)
    keep, last = [], -np.inf
    for row, ti in zip(rows, t):
        if ti > last:
            keep.append(row)
            last = ti
    return keep


def imu_by_id(row: dict, sensor_id: int) -> dict | None:
    for imu in row.get("imus", []):
        if imu.get("sensor_id") == sensor_id:
            return imu
    return None


def load_vest_session(folder: Path) -> dict:
    """One recorded session: raw.jsonl and/or quat.jsonl (backend `app.cli stream --record`), labels.csv
    and meta.json. Returns t (s from the first sample), angles (N,12) or None, acc (N,3) pelvis or None,
    labels and meta, all on a uniform FS grid."""
    meta_path = folder / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    raw_rows = _ordered(read_jsonl(folder / "raw.jsonl")) if (folder / "raw.jsonl").exists() else []
    quat_rows = _ordered(read_jsonl(folder / "quat.jsonl")) if (folder / "quat.jsonl").exists() else []
    if not raw_rows and not quat_rows:
        raise ValueError(f"{folder}: no raw.jsonl or quat.jsonl")
    starts = [_device_times(r)[0] for r in (raw_rows, quat_rows) if r]
    ends = [_device_times(r)[-1] for r in (raw_rows, quat_rows) if r]
    t0, t1 = max(starts), min(ends)
    grid = np.arange(0.0, t1 - t0, 1.0 / FS)
    labels = read_labels(folder / "labels.csv")
    neutral = next(((a, b) for a, b, act, _ in labels if act == "neutral"), (0.0, CALIBRATION_S))
    in_neutral = (grid >= neutral[0]) & (grid <= neutral[1])
    acc = None
    if raw_rows:
        t_raw = _device_times(raw_rows) - t0
        pelvis = np.array([imu_by_id(r, 1)["accel"] if imu_by_id(r, 1) and imu_by_id(r, 1).get("accel") else
                           [np.nan] * 3 for r in raw_rows], float)
        good = np.all(np.isfinite(pelvis), axis=1) & (np.linalg.norm(np.nan_to_num(pelvis), axis=1) > 0)
        acc = np.stack([np.interp(grid, t_raw[good], pelvis[good, i]) for i in range(3)], axis=1)
        acc = neutral_alignment(acc[in_neutral].mean(axis=0)).apply(acc)
    angles = None
    if quat_rows:
        t_q = _device_times(quat_rows) - t0
        segments = []
        for sensor_id in (1, 2, 3, 4):
            wxyz = np.array([imu_by_id(r, sensor_id)["quaternion"] for r in quat_rows], float)
            rot = Slerp(t_q, Rotation.from_quat(wxyz[:, [1, 2, 3, 0]]))(np.clip(grid, t_q[0], t_q[-1]))
            segments.append(rot)
        angles = vest_angles(segments, in_neutral)
    return {"session": folder.name, "t": grid, "angles": angles, "acc": acc, "labels": labels, "meta": meta}


def vest_angles(segments: list[Rotation], in_neutral: np.ndarray) -> np.ndarray:
    """Neutral-calibrate the four fused orientations (R' = R R_neutral^-1 Ry(h0), h0 the pelvis heading
    in the neutral stand, so every segment reads Ry(h0) there, as in the simulator) and decompose them
    with the inverse of spine.root_rotation / joint_rotation."""
    neutral = [seg[in_neutral].mean() for seg in segments]
    h0 = float(neutral[0].as_euler("YXZ")[0])
    ry = Rotation.from_rotvec([0.0, h0, 0.0])
    cal = [seg * mean.inv() * ry for seg, mean in zip(segments, neutral)]
    root = cal[0].as_euler("YXZ")
    out = np.zeros((len(root), 12))
    out[:, 0], out[:, 1], out[:, 2] = root[:, 0], -root[:, 1], -root[:, 2]
    for j in range(3):
        rel = (cal[j].inv() * cal[j + 1]).as_euler("XZY")
        out[:, 3 + j], out[:, 6 + j], out[:, 9 + j] = -rel[:, 0], -rel[:, 1], rel[:, 2]
    return out


def read_labels(path: Path) -> list[tuple[float, float, str, dict]]:
    """labels.csv: start_s,end_s,activity[,load_kg,technique,notes]; times from the first sample."""
    if not path.exists():
        return []
    labels = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            extra = {k: v for k, v in row.items() if k not in ("start_s", "end_s", "activity") and v not in (None, "")}
            if "load_kg" in extra:
                extra["load_kg"] = float(extra["load_kg"])
            labels.append((float(row["start_s"]), float(row["end_s"]), row["activity"].strip(), extra))
    return labels


def vest_windows(folder: Path) -> list[Window]:
    session = load_vest_session(folder)
    angles, acc = session["angles"], session["acc"]
    if angles is None:
        angles = np.zeros((len(session["t"]), 12))   # raw only: angle metrics are not meaningful
    height = session["meta"].get("height_m")
    windows = []
    for a, b, activity, extra in session["labels"]:
        if activity not in ACTIVITIES:
            continue
        i, j = int(a * FS), int(min(b * FS, len(session["t"])))
        if j - i < int(MIN_SECONDS.get(activity, 1.0) * FS):
            continue
        info = {"imu_band_hz": 44.0, "has_angles": session["angles"] is not None, **extra}
        windows.append(Window("vest", session["session"], activity, angles[i:j], None,
                              None if acc is None else acc[i:j], height, info))
    return windows


def recording_sessions(root: Path, include_synthetic: bool = False) -> list[Path]:
    if not root.exists():
        return []
    sessions = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (folder / "labels.csv").exists():
            continue
        meta = json.loads((folder / "meta.json").read_text()) if (folder / "meta.json").exists() else {}
        if meta.get("synthetic") and not include_synthetic:
            continue
        sessions.append(folder)
    return sessions


def write_example_session(folder: Path, scenario: str = "A-CARRY", seed: int = 0) -> Path:
    """Write a simulator run in the vest recording format (raw.jsonl, quat.jsonl, labels.csv, meta.json),
    marked synthetic. It exercises the loader end to end; it is never used as a reference."""
    from .viz.recorder import RecordingBuilder

    sink: list = []
    run = build_run(scenario, seed, obs_s=60.0, builder=lambda w, s, r: RecordingBuilder(w, s, r, sink=sink))
    track = run.track
    mounts = random_mounts(run.worker, np.random.default_rng(seed + 5))
    kin = strip_kinematics(track, run.worker, mounts)
    acc, gyro = virtual_imu(kin, track.fs)
    acc, gyro = apply_soft_tissue(acc, gyro, track.fs, run.soft_fn, run.soft_zeta)
    strip = ImuStrip(PROFILES["target"], 4, DeviceErrors(), seed=seed + 1)
    samples = strip.process(np.arange(len(track)) / track.fs, acc, gyro, strip.calibration_temp_c)
    folder.mkdir(parents=True, exist_ok=True)
    index = np.clip(np.round(samples.t_true * track.fs).astype(int), 0, len(track) - 1)
    with open(folder / "raw.jsonl", "w") as raw, open(folder / "quat.jsonl", "w") as quat:
        for k in range(len(index)):
            ms = int(samples.time_ms[k])
            imus = [{"sensor_id": s + 1, "accel": samples.acc[k, s].round(4).tolist(),
                     "gyro": samples.gyro[k, s].round(5).tolist(), "quaternion": None, "valid": True} for s in range(4)]
            raw.write(json.dumps({"format": "raw", "sample_time_ms": ms, "imus": imus, "sequence": k}) + "\n")
            quats = []
            for s in range(4):
                x, y, z, w = kin.chip_rot[s][int(index[k])].as_quat()
                quats.append({"sensor_id": s + 1, "accel": None, "gyro": None,
                              "quaternion": [round(float(v), 6) for v in (w, x, y, z)], "valid": True})
            quat.write(json.dumps({"format": "quaternion", "sample_time_ms": ms, "imus": quats, "sequence": k}) + "\n")
    t_first = float(samples.t_true[0])
    rows = [(0.0, CALIBRATION_S, "neutral", {})]
    rec = sink[0]
    for call in rec.calls:
        if call.name == "walk" and not rec.ancestors(call):
            rows.append((call.t0 + 1.0 - t_first, call.t1 - 1.0 - t_first, "walk", {}))
        elif call.name in ("lift", "put_down"):
            rows.append((call.t0 - 0.5 - t_first, call.t1 + 0.5 - t_first, "lift", {"notes": call.name}))
        elif call.name == "stand" and not rec.ancestors(call):
            rows.append((call.t0 + 0.5 - t_first, call.t1 - 0.5 - t_first, "idle", {}))
    with open(folder / "labels.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["start_s", "end_s", "activity", "load_kg", "technique", "notes"])
        for a, b, activity, extra in rows:
            if b > a:
                writer.writerow([f"{a:.2f}", f"{b:.2f}", activity, "", "", extra.get("notes", "")])
    meta = {"synthetic": True, "made_by": f"thundersim motionval --write-example ({scenario} seed {seed})",
            "consent_id": None, "height_m": round(run.worker.height_m, 3), "profile": "target"}
    (folder / "meta.json").write_text(json.dumps(meta, indent=1))
    return folder


# --- aggregation and report ---------------------------------------------------------------------------
def collect(scenarios=DEFAULT_SCENARIOS, seeds: int = 10, seed_base: int = 0, obs_s: float = 60.0,
            mocap_dir: Path = MOCAP_DIR, recordings: Path = RECORDINGS_DIR, include_synthetic: bool = False,
            jobs: int = 4) -> list[dict]:
    """Metric rows for every window of every source."""
    tasks = [(s, seed_base + i, obs_s) for s in scenarios for i in range(seeds)]
    rows: list[dict] = []
    if tasks:
        with ProcessPoolExecutor(jobs) as pool:
            for part in pool.map(_sim_task, tasks):
                rows.extend(part)
    for path in sorted(Path(mocap_dir).glob("*.npz")) if Path(mocap_dir).exists() else []:
        for w in mocap_windows(path):
            rows.extend(window_metrics(w))
    for folder in recording_sessions(Path(recordings), include_synthetic):
        for w in vest_windows(folder):
            for row in window_metrics(w):
                if not w.info.get("has_angles", True):
                    row = {k: v for k, v in row.items() if k.startswith("_") or k.startswith("imu") or k == "step_hz"}
                rows.append(row)
    return rows


METRICS = {
    "walk": ("speed_mps", "step_hz", "step_length_m", "bob_cm", "ml_cm", "pelvis_yaw_pp_deg", "pelvis_roll_pp_deg",
             "pelvis_pitch_pp_deg", "thorax_pelvis_yaw_pp_deg", "pelvis_pitch_mean_deg", "upper_incl_mean_deg",
             "trunk_rate_p95_rads", "imu_step_peak_g", "imu_step_peak_10hz_g", "imu_vert_rms_g", "imu_peak_norm_g",
             "imu_share_above_5hz", "imu_share_above_10hz"),
    "lift": ("peak_pelvis_deg", "peak_lumbar_deg", "peak_thoracic_deg", "peak_upper_deg", "pelvis_at_peak_deg",
             "pelvis_share", "spine_flex_at_peak_deg", "knee_deg", "pelvis_drop_m", "pelvis_drop_per_height",
             "peak_rate_thoracic_dps", "peak_accel_thoracic_dps2", "lower_s", "hold_s", "rise_s"),
    "kneel": ("upper_incl_mean_deg", "upper_incl_p10_deg", "upper_incl_p90_deg", "pelvis_pitch_mean_deg",
              "pelvis_share", "upper_incl_rate_p95_dps", "pelvis_height_per_height"),
    "ladder": ("rung_hz", "climb_speed_mps", "rise_per_rung_m", "pitch_mean_deg", "upper_incl_mean_deg",
               "roll_pp_deg", "roll_to_rung_freq", "sway_pp_cm"),
    "idle": ("sway_rms_mm", "pitch_rms_deg", "roll_rms_deg", "spine_flex_rms_deg"),
}
METRICS["carry_walk"] = METRICS["walk"]
METRICS["run"] = METRICS["walk"]


def per_clip_values(rows: list[dict], activity: str, source: str, metric: str) -> np.ndarray:
    """One value per clip (the median of its windows), so a long clip with many windows does not
    outweigh short ones. Lifts keep one value per bend."""
    by_clip: dict[str, list[float]] = {}
    for r in rows:
        value = r.get(metric)
        if r["_activity"] == activity and r["_source"] == source and value is not None and np.isfinite(value):
            key = f"{r['_clip']}#{len(by_clip)}" if activity == "lift" else r["_clip"]
            by_clip.setdefault(key, []).append(float(value))
    return np.array([np.median(v) for v in by_clip.values()], float)


def summarise(rows: list[dict]) -> dict:
    """(activity, metric) -> {source: {n, median, p10, p90}} plus the literature range and a verdict."""
    table = {}
    for activity in ACTIVITIES:
        for metric in METRICS[activity]:
            entry = {}
            for source in ("sim", "mocap", "vest"):
                values = per_clip_values(rows, activity, source, metric)
                if len(values):
                    entry[source] = {"n": int(len(values)), "median": float(np.median(values)),
                                     "p10": float(np.percentile(values, 10)), "p90": float(np.percentile(values, 90))}
            lit = LITERATURE.get((activity, metric))
            if lit:
                entry["lit"] = {"low": lit[0], "high": lit[1], "source": lit[2]}
            entry["verdict"] = "not judged" if (activity, metric) in REFERENCE_CAVEATS else verdict(entry)
            if len(entry) > 1:
                table[(activity, metric)] = entry
    return table


def verdict(entry: dict) -> str:
    """'ok', 'low' or 'high': the simulator median against the real range (vest and mocap 10-90 %
    spread, widened to their min-max for n < 5, else the literature range); '-' if nothing to compare."""
    sim = entry.get("sim")
    if sim is None:
        return "-"
    for source in ("vest", "mocap"):
        ref = entry.get(source)
        if ref:
            low, high = ref["p10"], ref["p90"]
            break
    else:
        if "lit" not in entry:
            return "-"
        low, high = entry["lit"]["low"], entry["lit"]["high"]
    pad = 0.05 * max(abs(low), abs(high), 1e-6)
    if sim["median"] < low - pad:
        return "low"
    if sim["median"] > high + pad:
        return "high"
    return "ok"


def _fmt(stats: dict | None) -> str:
    if not stats:
        return "-"
    return f"{stats['median']:.3g} ({stats['p10']:.3g}-{stats['p90']:.3g}) n={stats['n']}"


def write_report(rows: list[dict], out: Path, notes: list[str]) -> Path:
    """MOTION_VALIDATION.md (tables), metrics.json and figures/ in out."""
    out.mkdir(parents=True, exist_ok=True)
    table = summarise(rows)
    figures = plot_figures(rows, out / "figures")
    lines = ["# ThunderSim motion validation", "",
             "Simulated trunk motion against real human motion, with the same metrics for every source. "
             "Values are median (10th-90th percentile) over clips (simulator: runs), each clip the median of its "
             "windows; n = clips (lifts: bends). "
             "Verdict compares the simulator median with the team's vest recordings if any, else the mocap "
             "spread, else the literature range. Simulated and recorded data only: no medical claim.", ""]
    lines += [f"- {n}" for n in notes] + [""]
    counts = {}
    for r in rows:
        counts.setdefault(r["_activity"], {}).setdefault(r["_source"], set()).add(r["_clip"])
    lines += ["## Sources per activity", "", "| Activity | Simulator runs | Mocap clips | Vest sessions |", "|---|---|---|---|"]
    for activity in ACTIVITIES:
        c = counts.get(activity, {})
        lines.append(f"| {activity} | {len(c.get('sim', ()))} | {len(c.get('mocap', ()))} | {len(c.get('vest', ()))} |")
    lines.append("")
    for activity in ACTIVITIES:
        keys = [k for k in table if k[0] == activity]
        if not keys:
            continue
        lines += [f"## {activity}", "", "| Metric | Simulator | Mocap | Vest | Literature | Verdict |", "|---|---|---|---|---|---|"]
        for key in keys:
            e = table[key]
            lit = f"{e['lit']['low']:.3g}-{e['lit']['high']:.3g} [L]" if "lit" in e else "-"
            lines.append(f"| {key[1]} | {_fmt(e.get('sim'))} | {_fmt(e.get('mocap'))} | {_fmt(e.get('vest'))} | "
                         f"{lit} | {e['verdict']} |")
        caveats = [f"- {k[1]}: {REFERENCE_CAVEATS[k]}." for k in keys if k in REFERENCE_CAVEATS]
        if caveats:
            lines += ["", "Not judged:"] + caveats
        if activity in figures:
            lines += [""] + [f"![{activity}]({p.relative_to(out)})" for p in figures[activity]]
        lines.append("")
    lines += ["## Literature [L]", ""]
    for (activity, metric), (low, high, source) in LITERATURE.items():
        lines.append(f"- {activity} / {metric}: {low:g}-{high:g}. {source}")
    lines += ["", "## Metric definitions", "",
              "- Gait windows are steady stretches (simulator: the walk minus 1 s at each end; mocap: 0.6-2.0 m/s, "
              "> 2 m/s for running). step_hz is the spectral peak of the pelvis height (of the IMU vertical "
              "force without a floor). bob_cm, ml_cm and the *_pp_deg values are the median peak-to-peak per step "
              "(bob) or per stride (the rest), after removing the slow path and turning (0.4 Hz).",
              "- imu_*: pelvis specific force on the vest's axes after neutral alignment (+Y = gravity in the neutral "
              "stand). Simulator: the target-profile MPU-6050 emulator (44 Hz DLPF, 100 Hz). Mocap: the S1 path "
              "differentiated twice, limited to 0.4 x its frame rate (ACCAD 12 Hz, 100STYLE 20 Hz, CMU 20 Hz), so "
              "its step peaks are low bounds and its high-frequency shares are only listed above 20 Hz bandwidth.",
              "- Lifts are bends of the upper-thoracic segment past 30 deg. Inclinations are each segment's long "
              "axis from vertical in the heading plane, after a 6 Hz low-pass (all sources). pelvis_share = pelvis "
              "inclination / upper-thoracic inclination at the deepest point (lumbo-pelvic ratio). lower/hold/rise "
              "use 10 % and 90 % of the deepest upper-thoracic inclination. Simulator knee angles are the planned "
              "lowest posture; mocap knee angles come from the hip, knee and ankle joints.",
              "- Ladder: rung_hz is the spectral peak of |vertical pelvis speed|. roll_to_rung_freq ~ 0.5 means the "
              "pelvis rolls to alternate sides on successive rungs; ~ 1 means the same side every rung.",
              "- Idle: quiet standing at least 3 s; sway and angles after removing drift below 0.05 Hz.",
              "- Mocap trunk angles are relative to each performer's neutral stand; clips without one use upright "
              "walking frames or kneeling (see the cards), which biases their mean lean."]
    path = out / "MOTION_VALIDATION.md"
    path.write_text("\n".join(lines) + "\n")
    plain = [{k: v for k, v in r.items() if not k.startswith("_series")} for r in rows]
    summary = {f"{a}/{m}": e for (a, m), e in table.items()}
    (out / "metrics.json").write_text(json.dumps({"summary": summary, "rows": plain}, indent=1, default=float))
    return path


# --- figures ----------------------------------------------------------------------------------------------
COLOURS = {"sim": "#2a78d6", "mocap": "#eb6834", "vest": "#1baf7a"}   # dataviz reference palette, slots 1-3
MARKERS = {"sim": "o", "mocap": "s", "vest": "^"}
FIGURE_METRICS = {
    "walk": ("step_length_m", "bob_cm", "ml_cm", "pelvis_yaw_pp_deg", "pelvis_roll_pp_deg", "pelvis_pitch_pp_deg",
             "thorax_pelvis_yaw_pp_deg", "upper_incl_mean_deg"),
    "carry_walk": ("step_hz", "bob_cm", "pelvis_pitch_mean_deg", "upper_incl_mean_deg"),
    "run": ("step_hz", "bob_cm", "pelvis_roll_pp_deg", "trunk_rate_p95_rads"),
    "kneel": ("upper_incl_mean_deg", "pelvis_share", "upper_incl_rate_p95_dps", "pelvis_height_per_height"),
    "ladder": ("rung_hz", "pitch_mean_deg", "roll_pp_deg", "roll_to_rung_freq"),
    "idle": ("sway_rms_mm", "pitch_rms_deg", "roll_rms_deg", "spine_flex_rms_deg"),
}


def plot_figures(rows: list[dict], folder: Path) -> dict:
    """Strip plots (each window one dot, per source, literature as a grey band), lift overlays and the lift
    fingerprint. Returns {activity: [paths]}."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    made: dict[str, list[Path]] = {}
    rng = np.random.default_rng(0)
    for activity, metrics in FIGURE_METRICS.items():
        sel = [r for r in rows if r["_activity"] == activity]
        if not sel:
            continue
        fig, axes = plt.subplots(1, len(metrics), figsize=(1.9 * len(metrics), 2.6))
        for ax, metric in zip(np.atleast_1d(axes), metrics):
            lit = LITERATURE.get((activity, metric))
            if lit:
                ax.axhspan(lit[0], lit[1], color="#d9d8d3", zorder=0, label="literature")
            for i, source in enumerate(("sim", "mocap", "vest")):
                v = per_clip_values(rows, activity, source, metric)
                if len(v):
                    x = i + rng.uniform(-0.18, 0.18, len(v))
                    ax.scatter(x, v, s=12, color=COLOURS[source], marker=MARKERS[source], alpha=0.75,
                               edgecolors="#fcfcfb", linewidths=0.5, label=source)
                    ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color="#0b0b0b", lw=1.5)
            ax.set_xticks([0, 1, 2], ["sim", "mocap", "vest"])
            ax.set_title(metric, fontsize=7.5, color="#0b0b0b")
            ax.grid(axis="y", color="#e6e5e0", lw=0.6)
        handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
        fig.legend(*_unique(handles, labels), loc="upper right", fontsize=7, frameon=False)
        fig.suptitle(f"{activity}: one dot per clip or simulated run, bar = median", fontsize=9, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 0.93, 0.93))
        path = folder / f"{activity}_metrics.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        made.setdefault(activity, []).append(path)
    lifts = [r for r in rows if r["_activity"] == "lift"]
    if lifts:
        made["lift"] = [_plot_lift_overlay(lifts, folder, plt), _plot_fingerprint(lifts, folder, plt)]
        sel = lifts
        metrics = ("peak_thoracic_deg", "pelvis_share", "knee_deg", "peak_rate_thoracic_dps", "lower_s", "rise_s")
        fig, axes = plt.subplots(1, len(metrics), figsize=(1.9 * len(metrics), 2.6))
        for ax, metric in zip(axes, metrics):
            lit = LITERATURE.get(("lift", metric))
            if lit:
                ax.axhspan(lit[0], lit[1], color="#d9d8d3", zorder=0, label="literature")
            for i, source in enumerate(("sim", "mocap", "vest")):
                v = [r[metric] for r in sel if r["_source"] == source and metric in r and np.isfinite(r[metric])]
                if v:
                    ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=12, color=COLOURS[source],
                               marker=MARKERS[source], alpha=0.75, edgecolors="#fcfcfb", linewidths=0.5, label=source)
                    ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color="#0b0b0b", lw=1.5)
            ax.set_xticks([0, 1, 2], ["sim", "mocap", "vest"])
            ax.set_title(metric, fontsize=7.5)
            ax.grid(axis="y", color="#e6e5e0", lw=0.6)
        fig.legend(*_unique(*axes[0].get_legend_handles_labels()), loc="upper right", fontsize=7, frameon=False)
        fig.suptitle("lift: one dot per bend, bar = median", fontsize=9, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 0.93, 0.93))
        path = folder / "lift_metrics.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        made["lift"].append(path)
    return made


def _unique(handles, labels):
    seen = {}
    for h, label in zip(handles, labels):
        seen.setdefault(label, h)
    return list(seen.values()), list(seen.keys())


def _plot_lift_overlay(lifts: list[dict], folder: Path, plt) -> Path:
    """Thoracic inclination through each bend, time-normalised from start to end of the bend."""
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    for source in ("sim", "mocap", "vest"):
        curves = [r["_series"] for r in lifts if r["_source"] == source and len(r.get("_series", [])) > 10]
        grid = np.linspace(0, 1, 200)
        resampled = [np.interp(grid, np.linspace(0, 1, len(c)), c) for c in curves]
        for c in resampled[:40]:
            ax.plot(grid, c, color=COLOURS[source], lw=0.6, alpha=0.25)
        if resampled:
            ax.plot(grid, np.median(resampled, axis=0), color=COLOURS[source], lw=2, label=f"{source} median (n={len(curves)})")
    ax.set_xlabel("bend, normalised time")
    ax.set_ylabel("thoracic inclination (deg)")
    ax.grid(color="#e6e5e0", lw=0.6)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    path = folder / "lift_overlay.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _plot_fingerprint(lifts: list[dict], folder: Path, plt) -> Path:
    """Lift technique fingerprint: pelvis drop (fraction of height) against the deepest trunk inclination."""
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    shapes = {"stoop": "v", "semi_squat": "o", "squat": "^", None: "s"}
    for source in ("sim", "mocap"):
        for technique, marker in shapes.items():
            sel = [r for r in lifts if r["_source"] == source and r.get("_technique") == technique
                   and np.isfinite(r.get("pelvis_drop_per_height", np.nan))]
            if sel:
                ax.scatter([r["peak_upper_deg"] for r in sel], [100 * r["pelvis_drop_per_height"] for r in sel], s=16,
                           marker=marker, color=COLOURS[source], alpha=0.75, edgecolors="#fcfcfb", linewidths=0.5,
                           label=f"{source} {technique or 'untagged'}")
    ax.set_xlabel("deepest upper-thoracic inclination (deg)")
    ax.set_ylabel("pelvis drop (% of height)")
    ax.grid(color="#e6e5e0", lw=0.6)
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    fig.tight_layout()
    path = folder / "lift_fingerprint.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def run_validation(out: Path, scenarios=DEFAULT_SCENARIOS, seeds: int = 10, seed_base: int = 0, obs_s: float = 60.0,
                   recordings: Path = RECORDINGS_DIR, include_synthetic: bool = False, jobs: int = 4) -> Path:
    started = time.time()
    rows = collect(scenarios, seeds, seed_base, obs_s, MOCAP_DIR, recordings, include_synthetic, jobs)
    sessions = recording_sessions(Path(recordings), include_synthetic)
    notes = [f"Simulator: {', '.join(scenarios)}, seeds {seed_base}-{seed_base + seeds - 1}, observation {obs_s:.0f} s, "
             f"target profile.",
             f"Mocap: {len(list(MOCAP_DIR.glob('*.npz')))} canonical clips in motionlib/canonical (class A).",
             f"Vest recordings: {len(sessions)} session(s) in {recordings}"
             + ("" if sessions else " (none yet; see validation/RECORDING_PROTOCOL.md)") + ".",
             f"Built in {time.time() - started:.0f} s."]
    return write_report(rows, out, notes)
