# File: baseline.py
# Purpose: Reference rule-based detector from the research report (Stage-1 trigger, Stage-2
#          confirmation, man-down timers). It is the simulator's baseline and regression
#          anchor while learned models are developed; every threshold is a tunable default.
# Contributors: Unassigned
"""Baseline rules ("Fall and stroke detection models" report, recommended design).

Stage 1 (would run on the ESP32): on the pelvis/lumbar anchors, an impact (>= impact_g or a
clipped sample) together with a free-fall dip (< free_fall_g for >= free_fall_min_s) or fast
trunk rotation (>= rotation_rad_s, Bourke & Lyons 2008: 3.1 rad/s) within trigger_window_s.
A long free fall alone also triggers. A slow collapse (>= 3 anchors from upright to lying
within collapse_window_s) is a second trigger, off in work_mode="lying_work".

Stage 2 (laptop), confirm_window_s after the trigger:
- lying: median inclination > lying_deg at >= lying_slots anchors -> fall_confirmed;
- a free fall >= height_free_fall_s with no walking afterwards -> fall_confirmed (catches
  harness suspension, where the worker hangs near upright);
- otherwise near_fall if the trigger had fast rotation, else recovered.
Man-down: after a confirmed fall, continuous stillness for warning_s / critical_s raises
man_down_warning / man_down_critical; standing up again emits cleared.
All values are starting points to tune (thundersim tune), not validated thresholds.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from .model_api import GRAVITY, Event, FrameBatch, StreamContext


@dataclass
class BaselineConfig:
    trigger_slots: tuple[int, ...] = (0, 1)
    free_fall_g: float = 0.6
    free_fall_min_s: float = 0.10
    impact_g: float = 2.8
    rotation_rad_s: float = 3.1
    trigger_window_s: float = 1.0
    long_free_fall_s: float = 0.6        # ~1.8 m drop: confirm regardless of posture
    height_free_fall_s: float = 0.3      # ~0.45 m drop: confirm if no walking afterwards
    collapse_window_s: float = 10.0
    upright_deg: float = 30.0
    lying_deg: float = 60.0
    lying_slots: int = 3
    lying_pelvis_deg: float = 55.0       # the pelvis must tilt too: a deep stoop is not lying (rebar, kneeling)
    confirm_delay_s: float = 2.0
    confirm_window_s: float = 10.0
    still_window_s: float = 2.0
    still_acc_sd_g: float = 0.03
    still_gyro_rad_s: float = 0.08
    warning_s: float = 30.0
    critical_s: float = 60.0
    clear_upright_s: float = 3.0
    refractory_s: float = 3.0
    gravity_tau_s: float = 0.5
    # Simulator-informed additions (each fixes a failure the scenario suite exposed):
    near_dip_g: float = 0.85             # near-fall trigger: short dip (smeared by tissue + DLPF) ...
    near_dip_min_s: float = 0.03
    near_rotation_rad_s: float = 1.5     # ... plus moderate trunk rotation, no impact needed
    collapse_still_s: float = 5.0        # a collapse (no impact) is confirmed only if then still (A-LIE)
    worn_min_motion_g: float = 0.0015    # breathing/fidget motion; below this the strip is not worn (DEVICE-OFF)
    arrest_g: float = 2.0                # harness arrest: free fall, then a sustained lanyard pull ...
    arrest_min_s: float = 0.15           # ... of >= arrest_g for this long, then no walking (ARR)
    arrest_free_fall_s: float = 0.2


class BaselineRules:
    """Streaming implementation of the report's two-stage detector."""

    name = "baseline-rules"

    def __init__(self, config: BaselineConfig | None = None):
        self.config = config or BaselineConfig()

    def reset(self, context: StreamContext) -> None:
        self.ctx = context
        self.gravity = None                  # (4,3) low-passed specific force per slot
        self.last_t = None
        self.in_free_fall = False
        self.free_fall_start = 0.0
        self.free_fall_episodes: deque = deque(maxlen=20)  # (end_t, duration)
        self.last_rotation_t = -1e9
        self.last_near_rotation_t = -1e9
        self.in_dip = False
        self.dip_start = 0.0
        self.dip_episodes: deque = deque(maxlen=20)
        self.last_impact_t = -1e9
        self.frame_count = 0
        self.last_upright_t = 0.0
        self.candidate: dict | None = None
        self.last_candidate_t = -1e9
        self.alarm: dict | None = None       # active man-down tracking
        self.still_buf: deque = deque()      # (t, mean |a| g, mean gyro rad/s)
        self.gait_buf: deque = deque()       # (t, pelvis vertical specific force, g) for clearing
        self.upright_since: float | None = None

    # --- per-frame helpers -------------------------------------------------------------------
    def _inclinations(self) -> np.ndarray:
        g = self.gravity
        norm = np.linalg.norm(g, axis=1)
        cos = np.clip(g[:, 1] / np.maximum(norm, 1e-6), -1, 1)
        return np.degrees(np.arccos(cos))

    def _still(self, t: float) -> bool | None:
        c = self.config
        while self.still_buf and self.still_buf[0][0] < t - c.still_window_s:
            self.still_buf.popleft()
        if len(self.still_buf) < 5:
            return None  # not enough data to say (e.g. 0.5 Hz firmware)
        arr = np.array(self.still_buf)
        return bool(np.std(arr[:, 1]) < c.still_acc_sd_g and np.mean(arr[:, 2]) < c.still_gyro_rad_s)

    # --- streaming ---------------------------------------------------------------------------
    def update(self, batch: FrameBatch) -> list[Event]:
        c = self.config
        events: list[Event] = []
        acc_g = batch.acc / GRAVITY
        amag = np.linalg.norm(acc_g, axis=2)            # (N,4) g
        gmag = np.linalg.norm(batch.gyro, axis=2)       # (N,4) rad/s
        valid = batch.slot_mask & np.isfinite(amag)
        for i in range(len(batch)):
            t = float(batch.t[i])
            ok = valid[i]
            if not ok.any():
                continue
            dt = 0.01 if self.last_t is None else max(t - self.last_t, 1e-3)
            self.last_t = t
            frame = np.where(ok[:, None], batch.acc[i], np.nan)
            if self.gravity is None:
                self.gravity = np.nan_to_num(frame, nan=0.0) + np.array([0, GRAVITY, 0]) * (~ok)[:, None]
            alpha = 1 - np.exp(-dt / c.gravity_tau_s)
            upd = ok[:, None] & np.isfinite(frame)
            self.gravity = np.where(upd, self.gravity + alpha * (np.nan_to_num(frame) - self.gravity), self.gravity)
            incl = self._inclinations()
            incl_valid = incl[ok]
            trig = [s for s in c.trigger_slots if ok[s]] or list(np.flatnonzero(ok))
            a_trig = float(np.mean(amag[i, trig]))
            a_peak = float(np.max(amag[i, trig]))
            sat = bool(batch.sat_mask[i, trig].any())
            rot = float(np.max(gmag[i, ok]))
            self.still_buf.append((t, float(np.mean(amag[i, ok])), float(np.mean(gmag[i, ok]))))
            if self.alarm is not None:
                self.gait_buf.append((t, acc_g[i, 0, 1] if ok[0] else np.nan))
                while self.gait_buf and self.gait_buf[0][0] < t - 4.0:
                    self.gait_buf.popleft()

            # Stage 1 features.
            if a_trig < c.free_fall_g:
                if not self.in_free_fall:
                    self.in_free_fall, self.free_fall_start = True, t
            elif self.in_free_fall:
                self.in_free_fall = False
                duration = t - self.free_fall_start
                if duration >= c.free_fall_min_s:
                    self.free_fall_episodes.append((t, duration))
            if a_trig < c.near_dip_g:
                if not self.in_dip:
                    self.in_dip, self.dip_start = True, t
            elif self.in_dip:
                self.in_dip = False
                if t - self.dip_start >= c.near_dip_min_s:
                    self.dip_episodes.append((t, t - self.dip_start))
            if rot >= c.rotation_rad_s:
                self.last_rotation_t = t
            if rot >= c.near_rotation_rad_s:
                self.last_near_rotation_t = t
            if (np.sum(incl_valid < c.upright_deg) >= min(3, ok.sum())):
                self.last_upright_t = t

            recent_ff = [d for (end, d) in self.free_fall_episodes if t - end <= c.trigger_window_s]
            impact = a_peak >= c.impact_g or sat
            if impact:
                self.last_impact_t = t
            trigger = None
            if impact and (recent_ff or t - self.last_rotation_t <= c.trigger_window_s):
                trigger = "impact"
            if recent_ff and max(recent_ff) >= c.long_free_fall_s:
                trigger = "long_free_fall"
            pelvis_down = (not ok[0]) or incl[0] > c.lying_pelvis_deg
            lying_now = pelvis_down and np.sum(incl_valid > c.lying_deg) >= min(c.lying_slots, ok.sum())
            if (trigger is None and lying_now and batch.work_mode != "lying_work"
                    and t - self.last_upright_t <= c.collapse_window_s and self.alarm is None):
                trigger = "collapse"
            recent_dip = any(t - end <= 0.5 for (end, _) in self.dip_episodes)
            if trigger is None and recent_dip and t - self.last_near_rotation_t <= 0.5:
                trigger = "near_fall"
            if trigger and self.candidate is None and self.alarm is None and t - self.last_candidate_t >= c.refractory_s:
                ff = max([d for (end, d) in self.free_fall_episodes if t - end <= c.trigger_window_s + 0.5] or [0.0])
                self.candidate = {"t0": t, "kind": trigger, "free_fall_s": ff,
                                  "rotation": t - self.last_near_rotation_t <= c.trigger_window_s,
                                  "incl": [], "pelvis_y": [], "acc": [], "still_since": None, "still_max": 0.0,
                                  "high_since": None, "high_max": 0.0}
                self.last_candidate_t = t
                events.append(Event(t, "suspected_fall", 0.5, {"trigger": trigger, "free_fall_s": round(ff, 3)}))

            # Stage 2.
            cand = self.candidate
            if cand is not None:
                if t - cand["t0"] >= c.confirm_delay_s:
                    cand["incl"].append(np.where(ok, incl, np.nan))
                if t - cand["t0"] >= 0.5:
                    cand["pelvis_y"].append((t, acc_g[i, 0, 1] if ok[0] else np.nan))
                cand["acc"].append(np.where(ok[:, None], acc_g[i], np.nan))
                if t - cand["t0"] <= 2.0:  # sustained pull just after the trigger (lanyard arrest)
                    if a_trig >= c.arrest_g:
                        cand["high_since"] = cand["high_since"] or t
                        cand["high_max"] = max(cand["high_max"], t - cand["high_since"])
                    else:
                        cand["high_since"] = None
                self.frame_count += 1
                if self.frame_count % 10 == 0 and t - cand["t0"] >= c.confirm_delay_s:
                    still = self._still(t)
                    if still:
                        cand["still_since"] = cand["still_since"] or t
                        cand["still_max"] = max(cand["still_max"], t - cand["still_since"])
                    else:
                        cand["still_since"] = None
                # Late impact/free-fall information (e.g. landing after a collapse-type trigger).
                if recent_ff:
                    cand["free_fall_s"] = max(cand["free_fall_s"], max(recent_ff))
                if impact and cand["kind"] in ("collapse", "near_fall"):
                    cand["kind"] = "impact"
                if t - cand["t0"] >= c.confirm_window_s:
                    events.extend(self._decide(t, cand))
                    self.candidate = None

            # Man-down timers and clearing.
            if self.alarm is not None:
                events.extend(self._track_alarm(t, incl_valid))
        return events

    def _decide(self, t: float, cand: dict) -> list[Event]:
        c = self.config
        incl = np.array(cand["incl"]) if cand["incl"] else np.full((1, 4), np.nan)
        with np.errstate(all="ignore"):
            med = np.nanmedian(incl, axis=0)
        known = np.isfinite(med)
        pelvis_down = (not known[0]) or med[0] > c.lying_pelvis_deg
        lying = known.sum() > 0 and pelvis_down and np.sum(med[known] > c.lying_deg) >= min(c.lying_slots, known.sum())
        walking = self._walking(cand["pelvis_y"])
        worn = self._worn(cand["acc"])
        detail = {"trigger": cand["kind"], "median_incl_deg": [None if not np.isfinite(m) else round(float(m), 1) for m in med],
                  "free_fall_s": round(cand["free_fall_s"], 3), "walking": walking, "still_s": round(cand["still_max"], 1),
                  "worn": worn}
        if not worn:
            return [Event(t, "recovered", 0.7, {**detail, "rule": "not_worn"})]
        if lying and cand["kind"] == "collapse" and cand["still_max"] < c.collapse_still_s:
            return [Event(t, "recovered", 0.5, {**detail, "rule": "lying_but_moving"})]
        if lying:
            self.alarm = {"t0": cand["t0"], "still_since": None, "warned": False, "critical": False}
            return [Event(t, "fall_confirmed", 0.9, {**detail, "rule": "lying"})]
        if cand["kind"] == "collapse":
            return [Event(t, "recovered", 0.5, detail)]
        if cand["free_fall_s"] >= c.long_free_fall_s or (cand["free_fall_s"] >= c.height_free_fall_s and not walking):
            self.alarm = {"t0": cand["t0"], "still_since": None, "warned": False, "critical": False}
            return [Event(t, "fall_confirmed", 0.7, {**detail, "rule": "free_fall_no_gait"})]
        if cand["free_fall_s"] >= c.arrest_free_fall_s and cand["high_max"] >= c.arrest_min_s and not walking:
            self.alarm = {"t0": cand["t0"], "still_since": None, "warned": False, "critical": False}
            return [Event(t, "fall_confirmed", 0.7, {**detail, "rule": "arrest_pull", "pull_s": round(cand["high_max"], 2)})]
        if cand["rotation"]:
            return [Event(t, "near_fall", 0.6, detail)]
        return [Event(t, "recovered", 0.6, detail)]

    def _worn(self, frames: list) -> bool:
        """Worn strips show breathing/fidget motion; a strip on a table shows only sensor noise.

        Smooth each axis over 1 s (removes white noise, keeps 0.2-0.33 Hz breathing) and look at
        the largest remaining variation across slots. Unknown (too few frames) counts as worn.
        """
        c = self.config
        if len(frames) < 300:
            return True
        arr = np.array(frames)  # (N,4,3) g
        k = 100
        kernel = np.ones(k) / k
        spread = []
        for slot in range(arr.shape[1]):
            for axis in range(3):
                x = arr[:, slot, axis]
                x = x[np.isfinite(x)]
                if len(x) > 2 * k:
                    smooth = np.convolve(x - x.mean(), kernel, "valid")
                    spread.append(np.std(smooth[len(smooth) // 2:]))  # second half: after any landing
        return (max(spread) if spread else 1.0) >= c.worn_min_motion_g

    def _walking(self, series: list) -> bool:
        """Gait = periodic vertical pelvis acceleration at 1.2-2.6 Hz with real amplitude."""
        if len(series) < 50:
            return False
        arr = np.array(series)
        y = arr[:, 1]
        y = y[np.isfinite(y)]
        if len(y) < 50 or np.std(y) < 0.05:
            return False
        duration = arr[-1, 0] - arr[0, 0]
        fs = len(arr) / max(duration, 1e-3)
        y = y - y.mean()
        ac = np.correlate(y, y, "full")[len(y) - 1:]
        ac /= ac[0]
        lo, hi = int(fs / 2.6), int(fs / 1.2)
        if hi >= len(ac) or lo < 1:
            return False
        return bool(ac[lo:hi + 1].max() > 0.3)

    def _track_alarm(self, t: float, incl_valid: np.ndarray) -> list[Event]:
        c = self.config
        a = self.alarm
        out = []
        still = self._still(t)
        if still:
            a["still_since"] = a["still_since"] or t
        elif still is False:
            a["still_since"] = None
        if a["still_since"] is not None:
            held = t - a["still_since"]
            if held >= c.warning_s and not a["warned"]:
                a["warned"] = True
                out.append(Event(t, "man_down_warning", 0.8, {"still_s": round(held, 1)}))
            if held >= c.critical_s and not a["critical"]:
                a["critical"] = True
                out.append(Event(t, "man_down_critical", 0.9, {"still_s": round(held, 1)}))
        upright = np.sum(incl_valid < c.upright_deg) >= min(3, len(incl_valid))
        if upright:
            self.upright_since = self.upright_since or t
            # Clear only when upright AND walking: a harness-suspended worker hangs near upright.
            if t - self.upright_since >= c.clear_upright_s and self._walking(list(self.gait_buf)):
                out.append(Event(t, "cleared", 0.8, {}))
                self.alarm = None
                self.upright_since = None
        else:
            self.upright_since = None
        return out

    def describe(self) -> dict:
        return asdict(self.config)
