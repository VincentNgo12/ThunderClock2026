# File: main_model.py
# Purpose: MainModel (DETECTION_ARCHITECTURE.md 2-3): the single consumer of the sensor queue.
#          Runs falldet (T1/C1, T2/C2), PostureTrack, T3/C3, T4 and the arbiter, and emits
#          section-5 Events. The same object runs in ThunderSim (FallModel protocol) and the backend.
# Contributors: Unassigned
"""Usage
    model = MainModel()                       # thresholds from hub/config.json
    model.reset(context)                      # ThunderSim StreamContext, or None in the backend
    events = model.update(batch)              # FrameBatch (t, acc, gyro, slot_mask, sat_mask, work_mode)
    events = model.update_arrays(t, acc, gyro, slot_mask, sat_mask, work_mode)
    model.dismiss(episode_id)                 # supervisor dismissed that episode's alert
    model.status()                            # the "detector" block of section 5
    model.diagnostics                         # device diagnostics (section 5 "diagnostic")

Time: every event's t is device seconds of the current boot and never later than the batch's
last frame. Internally one clock continues across restarts (internal = device + offset), so the
posture ring, the last trigger and man-down arming survive a reboot.
Restart = device time goes back by more than restart_back_s: falldet is flushed and recreated,
the tilt filter and T3 timing are reset, T4 is armed if a trigger happened within
restart_arm_window_s, or if the worker is "down" in the first 10 s after the restart.
Gaps: > tilt_reset_gap_s reset only the tilt filter; >= long_gap_s also reset T3 timing.
falldet decides its own pending triggers after gaps > 5 s.
"""
import math

import numpy as np

from .arbiter import Arbiter
from .collapse import CollapseConfirmer
from .config import config_hash, load_config
from .events import diagnostic, make_event
from .falldet_path import FalldetPath, choose_profile
from .man_down import ManDownTimer
from .posture_track import PostureTrack
from .rate_watch import RateWatch
from .topple import ToppleTrigger

RESTART_DOWN_WATCH_S = 10.0
LINK_GAP_S = 0.3


def _context_dlpf(context):
    """DLPF of the ThunderSim device profile, when the simulator is importable."""
    try:
        from thundersim.device import PROFILES  # type: ignore
        profile = PROFILES.get(getattr(context, "profile", None))
        return None if profile is None else float(profile.dlpf_hz)
    except Exception:
        return None


class MainModel:
    name = "hub-collapse-v1"

    def __init__(self, config=None, overrides=None, profile=None, backend=None, name=None):
        self.cfg = config or load_config(overrides=overrides)
        self.forced_profile = profile
        self.backend = backend or self.cfg["falldet"]["backend"]
        if name:
            self.name = name
        self.config_hash = config_hash(self.cfg)
        self.reset(None)

    # ------------------------------------------------------------------ lifecycle
    def reset(self, context=None):
        cfg = self.cfg
        dlpf = _context_dlpf(context) if context is not None else None
        rng = getattr(context, "accel_range_g", None)
        self.model_profile, self.profile_reason = choose_profile(cfg["falldet"], dlpf, rng, self.forced_profile)
        self.falldet = FalldetPath(self.model_profile, self.backend)
        self.track = PostureTrack(cfg["posture"])
        self.topple = ToppleTrigger(cfg["topple"])
        self.collapse = CollapseConfirmer(cfg["collapse"])
        self.man_down = ManDownTimer(cfg["man_down"])
        self.arbiter = Arbiter(cfg["arbiter"])
        self.rate = RateWatch(cfg["rate"])
        self.offset = 0.0                 # internal = device + offset
        self.last_dev = None
        self.last_int = None
        self.boot_first_dev = None
        self.latched_since = None
        self.restart_watch_until = None
        self.last_trigger_int = -1e18
        self.lying_since = None
        self.counters = {"t3_fires": 0, "restarts": 0, "gaps": 0, "rate_recoveries": 0, "frames": 0,
                         "duplicates": 0}
        self.diagnostics = []
        self.work_mode = "normal"

    def update(self, batch):
        return self.update_arrays(batch.t, batch.acc, batch.gyro, getattr(batch, "slot_mask", None),
                                  getattr(batch, "sat_mask", None), getattr(batch, "work_mode", "normal"))

    def finish(self):
        """End of stream: decide falldet triggers still waiting for their 8 s aftermath."""
        if self.last_int is None:
            return []
        protos = self._falldet_events(self.falldet.flush(), self.last_int)
        return [self._event(p, self.last_dev) for p in protos]

    def dismiss(self, episode_id=None):
        if self.last_int is None:
            return False
        return self.arbiter.dismiss(self.last_int, episode_id)

    # ------------------------------------------------------------------ status
    def status(self):
        rate = self.rate.rate_hz()
        if self.falldet.error:
            status = "error"
        elif self.falldet.latched or (rate is not None and rate < self.cfg["rate"]["min_hz"]):
            status = "insufficient_rate"
        else:
            status = "ok"
        return {"status": status, "model_profile": self.model_profile, "profile_reason": self.profile_reason,
                "falldet_backend": self.falldet.backend, "raw_rate_hz_5s": None if rate is None else round(rate, 1),
                "rate_recoveries": self.counters["rate_recoveries"], "work_mode": self.work_mode,
                "config_hash": self.config_hash, **{k: v for k, v in self.counters.items() if k != "rate_recoveries"}}

    # ------------------------------------------------------------------ main loop
    def update_arrays(self, t, acc, gyro, slot_mask=None, sat_mask=None, work_mode="normal"):
        t = np.asarray(t, dtype=np.float64).reshape(-1)
        n = len(t)
        if n == 0:
            return []
        acc = np.asarray(acc, dtype=np.float64).reshape(n, 4, 3)
        gyro = np.asarray(gyro, dtype=np.float64).reshape(n, 4, 3)
        valid = np.isfinite(acc).all(-1) & np.isfinite(gyro).all(-1)
        if slot_mask is not None:
            valid &= np.asarray(slot_mask, bool).reshape(n, 4)
        sat = None if sat_mask is None else np.asarray(sat_mask, bool).reshape(n, 4)
        mode = self._mode(work_mode, t[0])
        events = []
        start = 0
        # split at proven restarts; drop duplicates and small backward steps
        keep = np.ones(n, bool)
        last = self.last_dev
        cuts = []
        for i in range(n):
            if last is not None and t[i] <= last:
                if last - t[i] > self.cfg["time"]["restart_back_s"]:
                    cuts.append(i)
                    last = t[i]
                    continue
                keep[i] = False
                self.counters["duplicates"] += 1
                continue
            last = t[i]
        for cut in cuts + [n]:
            if cut > start:
                if start in cuts:
                    events += self._restart(t[start])
                idx = np.arange(start, cut)[keep[start:cut]]
                if len(idx):
                    events += self._segment(t[idx], acc[idx], gyro[idx], valid[idx],
                                            None if sat is None else sat[idx], mode, slot_mask is not None)
            start = cut
        return events

    def _mode(self, work_mode, t0):
        """lying_work expires after lying_work_expiry_s (the backend's /worker/mode does the same)."""
        if work_mode != "lying_work":
            self.lying_since = None
            self.work_mode = "normal"
            return "normal"
        if self.lying_since is None:
            self.lying_since = t0 + self.offset
        expired = t0 + self.offset - self.lying_since > self.cfg["man_down"]["lying_work_expiry_s"]
        self.work_mode = "normal" if expired else "lying_work"
        return self.work_mode

    def _restart(self, first_dev):
        """Proven device restart: flush falldet, recreate it, keep the hub state on one clock."""
        self.counters["restarts"] += 1
        protos = self._falldet_events(self.falldet.flush(), self.last_int)
        self.falldet.new_stream()
        self.latched_since = None
        self.offset = self.last_int + 0.01 - first_dev
        self.last_dev = None
        self.boot_first_dev = first_dev
        self.track.reset_tilt()
        self.topple.reset_timing()
        t_int = first_dev + self.offset
        self.diagnostics.append(diagnostic("restart", t=float(first_dev)))
        if t_int - self.last_trigger_int <= self.cfg["time"]["restart_arm_window_s"]:
            if not self.man_down.armed:
                self.man_down.arm(t_int, "restart", self.last_trigger_int)
        else:
            self.restart_watch_until = t_int + RESTART_DOWN_WATCH_S
        for p in protos:
            p["tags"] = set(p["tags"]) | {"restart"}
        return [self._event(p, first_dev) for p in protos]

    def _segment(self, t, acc, gyro, valid, sat, mode, has_mask):
        cfg = self.cfg
        offset = self.offset
        # falldet runs on device time of this boot; masks keep its own slot handling
        f_events, diags = self.falldet.push(t, acc, gyro, valid if has_mask else None, sat, mode)
        for d in diags:
            self.diagnostics.append(diagnostic("internal_error" if d.get("code") == "internal_error"
                                               else "insufficient_rate", t=d.get("t_s"), message=d.get("message")))
        f_events.sort(key=lambda e: e.get("decided_t_s", e.get("t_s", 0.0)))
        fi = 0
        protos_out = []
        tl = (t + offset).tolist()
        al, gl, vl = acc.tolist(), gyro.tolist(), valid.tolist()
        last_int = self.last_int
        for i in range(len(tl)):
            ti = tl[i]
            if last_int is not None:
                gap = ti - last_int
                if gap > cfg["time"]["tilt_reset_gap_s"]:
                    self.track.reset_tilt()
                if gap >= LINK_GAP_S:
                    self.counters["gaps"] += 1
                if gap >= cfg["time"]["long_gap_s"]:
                    self.topple.reset_timing()
            last_int = ti
            self.rate.add(ti)
            while fi < len(f_events) and self._f_time(f_events[fi]) + offset <= ti + 1e-9:
                protos_out += self._falldet_events([f_events[fi]], ti)
                fi += 1
            protos_out += self._sample(ti, al[i], gl[i], vl[i], mode)
        self.last_int = last_int
        self.last_dev = float(t[-1])
        if fi < len(f_events):
            protos_out += self._falldet_events(f_events[fi:], last_int)
        protos_out += self._rate_check(last_int)
        self.counters["frames"] += len(tl)
        return [self._event(p, p["t"] - offset) for p in protos_out]

    @staticmethod
    def _f_time(e):
        return float(e.get("decided_t_s", e.get("t_s", 0.0)))

    def _sample(self, ti, acc, gyro, valid, mode):
        cfg = self.cfg
        info, closed = self.track.step(ti, acc, gyro, valid)
        out = []
        if cfg["topple"]["enabled"] and self.topple.step(ti, info["rot_pl"], info["down"], info["not_down"]):
            self.counters["t3_fires"] += 1
            self.last_trigger_int = ti
            evidence = {"rot_peak_rad_s": round(self.topple.last_peak, 2), "incl_deg": [None if x is None else round(x, 1) for x in info["incl"]]}
            out += self.arbiter.suspected(ti, "topple", ti, evidence)
            self.collapse.open(ti, evidence)
            if cfg["man_down"]["enabled"]:
                self.man_down.arm(ti, "topple")
        if self.restart_watch_until is not None:
            if ti > self.restart_watch_until:
                self.restart_watch_until = None
            elif info["down"] and cfg["man_down"]["enabled"]:
                self.man_down.arm(ti, "restart_down")
                self.restart_watch_until = None
        decision = self.collapse.on_sample(ti, self.track)
        if decision:
            out += self._c3(decision)
        if cfg["man_down"]["enabled"]:
            if self.arbiter.episode is not None and not self.man_down.armed:
                self.man_down.arm(ti, "episode", self.arbiter.episode["trigger_t"])
            self.man_down.on_sample(ti, self.track)
        for b in closed:
            decision = self.collapse.on_bin(b, self.track, ti)
            if decision:
                out += self._c3(decision)
            if cfg["man_down"]["enabled"]:
                for d in self.man_down.on_bin(b, self.track, ti, mode):
                    out += self.arbiter.man_down(ti, d["kind"], d["rule"], d["trigger_t"], d["tags"], d["evidence"])
        out += self.arbiter.on_sample(ti, self.track)
        return out

    def _c3(self, d):
        if d["kind"] == "fall_confirmed":
            return self.arbiter.confirmed(d["t"], "topple", d["rule"], d["trigger_t"], d["confidence"], d["tags"], d["evidence"])
        return self.arbiter.recovered(d["t"], "topple", d["rule"], d["trigger_t"], d["confidence"], d["tags"], d["evidence"])

    def _falldet_events(self, f_events, t_int):
        """falldet event dicts -> arbiter proto-events. t_int: internal time they are handled at."""
        out = []
        offset = self.offset
        for e in f_events:
            trig = float(e.get("t_s", 0.0)) + offset
            if e["type"] == "stage1_trigger":
                self.last_trigger_int = trig
                out += self.arbiter.suspected(t_int, "impact", trig, {"svm_g": round(e["svm_g"], 3),
                                                                      "rot_rad_s": round(e["rot_rad_s"], 3)})
                if self.cfg["man_down"]["enabled"]:
                    self.man_down.arm(t_int, "impact", trig)
            elif e["type"] == "nearfall_v2":
                out += self.arbiter.near_fall(t_int, "perturbation", "nearfall_v2", trig, e["confidence"])
            elif e["type"] == "stage2_result":
                tags = set()
                if e.get("gap_in_window"):
                    tags.add("gap_in_window")
                if "decided early" in str(e.get("note", "")):
                    tags.add("decided_early")
                f = e.get("features") or {}
                evidence = {k: round(f[k], 3) for k in ("impact_peak_g", "rot_peak_rad_s", "n_anchors_lying") if k in f}
                rule = str(e.get("rule"))
                if e["state"] == "fall_confirmed":
                    out += self.arbiter.confirmed(t_int, "impact", rule, trig, e["confidence"], tags, evidence)
                elif e["state"] == "near_fall":
                    out += self.arbiter.near_fall(t_int, "impact", rule, trig, e["confidence"], evidence)
                else:
                    out += self.arbiter.recovered(t_int, "impact", rule, trig, e["confidence"], tags, evidence)
        return out

    def _rate_check(self, t_int):
        """RateWatch latch recovery: >= recover_s at >= recover_hz while latched -> new falldet stream."""
        if not self.falldet.latched:
            self.latched_since = None
            return []
        if self.latched_since is None:
            self.latched_since = t_int
            return []
        rcfg = self.cfg["rate"]
        rate = self.rate.rate_hz(since=math.ceil(self.latched_since))
        if rate is not None and rate >= rcfg["recover_hz"] and t_int - self.latched_since >= rcfg["recover_s"]:
            protos = self._falldet_events(self.falldet.flush(), t_int)
            self.falldet.new_stream()
            self.latched_since = None
            self.counters["rate_recoveries"] += 1
            self.diagnostics.append(diagnostic("rate_recovered", t=t_int - self.offset, rate_hz_5s=round(rate, 1)))
            return protos
        return []

    def _event(self, p, t_dev):
        rate = self.rate.rate_hz()
        trigger = p.get("trigger_t")
        data = {"model_profile": self.model_profile, "falldet_backend": self.falldet.backend,
                "rate_hz_5s": None if rate is None else round(rate, 1), "config_hash": self.config_hash}
        evidence = dict(p.get("evidence") or {})
        return make_event(t_dev, p["kind"], p["confidence"], episode_id=p.get("episode_id"),
                          trigger_t=None if trigger is None else trigger - self.offset, path=p["path"],
                          rule=p["rule"], merged_from=p.get("merged_from"), tags=p.get("tags"),
                          evidence=evidence, data=data)
