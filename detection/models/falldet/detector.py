"""Public API of the fall detector: `detect()` for a window and `FallDetectorStream` for live data.
(For irregular timestamps, gaps, NaN or missing slots use falldet.sim_adapter.SpineFallModel,
which cleans the data and feeds this stream.)

Input contract (both entry points)
----------------------------------
* uniform 100 Hz, four anchors in this order: pelvis, lumbar, thoracic, upper_thoracic
  (sensor_id 1..4 in Hackathon2026/protocol/ORIENTATION.md), all finite
* accel in m/s^2 (specific force, ~+9.81 on +Y when upright and still), gyro in rad/s
* segment frame: +X wearer-right, +Y up, +Z posterior (mounting already corrected)

Output
------
    {"state": "normal" | "near_fall" | "fall_confirmed",
     "confidence": 0..1,                    # probability of `state` from the Stage-2 model
     "probabilities": {class: p},
     "stage1_triggered": bool,
     "trigger_index": int | None,           # sample index of the trigger inside the window
     "rule": "model" | "free_fall_no_gait" | "long_free_fall" | "lying_work_suppressed",
     "features": {name: value} | None, "aux": {"free_fall_s", "walking_after"} | None}

Training and serving share this file: scripts/train_eval.py builds Stage-2 training
rows with `window_around()` + `stage2_features()`, exactly as the stream does.
"""
import json
import os

import warnings

import numpy as np

from .constants import ANCHORS, CLASSES, FS_HZ, G
from .features import (FEATURE_NAMES, MIN_POST_S, POST_S, PRE_S, aux_event_features,
                       feature_vector, ground_contact, stage2_features)
from .nearfall import (NF_POST_S, NF_PRE_S, NearFallGBT, NearFallModel, PerturbationTrigger, decide_nf,
                       load_nf_config, trigger_signals)
from .resample import Preprocessor
from .stage1 import Stage1Scalar, Stage1Trigger, load_thresholds, run_stage1

LONG_GAP_S = 5.0   # after a data gap this long, pending triggers are decided with what exists
N_NF_PRE = int(NF_PRE_S * FS_HZ)
N_NF_POST = int(NF_POST_S * FS_HZ)
NF_DEDUPE_S = 10.0  # at most one near_fall per 10 s, whichever path produced it

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PKG_ROOT, "models")
N_PRE = int(PRE_S * FS_HZ)
N_POST = int(POST_S * FS_HZ)
DEFAULT_POST_RULES = {"height_free_fall_s": 0.35, "long_free_fall_s": 0.6,
                      "lying_work_pre_lying_deg": 60.0, "lying_work_min_anchors": 3}


def window_around(acc, gyro, i0):
    """Cut [i0 - 2 s, i0 + 8 s] (clipped to the recording). Returns acc_w, gyro_w, i0_in_window."""
    a = max(0, i0 - N_PRE)
    b = min(len(acc), i0 + N_POST + 1)
    return acc[a:b], gyro[a:b], i0 - a


class _LogRegJSON:
    """No-scikit-learn Stage-2 backend: the exported logistic regression, numpy only.
    p = softmax(W @ ((x - mean) / scale) + b). LOSO (target): Se 0.959, Sp 0.995, F1 0.976."""

    def __init__(self, path):
        with open(path) as f:
            d = json.load(f)
        assert d["feature_names"] == list(FEATURE_NAMES), "feature list mismatch"
        self.classes_ = np.array(d["classes"])
        self.mean, self.scale = np.array(d["mean"]), np.array(d["scale"])
        self.W, self.b = np.array(d["W"]), np.array(d["b"])

    def predict_proba(self, X):
        s = ((np.asarray(X) - self.mean) / self.scale) @ self.W.T + self.b
        e = np.exp(s - s.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)


class Detector:
    """Stage-1 thresholds + Stage-2 model + post-rules for one device profile ('target' or 'current').

    backend: "gbt"         gradient-boosted trees (.joblib; needs scikit-learn==1.9.1 + joblib)
             "logreg_json" logistic regression from JSON (numpy only)
             "auto"        gbt if it loads, else logreg_json (with a warning)
    """

    def __init__(self, profile="target", model_dir=MODEL_DIR, backend="auto"):
        suffix = "" if profile == "target" else f"_{profile}"
        self.profile = profile
        self.thresholds = load_thresholds(os.path.join(model_dir, f"stage1_thresholds{suffix}.json"))
        self.backend = None
        if backend in ("auto", "gbt"):
            try:
                import joblib  # noqa: deferred so the numpy-only path works without scikit-learn
                bundle = joblib.load(os.path.join(model_dir, f"stage2_model{suffix}.joblib"))
                assert list(bundle["feature_names"]) == list(FEATURE_NAMES), "feature list mismatch"
                self.model, self.backend = bundle["model"], "gbt"
                self.meta = {k: v for k, v in bundle.items() if k != "model"}
            except Exception as e:  # missing / wrong scikit-learn, unreadable file
                if backend == "gbt":
                    raise
                warnings.warn(f"falldet: GBT model unavailable ({type(e).__name__}: {e}); "
                              "using the numpy-only logistic regression (logreg_json)")
        if self.backend is None:
            self.model = _LogRegJSON(os.path.join(model_dir, f"stage2_logreg_coefficients{suffix}.json"))
            self.backend = "logreg_json"
            self.meta = {"model_type": "logreg_json"}
        self.classes = [str(c) for c in self.model.classes_]
        # near-fall v2 (falldet/nearfall.py): GBT with the GBT backend, else numpy-only LR
        nf_json = os.path.join(model_dir, "nearfall_v2.json")
        self.nf_cfg = load_nf_config(None)
        self.nf_model = None
        if os.path.exists(nf_json):
            with open(nf_json) as f:
                self.nf_cfg.update(json.load(f).get("config", {}))
            gbt_path = os.path.join(model_dir, "nearfall_v2_gbt.joblib")
            if self.backend == "gbt" and os.path.exists(gbt_path):
                self.nf_model, self.nf_threshold = NearFallGBT(gbt_path), self.nf_cfg["p_threshold"]
            else:
                self.nf_model = NearFallModel(nf_json)
                self.nf_threshold = self.nf_cfg.get("p_threshold_logreg", self.nf_cfg["p_threshold"])
        self.post_rules = dict(DEFAULT_POST_RULES)
        pr = os.path.join(model_dir, "post_rules.json")
        if os.path.exists(pr):
            with open(pr) as f:
                self.post_rules.update({k: v for k, v in json.load(f).items() if not k.startswith("_")})

    def classify(self, fdict):
        x = feature_vector(fdict)[None, :]
        p = self.model.predict_proba(x)[0]
        probs = {c: float(p[self.classes.index(c)]) if c in self.classes else 0.0 for c in CLASSES}
        state = max(probs, key=probs.get)
        return state, probs[state], probs


def apply_rules(state, conf, f, aux, rules, work_mode="normal"):
    """Hand-written corrections after the model (see models/post_rules.json).
    Returns (state, confidence, rule_name)."""
    ff = aux["free_fall_s"]
    if work_mode == "lying_work" and state == "fall_confirmed":
        pre = np.array([f[f"incl_pre_{a}"] for a in ANCHORS])
        if np.sum(pre > rules["lying_work_pre_lying_deg"]) >= rules["lying_work_min_anchors"]:
            return "normal", 0.5, "lying_work_suppressed"   # already lying before the trigger
    if state != "fall_confirmed":
        if ff >= rules["long_free_fall_s"]:
            return "fall_confirmed", 0.7, "long_free_fall"
        if ff >= rules["height_free_fall_s"] and not aux["walking_after"]:
            return "fall_confirmed", 0.6, "free_fall_no_gait"   # e.g. harness arrest, hanging upright
    return state, conf, "model"


_DEFAULT = {}


def get_detector(profile="target", backend="auto"):
    key = (profile, backend)
    if key not in _DEFAULT:
        _DEFAULT[key] = Detector(profile, backend=backend)
    return _DEFAULT[key]


def _as_arrays(window):
    """Accept {'acc': (N,4,3), 'gyro': (N,4,3)}, a tuple (acc, gyro), or one (N,4,6) array."""
    if isinstance(window, dict):
        acc, gyro = window["acc"], window["gyro"]
    elif isinstance(window, np.ndarray) and window.ndim == 3 and window.shape[1:] == (4, 6):
        acc, gyro = window[..., :3], window[..., 3:]
    else:
        acc, gyro = window
    acc = np.asarray(acc, dtype=np.float64)
    gyro = np.asarray(gyro, dtype=np.float64)
    assert acc.shape == gyro.shape and acc.ndim == 3 and acc.shape[1:] == (4, 3), \
        "expected (N, 4, 3) arrays: N samples x [pelvis, lumbar, thoracic, upper_thoracic] x xyz"
    return acc, gyro


def _decide_window(det, aw, gw, j0, work_mode):
    f = stage2_features(aw, gw, j0)
    aux = aux_event_features(aw, j0)
    state, conf, probs = det.classify(f)
    state, conf, rule = apply_rules(state, conf, f, aux, det.post_rules, work_mode)
    return state, conf, probs, rule, f, aux


def detect(window, detector=None, trigger_index=None, force_stage2=False, work_mode="normal"):
    """Classify one recorded window (>= ~10 s at uniform 100 Hz recommended).

    window: (acc, gyro) tuple of (N,4,3) arrays, {"acc":..., "gyro":...}, or an (N,4,6) array
            whose last axis is [ax, ay, az, gx, gy, gz].
    Stage 1 runs over the whole window; the FIRST trigger is classified by Stage 2.
    trigger_index: skip Stage 1 and classify around this sample instead.
    force_stage2:  if Stage 1 does not fire, classify around the largest acceleration anyway.
    work_mode:     "lying_work" suppresses confirmations when the worker was already lying.
    """
    det = detector or get_detector()
    acc, gyro = _as_arrays(window)
    triggered = True
    if trigger_index is None:
        trig = run_stage1(acc, gyro, det.thresholds)
        if trig:
            trigger_index = trig[0]
        elif force_stage2:
            triggered = False
            svm = np.linalg.norm(acc[:, :2], axis=-1).mean(-1)
            trigger_index = int(np.argmax(svm))
        else:
            return {"state": "normal", "confidence": 1.0,
                    "probabilities": {"normal": 1.0, "near_fall": 0.0, "fall_confirmed": 0.0},
                    "stage1_triggered": False, "trigger_index": None, "rule": None,
                    "features": None, "aux": None, "note": "Stage 1 did not fire; Stage 2 not run"}
    aw, gw, i0 = window_around(acc, gyro, trigger_index)
    state, conf, probs, rule, f, aux = _decide_window(det, aw, gw, i0, work_mode)
    out = {"state": state, "confidence": conf, "probabilities": probs, "stage1_triggered": triggered,
           "trigger_index": int(trigger_index), "rule": rule, "features": f, "aux": aux}
    if len(aw) - i0 < MIN_POST_S * FS_HZ:
        out["note"] = f"only {(len(aw) - i0) / FS_HZ:.1f} s after the trigger; aftermath poorly observed"
    return out


# ----------------------------------------------------------------------------- streaming
def parse_sample(sample):
    """Accept one 4-IMU sample in any of these forms and return (t_ms or None, acc(4,3), gyro(4,3)).

    1. Telemetry-style dict (firmware/proto/telemetry.proto ImuStripData, JSON-ified):
       {"time": 1234, "data": [{"accel": {"x":..,"y":..,"z":..}, "gyro": {...}}, x4]}
       optional per-IMU "sensor_id" (1..4) to give the order; otherwise list order = 1..4.
    2. Plain dict: {"t_ms": 1234, "acc": [[x,y,z] x4], "gyro": [[x,y,z] x4]}
    3. Tuple (acc(4,3), gyro(4,3)).
    """
    if isinstance(sample, dict) and "data" in sample:
        imus = sample["data"]
        if all("sensor_id" in m for m in imus):
            imus = sorted(imus, key=lambda m: m["sensor_id"])
        acc = np.array([[m["accel"].get(k, 0.0) for k in "xyz"] for m in imus], dtype=np.float64)
        gyro = np.array([[m["gyro"].get(k, 0.0) for k in "xyz"] for m in imus], dtype=np.float64)
        t = sample.get("time")
    elif isinstance(sample, dict):
        acc = np.asarray(sample["acc"], dtype=np.float64)
        gyro = np.asarray(sample["gyro"], dtype=np.float64)
        t = sample.get("t_ms")
    else:
        acc, gyro = (np.asarray(x, dtype=np.float64) for x in sample)
        t = None
    assert acc.shape == (4, 3) and gyro.shape == (4, 3), "need 4 IMUs x 3 axes"
    return t, acc, gyro


class FallDetectorStream:
    """Streaming detector. Three ways to feed it:

      push_batch(t, acc, gyro, slot_mask=None, sat_mask=None, work_mode="normal")
          REAL-WORLD frames: (N,) device seconds (irregular, gaps), (N,4,3) m/s^2 incl. gravity,
          (N,4,3) rad/s, (N,4) bool masks. Resampled to 100 Hz internally (falldet/resample.py);
          NaN / missing slots are filled from the nearest anchor; gaps > 0.1 s are not bridged and
          reset Stage-1 timing; < 40 Hz input gives one 'insufficient_rate' diagnostic and no
          detections. Same shapes as ThunderSim's FrameBatch.
      push(sample)                one uniform-100 Hz sample (telemetry dict / dict / tuple)
      push_block(acc, gyro, ...)  (M,4,3) uniform-100 Hz finite samples

    All return a list of event dicts:
      {"type": "stage1_trigger", "state": "suspected", "t_ms", "t_s", "sample", "svm_g", "rot_rad_s"}
          immediately when Stage 1 fires (maps to telemetry fall_state 'suspected')
      {"type": "stage2_result", "state", "confidence", "probabilities", "rule", "t_ms", "t_s"
       (trigger), "sample", "decided_at_ms", "decided_t_s", "decided_sample", "features", "aux",
       "slots_missing_frac", "gap_in_window", ["note"]}
          800 samples (8 s) after the trigger, once the aftermath has been observed
      {"type": "diagnostic", "code": "insufficient_rate" | "internal_error", "state": "normal", ...}
          push_batch only; insufficient_rate is emitted once per stream

    `state` (property): 'suspected' while a trigger awaits Stage 2, otherwise the last Stage-2
    decision for `hold_s` seconds, else 'normal'.
    """

    def __init__(self, profile="target", detector=None, hold_s=30.0, work_mode="normal",
                 backend="auto", nominal_fs_hz=None, nearfall="v2"):
        """nearfall: "v2" (default) adds the short-window near-fall path and requires ground contact
        (or v2 agreement) for the fall path's own near_fall; "v1" = the original behaviour."""
        self.det = detector or get_detector(profile, backend)
        self.stage1 = Stage1Scalar(self.det.thresholds)
        self.nf_on = nearfall == "v2" and self.det.nf_model is not None
        self.nearfall = "v2" if self.nf_on else "v1"
        self.nf_trig = PerturbationTrigger(self.det.nf_cfg)
        self.nf_pending = []          # (sample, t_ms) perturbation candidates awaiting 3 s
        self.nf_deferred = []         # positive near-fall events held while a fall trigger is pending
        self.nf_pos_samples = []      # samples of positive v2 decisions (for fall-path agreement)
        self.recent_fall = []         # (trigger sample, state) of fall-path decisions
        self.last_nf_emit_s = -1e18
        self.cap = N_PRE + N_POST + 16
        self.acc_buf = np.zeros((self.cap, 4, 3))
        self.gyro_buf = np.zeros((self.cap, 4, 3))
        self.t_buf = np.zeros(self.cap)               # time of each internal sample, s
        self.miss_buf = np.zeros((self.cap, 4), bool)  # slot was filled from a neighbour
        self.n = 0                    # samples received
        self.pending = []             # (sample_index, t_ms) of triggers awaiting Stage 2
        self.last = None              # last stage2_result event
        self.hold_ms = hold_s * 1000.0
        self.t_ms = 0
        self.work_mode = work_mode
        self.nominal_fs_hz = nominal_fs_hz
        self.pre = None
        self.rate_warned = False
        self.gap_marks = []           # internal sample indices that follow a data gap

    # ----------------------------------------------------------------- real-world frames
    def push_batch(self, t, acc, gyro, slot_mask=None, sat_mask=None, work_mode="normal"):
        """See class docstring. sat_mask is accepted but not used. Never raises."""
        try:
            self.work_mode = work_mode if work_mode in ("normal", "lying_work") else "normal"
            if self.pre is None:
                self.pre = Preprocessor(self.nominal_fs_hz)
            segs = self.pre.process(t, acc, gyro, slot_mask)
            events = []
            if self.pre.insufficient_rate:
                if not self.rate_warned:
                    self.rate_warned = True
                    dts = list(self.pre.recent_dt)
                    tt = np.asarray(t, dtype=np.float64)
                    events.append({
                        "type": "diagnostic", "code": "insufficient_rate", "state": "normal",
                        "t_s": float(np.nanmax(tt)) if np.isfinite(tt).any() else None,
                        "median_interval_s": float(np.median(dts)) if dts else None,
                        "message": "frame rate far below 100 Hz (e.g. 0.5 Hz firmware): free fall and "
                                   "impact cannot be observed; no detections will be made"})
                return events
            for seg in segs:
                if seg.gap_before:
                    events += self._on_gap(seg.gap_s)
                events += self.push_block(seg.acc, seg.gyro, t_ms=np.round(seg.t * 1000.0).astype(np.int64),
                                          t_s=seg.t, missing=seg.missing)
            return events
        except Exception as e:  # last line of defence: the caller's loop must never die
            return [{"type": "diagnostic", "code": "internal_error", "state": "normal",
                     "message": f"{type(e).__name__}: {e}"}]

    def _on_gap(self, gap_s):
        self.stage1.reset_timing()            # no free fall before + impact after a gap
        self.nf_trig.reset_timing()
        self.gap_marks.append(self.n)
        if gap_s > LONG_GAP_S:                # aftermath lost: decide now with what exists
            out = self.flush()
            for ev in out:
                ev["note"] = f"decided early: {gap_s:.1f} s data gap after the trigger"
            return out
        return []

    # ----------------------------------------------------------------- uniform 100 Hz
    def push(self, sample):
        t, acc, gyro = parse_sample(sample)
        return self.push_block(acc[None], gyro[None], None if t is None else [t])

    def push_block(self, acc, gyro, t_ms=None, t_s=None, missing=None):
        """acc, gyro: (M, 4, 3) consecutive 100 Hz samples (finite). t_ms: optional (M,) ints;
        t_s: optional (M,) float seconds; missing: optional (M, 4) bool (slot was filled)."""
        acc = np.asarray(acc, dtype=np.float64)
        gyro = np.asarray(gyro, dtype=np.float64)
        svm, rot = Stage1Trigger.signals(acc[:, :2], gyro[:, :2])
        svm, rot = svm.tolist(), rot.tolist()
        if self.nf_on:
            _, ut_g, rot_all = trigger_signals(acc, gyro)
            ut_g, rot_all = ut_g.tolist(), rot_all.tolist()
        events = []
        for j in range(len(acc)):
            k = self.n % self.cap
            self.acc_buf[k] = acc[j]
            self.gyro_buf[k] = gyro[j]
            self.miss_buf[k] = False if missing is None else missing[j]
            self.t_ms = int(t_ms[j]) if t_ms is not None else int(round(self.n * 1000.0 / FS_HZ))
            self.t_buf[k] = float(t_s[j]) if t_s is not None else self.t_ms / 1000.0
            if self.stage1.step(self.t_ms, svm[j], rot[j]):
                self.pending.append((self.n, self.t_ms))
                events.append({"type": "stage1_trigger", "state": "suspected", "t_ms": self.t_ms,
                               "t_s": self.t_buf[k], "sample": self.n, "svm_g": svm[j], "rot_rad_s": rot[j]})
            if self.nf_on and self.nf_trig.step(self.t_ms, svm[j], ut_g[j], rot_all[j]):
                self.nf_pending.append((self.n, self.t_ms))
            self.n += 1
            while self.nf_pending and self.n - 1 - self.nf_pending[0][0] >= N_NF_POST:
                events += self._decide_nf(*self.nf_pending.pop(0))
            while self.pending and self.n - 1 - self.pending[0][0] >= N_POST:
                events += self._decide(*self.pending.pop(0))
        return events

    def flush(self):
        """End of recording: decide pending triggers with whatever post-trigger data exists."""
        out = []
        while self.nf_pending:
            out += self._decide_nf(*self.nf_pending.pop(0))
        while self.pending:
            out += self._decide(*self.pending.pop(0))
        return out

    def _decide(self, n_trig, t_trig):
        a = max(0, n_trig - N_PRE, self.n - self.cap)
        b = min(self.n, n_trig + N_POST + 1)
        idx = np.arange(a, b) % self.cap
        self.gap_marks = [g for g in self.gap_marks if g > self.n - self.cap]
        ev = {"type": "stage2_result", "t_ms": t_trig, "t_s": float(self.t_buf[n_trig % self.cap]),
              "sample": n_trig, "decided_at_ms": self.t_ms,
              "decided_t_s": float(self.t_buf[(self.n - 1) % self.cap]), "decided_sample": self.n - 1,
              "slots_missing_frac": [round(float(x), 3) for x in self.miss_buf[idx].mean(axis=0)],
              "gap_in_window": any(a < g < b for g in self.gap_marks)}
        try:
            state, conf, probs, rule, f, aux = _decide_window(
                self.det, self.acc_buf[idx], self.gyro_buf[idx], n_trig - a, self.work_mode)
        except Exception as e:  # never raise from a live stream
            state, conf, rule, f, aux = "normal", 0.0, f"error: {type(e).__name__}: {e}", None, None
            probs = {"normal": 1.0, "near_fall": 0.0, "fall_confirmed": 0.0}
        if self.nf_on and state == "near_fall":
            # v2: the fall path's near_fall needs ground contact (fell and got up) or v2 agreement
            try:
                gc = ground_contact(self.acc_buf[idx], self.gyro_buf[idx], n_trig - a)
            except Exception:
                gc = True
            agree = any(abs(s - n_trig) <= N_NF_POST for s in self.nf_pos_samples)
            ev["ground_contact"] = gc
            if not (gc or agree):
                state, rule = "normal", "nf_v2_no_ground_contact"
            elif ev["decided_t_s"] - self.last_nf_emit_s < NF_DEDUPE_S:
                state, rule = "normal", "nf_v2_duplicate"
            else:
                self.last_nf_emit_s = ev["decided_t_s"]
        ev.update(state=state, confidence=conf, probabilities=probs, rule=rule, features=f, aux=aux)
        self.last = ev
        out = [ev]
        if self.nf_on:
            self.recent_fall = [(s, st) for s, st in self.recent_fall if s > self.n - 3000] + [(n_trig, state)]
            if not self.pending:
                out += self._release_deferred()
        return out

    # ----------------------------------------------------------------- near-fall v2
    def _decide_nf(self, n0, t0):
        a = max(0, n0 - N_NF_PRE, self.n - self.cap)
        idx = np.arange(a, min(self.n, n0 + N_NF_POST + 1)) % self.cap
        try:
            pos, p, f, why = decide_nf(self.det.nf_model, self.det.nf_threshold, self.det.nf_cfg,
                                       self.acc_buf[idx], self.gyro_buf[idx], n0 - a)
        except Exception:
            return []
        if not pos:
            return []
        self.nf_pos_samples = [s for s in self.nf_pos_samples if s > self.n - 3000] + [n0]
        ev = {"type": "nearfall_v2", "state": "near_fall", "confidence": p, "rule": "nearfall_v2",
              "t_ms": t0, "t_s": float(self.t_buf[n0 % self.cap]), "sample": n0, "decided_at_ms": self.t_ms,
              "decided_t_s": float(self.t_buf[(self.n - 1) % self.cap]), "decided_sample": self.n - 1,
              "nf_features": f}
        if self.pending:                      # a fall trigger is still open: wait for its verdict
            self.nf_deferred.append(ev)
            return []
        return self._emit_nf(ev)

    def _release_deferred(self):
        out = []
        for ev in self.nf_deferred:
            if any(st == "fall_confirmed" and abs(s - ev["sample"]) <= N_POST for s, st in self.recent_fall):
                continue                      # it was the start of a real fall
            ev.update(decided_at_ms=self.t_ms, decided_t_s=float(self.t_buf[(self.n - 1) % self.cap]),
                      decided_sample=self.n - 1, deferred=True)
            out += self._emit_nf(ev)
        self.nf_deferred = []
        return out

    def _emit_nf(self, ev):
        if ev["decided_t_s"] - self.last_nf_emit_s < NF_DEDUPE_S:
            return []
        self.last_nf_emit_s = ev["decided_t_s"]
        self.last = ev
        return [ev]

    @property
    def state(self):
        if self.pending:
            return "suspected"
        if self.last and self.t_ms - self.last["decided_at_ms"] <= self.hold_ms:
            return self.last["state"]
        return "normal"


def load_fixture(path):
    """Read a fixtures/*.jsonl file -> (header dict, list of sample dicts)."""
    header, samples = None, []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("type") == "header":
                header = rec
            else:
                samples.append(rec)
    return header, samples


def fixture_arrays(samples):
    """List of telemetry-style samples -> acc, gyro arrays (N, 4, 3)."""
    parsed = [parse_sample(s) for s in samples]
    return np.array([p[1] for p in parsed]), np.array([p[2] for p in parsed])


__all__ = ["detect", "FallDetectorStream", "Detector", "get_detector", "window_around", "apply_rules",
           "parse_sample", "load_fixture", "fixture_arrays", "ANCHORS", "CLASSES", "G"]
