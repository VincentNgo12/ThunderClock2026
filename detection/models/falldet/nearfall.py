"""Near-fall v2: a separate, short-window path for "balance lost, then recovered".

Why a separate path: the fall path (Stage 1 impact trigger + 8 s Stage-2 window) was built
to confirm falls that end on the floor. A stumble, a slip caught on a rail or a blow that
the worker absorbs has no hard impact and no lying aftermath, so Stage 2 called them
`normal` (ThunderSim NF-* scenarios: 0-5 % near_fall). Near-falls are logged as near-miss
events, never as man-down alarms.

    1. PERTURBATION TRIGGER (cheap, every sample, constants in models/nearfall_v2.json):
         dip    svm = mean(|a_pelvis|, |a_lumbar|) < dip_g for >= dip_ms   (feet unloaded)
         jolt   svm > jolt_g  or  |a_upper_thoracic| > jolt_g              (blow / arm arrest)
         turn   max |omega| over the four anchors > rot_rad_s              (trunk thrown)
       fires when `turn` and (`dip` or `jolt`) occur within coincidence_ms; refractory_ms.
    2. DECISION 3 s later on the window [-4 s, +3 s] around the trigger:
         recovery gate: every anchor's median inclination 2-3 s after the trigger < upright_deg
         classifier   : p(near_fall) from 22 short-window features (below) >= p_threshold
       otherwise the candidate is dropped (the fall path still handles real falls).

Features compare the perturbation with what the worker was doing just before (running and
jack-hammering have dips and spikes every step; a stumble stands out from its own context).
"""
import json
import os

import numpy as np

from .constants import ANCHORS, FS_HZ, G
from .tilt import inclination_deg, run_tilt

NF_PRE_S = 4.0
NF_POST_S = 3.0
NEVER = -10**9

DEFAULT_NF_CONFIG = {
    "dip_g": 0.85, "dip_ms": 30, "jolt_g": 1.8, "rot_rad_s": 1.2,
    "coincidence_ms": 500, "refractory_ms": 3000,
    "upright_deg": 45.0, "p_threshold": 0.5,
}

NF_FEATURE_NAMES = (
    "dip_min_g", "peak_g", "ut_peak_g", "jerk_svm_g_s", "jerk_vert_pelvis_g_s",
    "rot_peak", "rot_peak_pl", "pitch_rate_peak", "roll_rate_peak", "yaw_rate_peak",
    "incl_exc_ut", "incl_exc_pelvis", "incl_return", "incl_post_max",
    "pre_svm_std", "pre_rot_p95", "pre_svm_min", "pre_svm_max", "pre_periodicity",
    "post_svm_std", "rot_ratio", "dip_ratio",
)


def load_nf_config(path):
    cfg = dict(DEFAULT_NF_CONFIG)
    if path and os.path.exists(path):
        with open(path) as f:
            cfg.update({k: v for k, v in json.load(f).items() if not k.startswith("_")})
    return cfg


class PerturbationTrigger:
    """Scalar, one strip; same style as stage1.Stage1Scalar (ESP32-portable)."""

    def __init__(self, cfg=None):
        self.c = {**DEFAULT_NF_CONFIG, **(cfg or {})}
        self.reset_timing()
        self.last_trig = NEVER

    def reset_timing(self):
        self.dip_start = None
        self.last_dip = self.last_jolt = self.last_turn = NEVER

    def step(self, t_ms, svm_g, ut_g, rot_all):
        c = self.c
        if svm_g < c["dip_g"]:
            if self.dip_start is None:
                self.dip_start = t_ms
            if t_ms - self.dip_start >= c["dip_ms"]:
                self.last_dip = t_ms
        else:
            self.dip_start = None
        if svm_g > c["jolt_g"] or ut_g > c["jolt_g"]:
            self.last_jolt = t_ms
        if rot_all > c["rot_rad_s"]:
            self.last_turn = t_ms
        w = c["coincidence_ms"]
        fire = (t_ms - self.last_turn <= w) and ((t_ms - self.last_dip <= w) or (t_ms - self.last_jolt <= w)) \
            and t_ms - self.last_trig >= c["refractory_ms"]
        if fire:
            self.last_trig = t_ms
        return fire


def trigger_signals(acc, gyro):
    """acc, gyro (N,4,3) -> svm_g (N,), ut_g (N,), rot_all (N,)."""
    amag = np.linalg.norm(acc, axis=-1) / G
    return amag[:, :2].mean(axis=1), amag[:, 3], np.linalg.norm(gyro, axis=-1).max(axis=1)


def run_perturbation_trigger(acc, gyro, cfg=None, fs=FS_HZ):
    """Batch helper for one uniform recording -> list of trigger sample indices."""
    trig = PerturbationTrigger(cfg)
    svm, ut, rot = (x.tolist() for x in trigger_signals(acc, gyro))
    return [i for i in range(len(svm)) if trig.step(int(round(i * 1000.0 / fs)), svm[i], ut[i], rot[i])]


# --------------------------------------------------------------------------------- features
def _sl(x, i0, a_s, b_s, fs):
    a = min(max(0, i0 + int(round(a_s * fs))), len(x) - 1)
    b = min(len(x), i0 + int(round(b_s * fs)))
    return x[a:max(a + 1, b)]


def _smooth(x, k=5):
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode="same")


def _periodicity(x, fs):
    x = x - x.mean()
    if len(x) < int(1.5 * fs) or np.dot(x, x) < 1e-9:
        return 0.0
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac / ac[0]
    lo, hi = int(0.3 * fs), min(int(1.2 * fs), len(ac) - 1)
    return float(np.max(ac[lo:hi])) if hi > lo else 0.0


def nf_window(acc, gyro, i0, fs=FS_HZ):
    """Cut [i0 - 4 s, i0 + 3 s]. Returns acc_w, gyro_w, i0_in_window."""
    a = max(0, i0 - int(NF_PRE_S * fs))
    b = min(len(acc), i0 + int(NF_POST_S * fs) + 1)
    return acc[a:b], gyro[a:b], i0 - a


def nf_features(acc, gyro, i0, fs=FS_HZ):
    """acc, gyro: (N,4,3) window (segment frame, m/s^2, rad/s); i0 = trigger index in it."""
    acc = np.asarray(acc, dtype=np.float64)
    gyro = np.asarray(gyro, dtype=np.float64)
    amag = np.linalg.norm(acc, axis=-1) / G
    wmag = np.linalg.norm(gyro, axis=-1)
    svm = amag[:, :2].mean(axis=1)
    incl = inclination_deg(run_tilt(acc, gyro))
    svm_s = _smooth(svm)
    vert = _smooth(acc[:, 0, 1] / G)
    f = {}
    f["dip_min_g"] = float(_sl(svm, i0, -0.5, 0.5, fs).min())
    f["peak_g"] = float(_sl(svm, i0, -0.5, 1.0, fs).max())
    f["ut_peak_g"] = float(_sl(amag[:, 3], i0, -0.5, 1.0, fs).max())
    f["jerk_svm_g_s"] = float(np.abs(np.diff(_sl(svm_s, i0, -0.5, 1.0, fs))).max(initial=0.0) * fs)
    f["jerk_vert_pelvis_g_s"] = float(np.abs(np.diff(_sl(vert, i0, -0.5, 1.0, fs))).max(initial=0.0) * fs)
    seg_w = _sl(wmag, i0, -0.5, 1.0, fs)
    f["rot_peak"] = float(seg_w.max())
    f["rot_peak_pl"] = float(seg_w[:, :2].max())
    seg_g = np.abs(_sl(gyro, i0, -0.5, 1.0, fs))
    f["pitch_rate_peak"] = float(seg_g[..., 0].max())
    f["yaw_rate_peak"] = float(seg_g[..., 1].max())
    f["roll_rate_peak"] = float(seg_g[..., 2].max())
    pre_incl = np.median(_sl(incl, i0, -2.0, -0.3, fs), axis=0)
    exc = _sl(incl, i0, 0.0, 1.5, fs).max(axis=0) - pre_incl
    f["incl_exc_ut"] = float(exc[3])
    f["incl_exc_pelvis"] = float(exc[0])
    post_incl = np.median(_sl(incl, i0, 2.0, 3.0, fs), axis=0)
    f["incl_return"] = float(np.abs(post_incl - pre_incl).max())
    f["incl_post_max"] = float(post_incl.max())
    pre = _sl(svm, i0, -NF_PRE_S, -0.5, fs)
    f["pre_svm_std"] = float(pre.std())
    f["pre_rot_p95"] = float(np.percentile(_sl(wmag.max(axis=1), i0, -NF_PRE_S, -0.5, fs), 95))
    f["pre_svm_min"] = float(pre.min())
    f["pre_svm_max"] = float(pre.max())
    f["pre_periodicity"] = _periodicity(pre, fs)
    f["post_svm_std"] = float(_sl(svm, i0, 1.5, NF_POST_S, fs).std())
    f["rot_ratio"] = f["rot_peak"] / (f["pre_rot_p95"] + 0.3)
    f["dip_ratio"] = (1.0 - f["dip_min_g"]) / (1.0 - min(f["pre_svm_min"], 1.0) + 0.05)
    return f


def nf_vector(f):
    return np.array([f[n] for n in NF_FEATURE_NAMES], dtype=np.float64)


class NearFallModel:
    """Logistic regression from JSON (numpy only): p = sigmoid(w . (x - mean) / scale + b)."""

    def __init__(self, path):
        with open(path) as fh:
            d = json.load(fh)
        assert d["feature_names"] == list(NF_FEATURE_NAMES), "near-fall feature list mismatch"
        self.mean, self.scale = np.array(d["mean"]), np.array(d["scale"])
        self.w, self.b = np.array(d["w"]), float(d["b"])
        self.cfg = d.get("config", {})

    def proba(self, f):
        z = (nf_vector(f) - self.mean) / self.scale
        return float(1.0 / (1.0 + np.exp(-(z @ self.w + self.b))))


def decide_nf(model, threshold, cfg, acc_w, gyro_w, j0, fs=FS_HZ):
    """Returns (is_near_fall, p, features, reason)."""
    f = nf_features(acc_w, gyro_w, j0, fs)
    if f["incl_post_max"] >= cfg["upright_deg"]:
        return False, 0.0, f, "not_upright_after"
    p = model.proba(f)
    return p >= threshold, p, f, "classifier"


__all__ = ["PerturbationTrigger", "run_perturbation_trigger", "nf_features", "nf_window", "NF_FEATURE_NAMES",
           "NearFallModel", "decide_nf", "load_nf_config", "DEFAULT_NF_CONFIG", "ANCHORS"]


class NearFallGBT:
    """Main near-fall v2 model (models/nearfall_v2_gbt.joblib; needs scikit-learn 1.9.1 + joblib)."""

    def __init__(self, path):
        import joblib
        b = joblib.load(path)
        assert list(b["feature_names"]) == list(NF_FEATURE_NAMES), "near-fall feature list mismatch"
        self.model = b["model"]

    def proba(self, f):
        return float(self.model.predict_proba(nf_vector(f)[None, :])[0, 1])
