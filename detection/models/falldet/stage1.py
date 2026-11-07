"""Stage 1: cheap threshold trigger on the pelvis + lumbar IMUs (ESP32-portable).

Every sample (100 Hz):
    svm_g  = mean(|a_pelvis|, |a_lumbar|) / g          (signal magnitude, in g)
    rot    = max(|w_pelvis|, |w_lumbar|)                (trunk angular rate, rad/s)

Three conditions, each remembered with the time it last happened:
    FREEFALL  svm_g < freefall_g for at least freefall_min_ms   (body briefly "weightless")
    IMPACT    svm_g > impact_g                                   (hitting the ground)
    ROTATION  rot   > rotation_rad_s                             (trunk tipping over fast)

TRIGGER when, within the last coincidence_window_ms, at least `min_conditions` of the
three have happened (IMPACT must be one of them if impact_required) and no trigger
fired in the last refractory_ms. All constants live in models/stage1_thresholds*.json.

The class is vectorised over a leading axis K (K recordings in lock-step) so the
same code tunes thresholds offline and runs live with K = 1. Porting to C++ means
replacing each numpy array with one float/int per strip; there is no other math.
"""
import json

import numpy as np

from .constants import G

DEFAULT_THRESHOLDS = {
    "freefall_g": 0.6,
    "freefall_min_ms": 100,
    "impact_g": 2.5,
    "rotation_rad_s": 2.0,
    "coincidence_window_ms": 1000,
    "min_conditions": 3,
    "impact_required": True,
    "refractory_ms": 5000,
}

NEVER = -10**9  # "has not happened yet", in ms


def load_thresholds(path):
    with open(path) as f:
        th = json.load(f)
    return {**DEFAULT_THRESHOLDS, **{k: v for k, v in th.items() if k in DEFAULT_THRESHOLDS}}


class Stage1Trigger:
    def __init__(self, thresholds=None, k=1):
        self.th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.k = k
        self.ff_start_ms = np.full(k, NEVER, dtype=np.int64)  # start of current low-svm run
        self.last_ff_ms = np.full(k, NEVER, dtype=np.int64)
        self.last_imp_ms = np.full(k, NEVER, dtype=np.int64)
        self.last_rot_ms = np.full(k, NEVER, dtype=np.int64)
        self.last_trig_ms = np.full(k, NEVER, dtype=np.int64)

    @staticmethod
    def signals(acc_pl, gyro_pl):
        """acc_pl, gyro_pl: (K, 2, 3) = pelvis and lumbar. Returns svm_g (K,), rot (K,)."""
        svm_g = np.linalg.norm(acc_pl, axis=-1).mean(axis=-1) / G
        rot = np.linalg.norm(gyro_pl, axis=-1).max(axis=-1)
        return svm_g, rot

    def step(self, t_ms, svm_g, rot):
        """One sample. t_ms: int time in ms; svm_g, rot: (K,) arrays. Returns bool (K,) trigger."""
        th = self.th
        # free-fall: track how long svm has stayed below the threshold
        low = svm_g < th["freefall_g"]
        self.ff_start_ms = np.where(low & (self.ff_start_ms == NEVER), t_ms, self.ff_start_ms)
        self.ff_start_ms = np.where(low, self.ff_start_ms, NEVER)
        ff_now = low & (t_ms - self.ff_start_ms >= th["freefall_min_ms"])
        self.last_ff_ms = np.where(ff_now, t_ms, self.last_ff_ms)
        self.last_imp_ms = np.where(svm_g > th["impact_g"], t_ms, self.last_imp_ms)
        self.last_rot_ms = np.where(rot > th["rotation_rad_s"], t_ms, self.last_rot_ms)

        w = th["coincidence_window_ms"]
        ff_on = (t_ms - self.last_ff_ms) <= w
        imp_on = (t_ms - self.last_imp_ms) <= w
        rot_on = (t_ms - self.last_rot_ms) <= w
        count = ff_on.astype(int) + imp_on.astype(int) + rot_on.astype(int)
        fire = count >= th["min_conditions"]
        if th["impact_required"]:
            fire &= imp_on
        fire &= (t_ms - self.last_trig_ms) >= th["refractory_ms"]
        self.last_trig_ms = np.where(fire, t_ms, self.last_trig_ms)
        return fire


def run_stage1(acc, gyro, thresholds=None, fs_hz=100.0):
    """Batch helper for one recording. acc, gyro: (N, 4, 3). Returns trigger sample indices."""
    trig = Stage1Scalar(thresholds)
    svm, rot = Stage1Trigger.signals(acc[:, :2], gyro[:, :2])  # (N,), (N,)
    svm, rot = svm.tolist(), rot.tolist()
    return [i for i in range(len(svm)) if trig.step(int(round(i * 1000.0 / fs_hz)), svm[i], rot[i])]


class Stage1Scalar:
    """Same logic as Stage1Trigger for ONE strip, in plain Python floats (fast for live use and
    the closest thing to the ESP32 C++ version). tests/ check it fires on the same samples."""

    def __init__(self, thresholds=None):
        self.th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.ff_start = None
        self.last_ff = self.last_imp = self.last_rot = self.last_trig = NEVER

    def reset_timing(self):
        """Forget free-fall / impact / rotation history (after a data gap) but keep the
        refractory timer, so a condition before the gap cannot pair with one after it."""
        self.ff_start = None
        self.last_ff = self.last_imp = self.last_rot = NEVER

    def step(self, t_ms, svm_g, rot):
        th = self.th
        if svm_g < th["freefall_g"]:
            if self.ff_start is None:
                self.ff_start = t_ms
            if t_ms - self.ff_start >= th["freefall_min_ms"]:
                self.last_ff = t_ms
        else:
            self.ff_start = None
        if svm_g > th["impact_g"]:
            self.last_imp = t_ms
        if rot > th["rotation_rad_s"]:
            self.last_rot = t_ms
        w = th["coincidence_window_ms"]
        imp_on = t_ms - self.last_imp <= w
        count = (t_ms - self.last_ff <= w) + imp_on + (t_ms - self.last_rot <= w)
        fire = count >= th["min_conditions"] and (imp_on or not th["impact_required"]) \
            and t_ms - self.last_trig >= th["refractory_ms"]
        if fire:
            self.last_trig = t_ms
        return fire
