"""Stage 2 features over a post-trigger window (all four anchors).

Window layout (100 Hz):  [trigger - PRE_S, trigger + POST_S]
                          ^ 2 s before    ^ 8 s after          (10 s total)

Two feature families, as recommended in the research report:

Multiphase (Palmerini et al. 2020 style), from svm = mean(|a_pelvis|, |a_lumbar|) in g:
    ff_min_g          deepest free-fall dip around the trigger
    impact_peak_g     impact peak
    jerk_peak_g_s     steepest change of svm (bandwidth-sensitive)
    rot_peak_rad_s    peak trunk angular rate (pelvis/lumbar)
    pre_std_g         activity level before the event (walking vs standing)
    post_std_g        acceleration variability after the impact (still vs moving)
    post_std_late_g   same, later part only (got up again?)
    post_periodicity  autocorrelation peak at 0.3-1.5 s lag (walking resumes -> periodic)
    post_gyro_mean    mean trunk angular rate after the impact

Four-anchor (the spine strip's own advantage):
    incl_post_<anchor>   median inclination 2-8 s after trigger, deg (0 upright, 90 lying)
    dincl_<anchor>       change from before the trigger, deg
    n_anchors_lying      how many anchors end up > 60 deg
    incl_post_min / incl_post_range / incl_post_std   cross-anchor agreement
    ut_minus_pelvis_post upper thoracic minus pelvis inclination (bent over vs lying flat)
    still_frac           fraction of post samples where every anchor is still
    longest_still_s      longest continuous still run in the post window
"""
import numpy as np

from .constants import ANCHORS, FS_HZ, G
from .tilt import inclination_deg, run_tilt

PRE_S = 2.0
POST_S = 8.0
MIN_POST_S = 3.0          # below this, the window is too short to judge the aftermath
LYING_DEG = 60.0
STILL_GYRO_RAD_S = 0.3
STILL_ACC_G = 0.10

FEATURE_NAMES = (
    ["ff_min_g", "impact_peak_g", "jerk_peak_g_s", "rot_peak_rad_s", "pre_std_g",
     "post_std_g", "post_std_late_g", "post_periodicity", "post_gyro_mean"]
    + [f"incl_post_{a}" for a in ANCHORS]
    + [f"incl_pre_{a}" for a in ANCHORS]
    + [f"dincl_{a}" for a in ANCHORS]
    + ["n_anchors_lying", "incl_post_min", "incl_post_range", "incl_post_std",
       "ut_minus_pelvis_post", "still_frac", "longest_still_s"]
)


def _sl(x, i0, a_s, b_s, fs):
    """Slice x between i0+a_s and i0+b_s seconds (clipped to the array)."""
    a = min(max(0, i0 + int(round(a_s * fs))), len(x) - 1)   # never empty: at least one sample
    b = min(len(x), i0 + int(round(b_s * fs)))
    return x[a:max(a + 1, b)]


def _periodicity(x, fs):
    """Max normalised autocorrelation at lags 0.3-1.5 s (0 = aperiodic, 1 = perfectly periodic)."""
    x = x - x.mean()
    if len(x) < int(2.0 * fs) or np.dot(x, x) < 1e-9:
        return 0.0
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac / ac[0]
    lo, hi = int(0.3 * fs), min(int(1.5 * fs), len(ac) - 1)
    return float(np.max(ac[lo:hi])) if hi > lo else 0.0


def _longest_run(mask):
    best = run = 0
    for m in mask:
        run = run + 1 if m else 0
        best = max(best, run)
    return best


def stage2_features(acc, gyro, i0, fs=FS_HZ, up=None):
    """acc, gyro: (N, 4, 3) window in segment frame, m/s^2 and rad/s.
    i0: index of the Stage-1 trigger inside the window. Returns dict name -> float."""
    acc = np.asarray(acc, dtype=np.float64)
    gyro = np.asarray(gyro, dtype=np.float64)
    if up is None:
        up = run_tilt(acc, gyro)                       # (N, 4, 3), restarts at window start
    incl = inclination_deg(up)                         # (N, 4)
    amag = np.linalg.norm(acc, axis=-1) / G            # (N, 4) in g
    wmag = np.linalg.norm(gyro, axis=-1)               # (N, 4) rad/s
    svm = amag[:, :2].mean(axis=1)                     # pelvis+lumbar, g
    rot = wmag[:, :2].max(axis=1)

    f = {}
    f["ff_min_g"] = float(_sl(svm, i0, -1.0, 0.5, fs).min())
    f["impact_peak_g"] = float(_sl(svm, i0, -0.5, 1.0, fs).max())
    f["jerk_peak_g_s"] = float(np.abs(np.diff(_sl(svm, i0, -0.5, 1.0, fs))).max(initial=0.0) * fs)
    f["rot_peak_rad_s"] = float(_sl(rot, i0, -1.0, 1.0, fs).max())
    f["pre_std_g"] = float(_sl(svm, i0, -PRE_S, -0.5, fs).std())
    post = _sl(svm, i0, 1.5, POST_S, fs)
    f["post_std_g"] = float(post.std())
    f["post_std_late_g"] = float(_sl(svm, i0, 4.0, POST_S, fs).std())
    f["post_periodicity"] = _periodicity(post, fs)
    f["post_gyro_mean"] = float(_sl(wmag.max(axis=1), i0, 1.5, POST_S, fs).mean())

    inc_post = np.median(_sl(incl, i0, 2.0, POST_S, fs), axis=0)   # (4,)
    inc_pre = np.median(_sl(incl, i0, -PRE_S, -0.5, fs), axis=0)
    for k, a in enumerate(ANCHORS):
        f[f"incl_post_{a}"] = float(inc_post[k])
    for k, a in enumerate(ANCHORS):
        f[f"incl_pre_{a}"] = float(inc_pre[k])
    for k, a in enumerate(ANCHORS):
        f[f"dincl_{a}"] = float(inc_post[k] - inc_pre[k])
    f["n_anchors_lying"] = float(np.sum(inc_post > LYING_DEG))
    f["incl_post_min"] = float(inc_post.min())
    f["incl_post_range"] = float(inc_post.max() - inc_post.min())
    f["incl_post_std"] = float(inc_post.std())
    f["ut_minus_pelvis_post"] = float(inc_post[3] - inc_post[0])

    still = ((_sl(wmag, i0, 2.0, POST_S, fs) < STILL_GYRO_RAD_S).all(axis=1)
             & (np.abs(_sl(amag, i0, 2.0, POST_S, fs) - 1.0) < STILL_ACC_G).all(axis=1))
    f["still_frac"] = float(still.mean())
    f["longest_still_s"] = _longest_run(still) / fs
    return f


def feature_vector(fdict):
    return np.array([fdict[n] for n in FEATURE_NAMES], dtype=np.float64)


def rule_based_stage2(fdict):
    """The report's hand-written confirmer, kept as a baseline / fallback:
    lying at >= 3 of 4 anchors AND still for >= 4 s after the impact."""
    if fdict["n_anchors_lying"] >= 3 and fdict["longest_still_s"] >= 4.0:
        return "fall_confirmed"
    return "near_fall"


# ---------------------------------------------------------------------------------------------
# Auxiliary event measures used by the post-rules (detector.apply_rules), NOT by the trained
# model. They cover a case the training data never contained: a fall arrested by a harness,
# after which the worker hangs near-upright (not lying) and may be still or struggling.
FREEFALL_AUX_G = 0.5      # "weightless" threshold for measuring free-fall duration


def free_fall_duration_s(acc, i0, fs=FS_HZ):
    """Longest run of svm < 0.5 g in [trigger - 1.5 s, trigger + 0.3 s]. A drop of height h
    gives t = sqrt(2h/g): 0.3 s ~ 0.45 m, 0.6 s ~ 1.8 m."""
    svm = (np.linalg.norm(np.asarray(acc, dtype=np.float64)[:, :2], axis=-1) / G).mean(axis=1)
    seg = _sl(svm, i0, -1.5, 0.3, fs)
    return _longest_run(seg < FREEFALL_AUX_G) / fs


def walking_after(acc, i0, fs=FS_HZ):
    """Gait after the event: periodic pelvis vertical specific force at 1.2-2.6 Hz with a real
    amplitude, in [trigger + 2 s, trigger + 8 s]. Same idea as the simulator baseline."""
    y = _sl(np.asarray(acc, dtype=np.float64)[:, 0, 1] / G, i0, 2.0, POST_S, fs)
    if len(y) < int(2.0 * fs) or y.std() < 0.05:
        return False
    y = y - y.mean()
    ac = np.correlate(y, y, mode="full")[len(y) - 1:]
    ac = ac / ac[0]
    lo, hi = int(fs / 2.6), int(fs / 1.2)
    return bool(hi < len(ac) and ac[lo:hi + 1].max() > 0.3)


def aux_event_features(acc, i0, fs=FS_HZ):
    return {"free_fall_s": float(free_fall_duration_s(acc, i0, fs)),
            "walking_after": walking_after(acc, i0, fs)}


def ground_contact(acc, gyro, i0, fs=FS_HZ, deg=LYING_DEG):
    """True if at least 2 anchors exceeded `deg` inclination within 6 s after the trigger
    (the body went to the floor). Used by near-fall v2 to keep only fall-path near_falls with
    real ground contact (fell and got up), not jolts while upright or seated."""
    incl = inclination_deg(run_tilt(np.asarray(acc, dtype=np.float64), np.asarray(gyro, dtype=np.float64)))
    peak = _sl(incl, i0, 0.0, 6.0, fs).max(axis=0)
    return bool(np.sort(peak)[-2] > deg)
