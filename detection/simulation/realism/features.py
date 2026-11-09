# File: realism/features.py
# Purpose: Features computed identically on real and emulated-simulated signals: event level
#          (one aligned fall or fall-like event) and activity level (5 s windows).
# Contributors: Unassigned
"""All inputs are (N, 2, 3) segment-frame arrays at the dataset rate: slot 0 waist/hip, slot 1
chest/back; acc in m/s^2 (gravity included), gyro in rad/s. Magnitudes are in g.

The event reference i0 is the waist |a| peak inside a search window, found the same way on both
sides (the simulator's ground-truth impact time is NOT used for alignment).
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from .paths import G

EVENT_FEATURES = {
    # name: (description, downstream relevance 0..1: 1 = a falldet Stage-1/2 input or impact severity)
    "peak_w": ("waist |a| impact peak, g", 1.0),
    "peak_c": ("chest |a| peak within 0.3 s of the waist peak, g", 0.7),
    "ffmin_w": ("waist |a| minimum in the 1 s before the peak, g", 1.0),
    "ffdur_w": ("longest waist |a| < 0.6 g run in the 1.5 s before the peak, s", 1.0),
    "gpk_w": ("waist |w| peak, -1.5..+0.5 s, rad/s", 1.0),
    "gpk_c": ("chest |w| peak, -1.5..+0.5 s, rad/s", 0.7),
    "tgpk_c": ("chest |w| peak time minus impact time, s", 0.5),
    "descent_s": ("10-90 % time of cumulative chest rotation above the pre-event baseline (-3..+1 s), s", 0.7),
    "rot_total": ("chest rotation above the pre-event baseline accumulated -3..+1 s, rad", 0.7),
    "dincl_w": ("waist inclination change (post - pre), deg", 1.0),
    "dincl_c": ("chest inclination change (post - pre), deg", 1.0),
    "post_incl_w": ("waist inclination 2-5 s after, deg", 1.0),
    "post_incl_c": ("chest inclination 2-5 s after, deg", 1.0),
    "npk_w": ("waist |a| peaks > 1.5 g in -0.5..+1.5 s (impact sequence)", 0.7),
    "rebound_w": ("waist |a| std 0.3-1.5 s after the peak (rebound/oscillation), g", 1.0),
    "settle_s": ("time until waist |a| within 0.15 g of 1 g and chest |w| < 0.5 rad/s for 0.5 s, s", 0.7),
    "still_acc_w": ("log10 waist |a| std 2.5-5 s after (post-fall stillness)", 1.0),
    "still_gyro_c": ("log10 chest |w| rms 2.5-5 s after", 1.0),
    "pre_std_w": ("waist |a| std -3..-1.5 s (pre-fall activity), g", 1.0),
    "jerk_w": ("waist peak |d|a|/dt| within 0.5 s of the peak, g/s", 1.0),
    # added after the baseline (computed from the stored traces, so earlier runs rescore):
    "postmin_w": ("waist |a| minimum 0.08-0.6 s after the peak (rebound flight reads < 1 g), g", 1.0),
    "gyro_post_c": ("chest |w| maximum 0.2-0.8 s after the peak (trunk rebound / second landing), rad/s", 1.0),
}
TRACE_FS = 25.0  # event traces are stored at the Erciyes rate


def trace_features(trace: np.ndarray, fs: float = TRACE_FS, before: float = 3.0) -> dict:
    """Features read from an event_trace (|a| waist, |a| chest, |w| chest; i0 at `before` s)."""
    i0 = int(before * fs)
    seg = lambda col, a, b: trace[i0 + int(round(a * fs)):i0 + int(round(b * fs)), col]
    with np.errstate(all="ignore"):
        return {"postmin_w": float(np.nanmin(seg(0, 0.08, 0.6))), "gyro_post_c": float(np.nanmax(seg(2, 0.2, 0.8)))}

ACT_BASE = ["astd_x", "astd_y", "astd_z", "amag_std", "amag_p95", "gstd_x", "gstd_y", "gstd_z", "gmag_mean",
            "incl_mean", "incl_std", "fdom", "bp_lo", "bp_mid", "bp_hi", "jerk", "period"]
ACT_RELEVANCE = {"amag_std": 1.0, "amag_p95": 1.0, "gmag_mean": 1.0, "incl_mean": 1.0, "incl_std": 0.7,
                 "gstd_x": 0.7, "gstd_y": 0.7, "gstd_z": 0.7, "jerk": 0.7}


def activity_feature_names() -> list[str]:
    return [f"{n}_{s}" for s in ("w", "c") for n in ACT_BASE] + ["incl_diff"]


def _mag(x):
    return np.linalg.norm(x, axis=-1)


def inclination_deg(acc: np.ndarray) -> np.ndarray:
    """Angle between the specific-force vector and segment +Y, deg (0 upright, 90 lying)."""
    a = acc / np.maximum(_mag(acc)[..., None], 1e-9)
    return np.degrees(np.arccos(np.clip(a[..., 1], -1, 1)))


def _sl(x, i0, a_s, b_s, fs):
    a = int(max(0, i0 + round(a_s * fs)))
    b = int(min(len(x), i0 + round(b_s * fs)))
    return x[a:max(a, b)]


def _longest(mask) -> int:
    best = run = 0
    for m in mask:
        run = run + 1 if m else 0
        best = max(best, run)
    return best


def find_event(acc: np.ndarray, fs: float, t0: float | None = None, t1: float | None = None) -> int:
    a = _mag(acc[:, 0])
    lo = 0 if t0 is None else max(int(t0 * fs), 0)
    hi = len(a) if t1 is None else min(int(t1 * fs), len(a))
    return lo + int(np.argmax(a[lo:hi]))


def event_features(acc: np.ndarray, gyro: np.ndarray, fs: float, i0: int) -> dict:
    a = _mag(acc) / G           # (N,2)
    w = _mag(gyro)              # (N,2)
    lp = signal.butter(2, 1.0, fs=fs, output="sos")
    incl = inclination_deg(signal.sosfiltfilt(lp, acc, axis=0)) if len(acc) > 15 else inclination_deg(acc)
    nan = float("nan")
    f = {}
    f["peak_w"] = float(a[i0, 0])
    f["peak_c"] = float(_sl(a[:, 1], i0, -0.3, 0.3, fs).max())
    pre = _sl(a[:, 0], i0, -1.0, 0.0, fs)
    f["ffmin_w"] = float(pre.min()) if len(pre) else nan
    pre15 = _sl(a[:, 0], i0, -1.5, 0.0, fs)
    f["ffdur_w"] = _longest(pre15 < 0.6) / fs if len(pre15) else nan
    f["gpk_w"] = float(_sl(w[:, 0], i0, -1.5, 0.5, fs).max())
    seg = _sl(w[:, 1], i0, -1.5, 0.5, fs)
    f["gpk_c"] = float(seg.max())
    start = max(0, i0 + int(round(-1.5 * fs)))
    f["tgpk_c"] = (start + int(np.argmax(seg)) - i0) / fs
    if i0 - 3 * fs >= 0 and i0 + 1 * fs <= len(a):
        # Rotation above the pre-event baseline (walking sway is removed), so a walking lead-in
        # and a standing lead-in measure the same fall rotation.
        base = float(np.median(_sl(w[:, 1], i0, -3.0, -1.5, fs)))
        wc = np.clip(_sl(w[:, 1], i0, -3.0, 1.0, fs) - base, 0.0, None)
        cum = np.cumsum(wc) / fs
        total = cum[-1]
        f["rot_total"] = float(total)
        if total > 1e-6:
            t10 = np.searchsorted(cum, 0.1 * total) / fs
            t90 = np.searchsorted(cum, 0.9 * total) / fs
            f["descent_s"] = float(t90 - t10)
        else:
            f["descent_s"] = nan
    else:
        f["rot_total"] = f["descent_s"] = nan
    ok_pre = i0 - 3 * fs >= 0
    ok_post = i0 + 5 * fs <= len(a)
    inc_pre = np.median(_sl(incl, i0, -3.0, -1.5, fs), axis=0) if ok_pre else np.full(2, nan)
    inc_post = np.median(_sl(incl, i0, 2.0, 5.0, fs), axis=0) if ok_post else np.full(2, nan)
    f["dincl_w"], f["dincl_c"] = float(inc_post[0] - inc_pre[0]), float(inc_post[1] - inc_pre[1])
    f["post_incl_w"], f["post_incl_c"] = float(inc_post[0]), float(inc_post[1])
    win = _sl(a[:, 0], i0, -0.5, 1.5, fs)
    peaks, _ = signal.find_peaks(win, height=1.5, distance=max(int(0.12 * fs), 1))
    f["npk_w"] = float(len(peaks))
    f["rebound_w"] = float(_sl(a[:, 0], i0, 0.3, 1.5, fs).std())
    quiet = (np.abs(a[:, 0] - 1.0) < 0.15) & (w[:, 1] < 0.5)
    need = int(0.5 * fs)
    settle = nan
    for j in range(i0 + 1, min(len(a) - need, i0 + int(5 * fs))):
        if quiet[j:j + need].all():
            settle = (j - i0) / fs
            break
    f["settle_s"] = settle if not np.isnan(settle) else (5.0 if ok_post else nan)
    f["still_acc_w"] = float(np.log10(_sl(a[:, 0], i0, 2.5, 5.0, fs).std() + 1e-3)) if ok_post else nan
    f["still_gyro_c"] = float(np.log10(np.sqrt(np.mean(_sl(w[:, 1], i0, 2.5, 5.0, fs) ** 2)) + 1e-3)) if ok_post else nan
    f["pre_std_w"] = float(_sl(a[:, 0], i0, -3.0, -1.5, fs).std()) if ok_pre else nan
    near = _sl(a[:, 0], i0, -0.5, 0.5, fs)
    f["jerk_w"] = float(np.abs(np.diff(near)).max() * fs) if len(near) > 1 else nan
    return f


def event_trace(acc: np.ndarray, gyro: np.ndarray, fs: float, i0: int, before: float = 3.0, after: float = 5.0):
    """Aligned |a| (both slots, g) and chest |w| around the event for the overlay plots (nan-padded)."""
    n0, n1 = int(before * fs), int(after * fs)
    out = np.full((n0 + n1, 3), np.nan)
    a = _mag(acc) / G
    w = _mag(gyro[:, 1])
    lo, hi = max(0, i0 - n0), min(len(a), i0 + n1)
    out[lo - (i0 - n0):hi - (i0 - n0), :2] = a[lo:hi]
    out[lo - (i0 - n0):hi - (i0 - n0), 2] = w[lo:hi]
    return out


def _slot_window(acc: np.ndarray, gyro: np.ndarray, fs: float) -> list[float]:
    a = _mag(acc) / G
    w = _mag(gyro)
    lp = signal.butter(2, min(1.0, fs / 4), fs=fs, output="sos")
    incl = inclination_deg(signal.sosfiltfilt(lp, acc, axis=0, padlen=min(len(acc) - 1, 15)))
    n = len(a)
    ad = a - a.mean()
    freqs, psd = signal.welch(ad, fs=fs, nperseg=min(n, int(4 * fs)))
    band = lambda lo, hi: float(np.log10(np.sqrt(np.trapezoid(psd[(freqs >= lo) & (freqs < hi)],
                                                              freqs[(freqs >= lo) & (freqs < hi)]) + 1e-12) + 1e-4))
    sel = (freqs >= 0.5) & (freqs <= 5.0)
    fdom = float(freqs[sel][np.argmax(psd[sel])]) if sel.any() else 0.0
    ac = np.correlate(ad, ad, mode="full")[n - 1:]
    ac = ac / max(ac[0], 1e-12)
    lo, hi = int(0.3 * fs), min(int(1.5 * fs), n - 1)
    period = float(ac[lo:hi].max()) if hi > lo else 0.0
    hi_band = (8.0, min(12.0, fs / 2 - 0.5)) if fs <= 30 else (8.0, 40.0)
    return [float(acc[:, 0].std() / G), float(acc[:, 1].std() / G), float(acc[:, 2].std() / G),
            float(a.std()), float(np.percentile(a, 95)),
            float(gyro[:, 0].std()), float(gyro[:, 1].std()), float(gyro[:, 2].std()), float(w.mean()),
            float(np.degrees(np.arccos(np.clip(acc[:, 1].mean() / max(_mag(acc.mean(0)), 1e-9), -1, 1)))),
            float(incl.std()), fdom, band(0.3, 3.0), band(3.0, 8.0), band(*hi_band),
            float(np.log10(np.std(np.diff(a)) * fs + 1e-4)), period]


def activity_windows(acc: np.ndarray, gyro: np.ndarray, fs: float, win_s: float = 5.0, hop_s: float = 2.5,
                     max_windows: int = 24) -> list[list[float]]:
    n = int(win_s * fs)
    hop = int(hop_s * fs)
    rows = []
    for start in range(0, len(acc) - n + 1, hop):
        a, g = acc[start:start + n], gyro[start:start + n]
        feats = _slot_window(a[:, 0], g[:, 0], fs) + _slot_window(a[:, 1], g[:, 1], fs)
        feats.append(feats[ACT_BASE.index("incl_mean") + len(ACT_BASE)] - feats[ACT_BASE.index("incl_mean")])
        rows.append(feats)
        if len(rows) >= max_windows:
            break
    return rows
