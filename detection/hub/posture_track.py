# File: posture_track.py
# Purpose: Per-sample inclination and stillness for the topple trigger, collapse confirmer and
#          man-down timer (DETECTION_ARCHITECTURE.md 3.2, 3.5, 3.6). A detection feature
#          computation, not posture fusion: the dashboard posture stays on firmware quaternions.
# Contributors: Unassigned
"""PostureTrack: one heading-free tilt filter per anchor, run with the REAL sample interval.

falldet's TiltFilter assumes dt = 0.01 s; at 50 Hz that halves every gyro step. Here each
step uses the measured dt (clamped to 0.1 s), so 50 Hz and 20 Hz input integrate correctly.

Per sample (all scalar, ESP32-portable):
    incl[i]   angle between anchor +Y and true up, degrees (None for a dead slot)
    down      >= 3 live anchors > down_deg and pelvis (lumbar if the pelvis is dead) > down_pelvis_deg
    not_down  >= 3 live anchors < down_deg
    upright   >= 3 live anchors < upright_deg
    still     every live anchor: |w - w_lp| < still_gyro_rad_s and |a - a_lp| < still_acc_g, where
              w_lp, a_lp are still_lp_tau_s (0.5 s) low-pass (EMA) vectors, plus loose absolute bounds
              (|w| < still_gyro_abs_rad_s, ||a| - g| < still_acc_abs_g). Deviation from the
              low-pass makes stillness insensitive to gyro bias (MPU-6050 zero-rate offset up to
              20 dps uncalibrated) and accel bias/scale errors.
    rot_pl    max(|w_pelvis|, |w_lumbar|) (live ones; the remaining anchors if both are dead)
With fewer than 3 live anchors, down / not_down / upright are all False (unknown).

One-second bins (fixed memory, 12 bins) give the window statistics C3 and T4 use:
still fraction, down fraction, coverage, median inclination and a "no motion" flag
(acc bin means of every live anchor vary by < no_motion_acc_g: strip on a table, or a
body with no breathing - which must escalate, never suppress).
"""
import math
from collections import deque

G = 9.80665
N_BINS = 12


class Bin:
    __slots__ = ("index", "n", "n_still", "n_quiet", "n_down", "n_upright", "incl_sum", "incl_n", "acc_sum", "acc_n")

    def __init__(self, index):
        self.index = index            # floor(t / 1 s) of internal time
        self.n = self.n_still = self.n_quiet = self.n_down = self.n_upright = 0
        self.incl_sum = [0.0] * 4
        self.incl_n = [0] * 4
        self.acc_sum = [[0.0, 0.0, 0.0] for _ in range(4)]
        self.acc_n = [0] * 4

    @property
    def start(self):
        return float(self.index)

    @property
    def end(self):
        return float(self.index + 1)


class PostureTrack:
    def __init__(self, cfg):
        self.cfg = cfg
        self.gain = cfg["acc_gain_per_s"]
        self.gate = cfg["acc_gate_g"] * G
        self.bins = deque(maxlen=N_BINS)
        self.current = None
        self.reset()

    def reset(self):
        """Full reset (new stream)."""
        self.reset_tilt()
        self.bins.clear()
        self.current = None
        self.last_t = None
        self.upright_since = None
        self.peak_times = deque(maxlen=12)
        self.gait_armed = True

    def reset_tilt(self):
        """After a gap the gyro integration cannot be bridged: re-seed from the accelerometer."""
        self.up = [None, None, None, None]
        self.w_lp = [None, None, None, None]
        self.a_lp = [None, None, None, None]
        self.amag_lp = None

    # ------------------------------------------------------------------ per sample
    def _tilt(self, i, a, w, dt):
        ax, ay, az = a
        n = math.sqrt(ax * ax + ay * ay + az * az)
        up = self.up[i]
        if up is None:
            if n < 1e-6:
                return None
            self.up[i] = [ax / n, ay / n, az / n]
            return self.up[i]
        ux, uy, uz = up
        wx, wy, wz = w
        # d(up)/dt = -w x up (cross product from the old vector)
        cx, cy, cz = wy * uz - wz * uy, wz * ux - wx * uz, wx * uy - wy * ux
        ux, uy, uz = ux - cx * dt, uy - cy * dt, uz - cz * dt
        if abs(n - G) < self.gate:
            k = min(self.gain * dt, 1.0)
            ux += k * (ax / n - ux)
            uy += k * (ay / n - uy)
            uz += k * (az / n - uz)
        m = math.sqrt(ux * ux + uy * uy + uz * uz) or 1.0
        self.up[i] = [ux / m, uy / m, uz / m]
        return self.up[i]

    def step(self, t, acc, gyro, valid):
        """t: internal seconds; acc, gyro: 4 x (x, y, z) m/s^2, rad/s; valid: 4 bools.
        Returns (info dict, closed Bin list)."""
        cfg = self.cfg
        dt = 0.01 if self.last_t is None else min(max(t - self.last_t, 0.0), 0.1)
        self.last_t = t
        incl = [None, None, None, None]
        wmag = [0.0, 0.0, 0.0, 0.0]
        still = True
        quiet = True     # strict, breathing-only stillness (lying_work timer)
        alpha = min(dt / cfg["still_lp_tau_s"], 1.0)
        for i in range(4):
            if not valid[i]:
                self.up[i] = self.w_lp[i] = self.a_lp[i] = None
                continue
            a, w = acc[i], gyro[i]
            up = self._tilt(i, a, w, dt)
            if up is None:
                continue
            incl[i] = math.degrees(math.acos(max(-1.0, min(1.0, up[1]))))
            wm = math.sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2])
            am = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
            wmag[i] = wm
            wl, al = self.w_lp[i], self.a_lp[i]
            if wl is None:
                wl, al = list(w), list(a)
            else:
                wl = [wl[k] + alpha * (w[k] - wl[k]) for k in range(3)]
                al = [al[k] + alpha * (a[k] - al[k]) for k in range(3)]
            self.w_lp[i], self.a_lp[i] = wl, al
            dw = math.sqrt(sum((w[k] - wl[k]) ** 2 for k in range(3)))
            da = math.sqrt(sum((a[k] - al[k]) ** 2 for k in range(3)))
            if (dw >= cfg["still_gyro_rad_s"] or da >= cfg["still_acc_g"] * G
                    or wm >= cfg["still_gyro_abs_rad_s"] or abs(am - G) >= cfg["still_acc_abs_g"] * G):
                still = False
            if dw >= cfg["quiet_gyro_rad_s"] or da >= cfg["quiet_acc_g"] * G:
                quiet = False
        live = [x for x in incl if x is not None]
        n_live = len(live)
        k = cfg["min_anchors"]
        if n_live >= k:
            pelvis = incl[0] if incl[0] is not None else incl[1]
            down = sum(x > cfg["down_deg"] for x in live) >= k and pelvis is not None and pelvis > cfg["down_pelvis_deg"]
            not_down = sum(x < cfg["down_deg"] for x in live) >= k
            upright = sum(x < cfg["upright_deg"] for x in live) >= k
        else:
            down = not_down = upright = False
            still = False if n_live == 0 else still
        quiet = quiet and still
        pl = [wmag[i] for i in (0, 1) if incl[i] is not None]
        rot_pl = max(pl) if pl else max(wmag)
        # upright hold (C3 return-to-upright, T4 recovery, arbiter cleared)
        if upright:
            if self.upright_since is None:
                self.upright_since = t
        else:
            self.upright_since = None
        self._gait(t, acc, incl)
        closed = self._bin(t, incl, acc, valid, still, down, upright, quiet)
        info = {"t": t, "incl": incl, "n_live": n_live, "down": down, "not_down": not_down,
                "upright": upright, "still": still, "rot_pl": rot_pl}
        return info, closed

    def upright_for(self, t):
        return 0.0 if self.upright_since is None else t - self.upright_since

    def _gait(self, t, acc, incl):
        """Heel-strike peaks of the pelvis (lumbar if dead) specific-force magnitude."""
        slot = 0 if incl[0] is not None else (1 if incl[1] is not None else None)
        if slot is None:
            return
        a = acc[slot]
        g = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) / G
        # relative to a slow mean, so an accel scale/bias error does not fake or hide steps
        self.amag_lp = g if self.amag_lp is None else self.amag_lp + 0.01 * (g - self.amag_lp)
        rel = g - self.amag_lp + 1.0
        if self.gait_armed and rel > self.cfg["gait_peak_g"]:
            self.peak_times.append(t)
            self.gait_armed = False
        elif rel < 1.0:
            self.gait_armed = True

    def gait(self, t, span_s=5.0):
        """True if >= gait_min_steps step intervals of 0.3-1.0 s occurred in the last span_s."""
        cfg = self.cfg
        times = [p for p in self.peak_times if p >= t - span_s]
        good = sum(cfg["gait_min_interval_s"] <= b - a <= cfg["gait_max_interval_s"] for a, b in zip(times, times[1:]))
        return good >= cfg["gait_min_steps"]

    # ------------------------------------------------------------------ 1 s bins
    def _bin(self, t, incl, acc, valid, still, down, upright, quiet=False):
        index = int(math.floor(t))
        closed = []
        if self.current is None:
            self.current = Bin(index)
        elif index != self.current.index:
            closed.append(self.current)
            self.bins.append(self.current)
            # empty bins for a gap (bounded by the ring size)
            for missing in range(max(self.current.index + 1, index - N_BINS), index):
                empty = Bin(missing)
                closed.append(empty)
                self.bins.append(empty)
            self.current = Bin(index)
        b = self.current
        b.n += 1
        b.n_still += still
        b.n_quiet += quiet
        b.n_down += down
        b.n_upright += upright
        for i in range(4):
            if incl[i] is not None:
                b.incl_sum[i] += incl[i]
                b.incl_n[i] += 1
                s = b.acc_sum[i]
                s[0] += acc[i][0]
                s[1] += acc[i][1]
                s[2] += acc[i][2]
                b.acc_n[i] += 1
        return closed

    def window(self, n_bins, since=None):
        """Statistics of the last n_bins closed bins (optionally only bins starting >= since)."""
        bins = [b for b in list(self.bins)[-n_bins:] if since is None or b.start >= since - 1e-9]
        n = sum(b.n for b in bins)
        stats = {"bins": len(bins), "n": n, "coverage": (sum(b.n > 0 for b in bins) / n_bins) if n_bins else 0.0,
                 "still_frac": (sum(b.n_still for b in bins) / n) if n else 0.0,
                 "down_frac": (sum(b.n_down for b in bins) / n) if n else 0.0,
                 "upright_frac": (sum(b.n_upright for b in bins) / n) if n else 0.0}
        medians = []
        for i in range(4):
            means = sorted(b.incl_sum[i] / b.incl_n[i] for b in bins if b.incl_n[i])
            medians.append(round(means[len(means) // 2], 1) if means else None)
        stats["incl_median_deg"] = medians
        stats["no_motion"] = self._no_motion(bins)
        return stats

    def _no_motion(self, bins):
        limit = self.cfg["no_motion_acc_g"] * G
        live = False
        for i in range(4):
            means = [[s / b.acc_n[i] for s in b.acc_sum[i]] for b in bins if b.acc_n[i]]
            if len(means) < 5:
                continue
            live = True
            for axis in range(3):
                values = [m[axis] for m in means]
                if max(values) - min(values) >= limit:
                    return False
        return live
