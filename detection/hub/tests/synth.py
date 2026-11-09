# File: synth.py
# Purpose: Synthetic 4-anchor IMU streams for the hub unit tests: each anchor pitches about its
#          +X axis by a scripted angle, so inclination, gyro and gravity are exactly consistent.
# Contributors: Unassigned
"""Segment frame: up = (0, cos th, sin th); still specific force = g * up; gyro = (-dth/dt, 0, 0)
(this sign matches d(up)/dt = -w x up used by the tilt filters)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DETECTION = os.path.dirname(os.path.dirname(HERE))
for p in (DETECTION, os.path.join(DETECTION, "models"), os.path.join(DETECTION, "simulation")):
    if p not in sys.path:
        sys.path.insert(0, p)

G = 9.80665


def minjerk(u):
    u = np.clip(u, 0.0, 1.0)
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


class Script:
    """Piecewise angle script, degrees per anchor. Build with .hold() / .move() / .jerk()."""

    def __init__(self, start_deg=(0, 0, 0, 0)):
        self.t = [0.0]
        self.angle = [np.array(start_deg, float)]
        self.parts = []          # (t0, t1, a0, a1, kind)

    @property
    def now(self):
        return self.t[-1]

    @property
    def last(self):
        return self.angle[-1]

    def hold(self, seconds):
        self.parts.append((self.now, self.now + seconds, self.last, self.last, "hold"))
        self.t.append(self.now + seconds)
        self.angle.append(self.last)
        return self

    def move(self, seconds, target_deg):
        target = np.array(target_deg, float)
        self.parts.append((self.now, self.now + seconds, self.last, target, "move"))
        self.t.append(self.now + seconds)
        self.angle.append(target)
        return self

    def jerk(self, seconds=0.2, amp_deg=15.0):
        """Out-and-back bump (brief movement) that ends where it started."""
        self.parts.append((self.now, self.now + seconds, self.last, self.last + amp_deg, "bump"))
        self.t.append(self.now + seconds)
        self.angle.append(self.last)
        return self

    def angles(self, t):
        out = np.zeros((len(t), 4))
        for t0, t1, a0, a1, kind in self.parts:
            sel = (t >= t0) & (t < t1 + 1e-12)
            u = (t[sel] - t0) / max(t1 - t0, 1e-9)
            if kind == "hold":
                out[sel] = a0
            elif kind == "move":
                out[sel] = a0 + (a1 - a0) * minjerk(u)[:, None]
            else:
                out[sel] = a0 + (a1 - a0) * (np.sin(np.pi * u) ** 2)[:, None]
        out[t >= self.now] = self.last
        return out


def stream(script, fs=100.0, breathing=True, noise=0.0, seed=0, t0=0.0, accel_scale=1.0):
    """Returns t (N,), acc (N,4,3), gyro (N,4,3), angles (N,4) degrees."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, script.now, 1.0 / fs)
    th_deg = script.angles(t)
    if breathing:
        th_deg = th_deg + 0.6 * np.sin(2 * np.pi * 0.25 * t)[:, None]
    th = np.radians(th_deg)
    # analytic derivative from a fine finite difference (smooth scripts)
    dt = 1e-4
    th2 = np.radians(script.angles(t + dt) + (0.6 * np.sin(2 * np.pi * 0.25 * (t + dt)))[:, None] * breathing)
    w = (th2 - th) / dt
    acc = np.zeros((len(t), 4, 3))
    acc[..., 1] = G * np.cos(th) * accel_scale
    acc[..., 2] = G * np.sin(th) * accel_scale
    gyro = np.zeros((len(t), 4, 3))
    gyro[..., 0] = -w
    if noise:
        acc += rng.normal(0, noise * G, acc.shape)
        gyro += rng.normal(0, noise, gyro.shape)
    return t + t0, acc, gyro, th_deg


def feed(model, t, acc, gyro, batch_n=50, slot_mask=None, work_mode="normal"):
    """Feed a stream in chunks of batch_n frames (index order, so restarts work); returns [(event, batch_end)]."""
    out = []
    for a in range(0, len(t), batch_n):
        sel = slice(a, a + batch_n)
        mask = None if slot_mask is None else slot_mask[sel]
        for e in model.update_arrays(t[sel], acc[sel], gyro[sel], mask, None, work_mode):
            out.append((e, float(t[sel][-1])))
    return out


def topple_script(pre_s=10.0, fall_s=0.6, after_s=40.0, end=(90, 90, 95, 100)):
    return Script().hold(pre_s).move(fall_s, end).hold(after_s)
