# File: intake.py
# Purpose: RawFrames (DETECTION_ARCHITECTURE.md 3.1-3.2, 3.9): decoded backend raw samples ->
#          FrameBatch-like arrays for MainModel. Unwraps the uint32 ms clock, drops duplicates,
#          applies the mounting remap and the neutral gravity alignment, marks dead slots.
# Contributors: Unassigned
"""Input: backend `Sample` objects (backend/app/models.py at f3337e3) or any object with
`sample_time_ms` and `imus` (each with `sensor_id`, `accel` m/s^2 incl. gravity, `gyro` rad/s,
in board axes). Output of pop(): (t_s, acc, gyro, slot_mask) with t in device seconds,
(N, 4, 3) arrays in the segment frame (+X right, +Y up, +Z posterior), slot order 1..4.

* Clock: uint32 milliseconds since boot. A jump back of more than 2^31 ms is a wrap; any other
  backward jump is passed through unchanged so MainModel can detect a device restart
  (time back > 1 s). An identical time is a duplicate and is dropped.
* Mounting remap: the RAW_IMU_AXES format of backend/app/fusion.py ("-y,z,-x" per board, four
  maps separated by ';'); default -y,z,-x like the backend.
* Dead slot: an all-zero accel+gyro record (or a missing / invalid IMU) -> slot_mask False, NaN.
* Gravity alignment: per slot, the shortest rotation taking the mean specific force of the first
  `still_s` of quiet standing onto +Y (same as thundersim/canonical.py, 0.15 g tolerance).
  Until aligned, frames are held back and `calibrated` is False ("uncalibrated").
"""
import math

import numpy as np

G = 9.80665
SIGNED = {"x": 1, "y": 2, "z": 3, "-x": -1, "-y": -2, "-z": -3}


def parse_axes(value):
    groups = ["-y,z,-x"] * 4 if not value else value.split(";")
    if len(groups) != 4:
        raise ValueError("need four axis maps")
    maps = []
    for group in groups:
        axes = tuple(SIGNED[a.strip()] for a in group.split(","))
        if sorted(abs(a) for a in axes) != [1, 2, 3]:
            raise ValueError("each map must use x, y and z once")
        maps.append(axes)
    return maps


def align_matrix(mean_acc):
    """3x3 rotation taking mean_acc onto +Y (heading left as mounted)."""
    a = np.asarray(mean_acc, float)
    a = a / np.linalg.norm(a)
    up = np.array([0.0, 1.0, 0.0])
    axis = np.cross(a, up)
    s, c = np.linalg.norm(axis), float(np.dot(a, up))
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = axis / s
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    angle = math.atan2(s, c)
    return np.eye(3) + math.sin(angle) * kx + (1 - math.cos(angle)) * kx @ kx


class RawFrames:
    def __init__(self, axes=None, still_s=2.0, align=True):
        self.maps = parse_axes(axes)
        self.still_s = still_s
        self.align = align
        self.reset()

    def reset(self):
        self.last_ms = None
        self.wraps = 0
        self.duplicates = 0
        self.rows = []                      # pending (t, acc, gyro, mask)
        self.calibrate()

    def calibrate(self):
        """Forget the alignment; it is redone on the next still_s of quiet standing."""
        self.rot = None if self.align else [np.eye(3)] * 4
        self.quiet = []

    @property
    def calibrated(self):
        return self.rot is not None

    def _remap(self, vec, axes):
        return [vec[a - 1] if a > 0 else -vec[-a - 1] for a in axes]

    def push(self, sample):
        """One decoded raw sample."""
        ms = int(sample.sample_time_ms) & 0xFFFFFFFF
        if self.last_ms is not None:
            if ms == self.last_ms:
                self.duplicates += 1
                return
            if self.last_ms - ms > 2 ** 31:
                self.wraps += 1
            elif self.last_ms - ms > 1000:
                self.quiet = []             # device restart while learning: those samples are another boot's
        self.last_ms = ms
        t = (ms + self.wraps * 2 ** 32) / 1000.0
        acc = np.full((4, 3), np.nan)
        gyro = np.full((4, 3), np.nan)
        mask = np.zeros(4, bool)
        for imu in sample.imus:
            i = int(imu.sensor_id) - 1
            if not 0 <= i < 4 or imu.accel is None or imu.gyro is None or not getattr(imu, "valid", True):
                continue
            if not any(imu.accel) and not any(imu.gyro):
                continue                    # all-zero record = dead sensor (B5)
            acc[i] = self._remap(imu.accel, self.maps[i])
            gyro[i] = self._remap(imu.gyro, self.maps[i])
            mask[i] = True
        if self.rot is None:
            self._learn(t, acc, gyro, mask)
            return
        for i in range(4):
            if mask[i]:
                acc[i] = self.rot[i] @ acc[i]
                gyro[i] = self.rot[i] @ gyro[i]
        self.rows.append((t, acc, gyro, mask))

    def _learn(self, t, acc, gyro, mask):
        quiet = mask.sum() >= 3 and all(abs(np.linalg.norm(acc[i]) - G) < 0.15 * G and np.linalg.norm(gyro[i]) < 0.15
                                        for i in range(4) if mask[i])
        if not quiet:
            self.quiet = []
            return
        self.quiet.append((t, acc.copy(), mask.copy()))
        if self.quiet[-1][0] - self.quiet[0][0] >= self.still_s:
            rot = []
            for i in range(4):
                rows = [a[i] for _, a, m in self.quiet if m[i]]
                rot.append(align_matrix(np.mean(rows, axis=0)) if rows else np.eye(3))
            self.rot = rot
            self.quiet = []

    def pop(self):
        """Return (t, acc, gyro, slot_mask) arrays for everything pushed since the last pop."""
        if not self.rows:
            return None
        t = np.array([r[0] for r in self.rows])
        acc = np.array([r[1] for r in self.rows])
        gyro = np.array([r[2] for r in self.rows])
        mask = np.array([r[3] for r in self.rows])
        self.rows = []
        return t, acc, gyro, mask
