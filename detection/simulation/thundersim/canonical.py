# File: canonical.py
# Purpose: Stand-in canonicalize() step: decoded raw samples -> calibrated FrameBatch chunks.
#          Applies the product's neutral-pose calibration (gravity alignment per slot).
# Contributors: Unassigned
"""Canonical frames for models.

This mirrors what the backend will eventually do (see the report's "canonical record"):
deduplicate repeated device times, unwrap millis(), flag full-scale clipping, mark dead
slots, and rotate each slot so that +Y points along gravity during the neutral stand.
Replace it with the team's real preprocessing once that exists, keeping FrameBatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from .model_api import GRAVITY, FrameBatch

UP = np.array([0.0, 1.0, 0.0])


def neutral_alignment(mean_acc: np.ndarray) -> Rotation:
    """Shortest rotation taking the mean neutral-stand accelerometer vector onto +Y.

    Heading (rotation about +Y) is unobservable from gravity alone, so it is left as mounted.
    """
    a = np.asarray(mean_acc, float)
    a = a / np.linalg.norm(a)
    axis = np.cross(a, UP)
    sin_angle = np.linalg.norm(axis)
    cos_angle = float(np.dot(a, UP))
    if sin_angle < 1e-9:
        return Rotation.identity() if cos_angle > 0 else Rotation.from_rotvec([np.pi, 0, 0])
    return Rotation.from_rotvec(axis / sin_angle * np.arctan2(sin_angle, cos_angle))


@dataclass
class Canonicalizer:
    """Stateful: feed decoded raw samples in arrival order, collect FrameBatches.

    The first calibration_s seconds of the stream must be a quiet neutral stand
    (every scenario starts with one, like the product's neutral capture).
    """

    fs_hz: float
    accel_range_g: float
    gyro_range_dps: float
    calibration_s: float = 5.0
    batch_s: float = 0.5
    _times: list = field(default_factory=list)
    _acc: list = field(default_factory=list)
    _gyro: list = field(default_factory=list)
    _modes: list = field(default_factory=list)
    _last_ms: int | None = None
    _wraps: int = 0
    _first_t: float | None = None
    _align: list | None = None
    _pending_start: int = 0

    def push(self, time_ms: int, acc: np.ndarray, gyro: np.ndarray, work_mode: str = "normal") -> None:
        """Add one decoded sample: time_ms is the packet's uint32 millis(); acc/gyro (4,3)."""
        if self._last_ms is not None:
            if time_ms == self._last_ms:
                return  # repeated snapshot (e.g. a resend), not a new acquisition
            if time_ms < self._last_ms and self._last_ms - time_ms > 2**31:
                self._wraps += 1  # uint32 millis() wrapped (~49.7 days)
            elif time_ms < self._last_ms:
                return  # backwards time: reject, as the backend reader does
        self._last_ms = time_ms
        t = (time_ms + self._wraps * 2**32) / 1000.0
        if self._first_t is None:
            self._first_t = t
        self._times.append(t)
        self._acc.append(np.asarray(acc, float))
        self._gyro.append(np.asarray(gyro, float))
        self._modes.append(work_mode)

    def _calibrate(self) -> bool:
        if self._align is not None:
            return True
        if not self._times or self._times[-1] - self._first_t < self.calibration_s:
            return False
        acc = np.asarray(self._acc)
        times = np.asarray(self._times)
        quiet = acc[times - self._first_t <= self.calibration_s]
        self._align = []
        for slot in range(acc.shape[1]):
            norms = np.linalg.norm(quiet[:, slot], axis=1)
            alive = norms > 0
            mean = quiet[alive, slot].mean(axis=0) if alive.any() else UP * GRAVITY
            self._align.append(neutral_alignment(mean))
        return True

    def pop_batches(self, final: bool = False) -> list[FrameBatch]:
        """Return complete batches (or everything left when final=True)."""
        if not self._calibrate():
            return []
        batches = []
        while self._pending_start < len(self._times):
            start = self._pending_start
            t0 = self._times[start]
            end = start
            while end < len(self._times) and self._times[end] < t0 + self.batch_s and self._modes[end] == self._modes[start]:
                end += 1
            complete = end < len(self._times)
            if not complete and not final:
                break
            batches.append(self._make_batch(start, end))
            self._pending_start = end
        # Drop consumed history to keep memory bounded on long shifts.
        if self._pending_start > 10000:
            cut = self._pending_start
            del self._times[:cut], self._acc[:cut], self._gyro[:cut], self._modes[:cut]
            self._pending_start = 0
        return batches

    def _make_batch(self, start: int, end: int) -> FrameBatch:
        t = np.asarray(self._times[start:end])
        acc = np.asarray(self._acc[start:end])
        gyro = np.asarray(self._gyro[start:end])
        n = len(t)
        slot_mask = np.linalg.norm(acc, axis=2) > 0  # all-zero record = dead/unplugged sensor
        acc_limit = self.accel_range_g * GRAVITY * (32767 / 32768) - 1e-3
        gyro_limit = np.deg2rad(self.gyro_range_dps) * (32767 / 32768) - 1e-5
        sat = (np.abs(acc) >= acc_limit).any(axis=2) | (np.abs(gyro) >= gyro_limit).any(axis=2)
        acc_out = np.full((n, 4, 3), np.nan, np.float32)
        gyro_out = np.full((n, 4, 3), np.nan, np.float32)
        for slot in range(4):
            rows = slot_mask[:, slot]
            if rows.any():
                acc_out[rows, slot] = self._align[slot].apply(acc[rows, slot])
                gyro_out[rows, slot] = self._align[slot].apply(gyro[rows, slot])
        return FrameBatch(t, acc_out, gyro_out, slot_mask, sat & slot_mask, self.fs_hz, self._modes[start])
