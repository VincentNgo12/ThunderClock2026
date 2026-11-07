"""Heading-free tilt (inclination) estimator, one per anchor.

We only need "how far is this segment tilted away from upright", which depends on
gravity alone, so we track the unit *up* vector expressed in the segment frame
(no yaw, so no magnetometer and no heading drift problems).

    gyro step:   a world-fixed vector seen from a rotating body changes as
                 d(up_b)/dt = -omega x up_b
    accel step:  when |a| is close to 1 g, the accelerometer points (mostly) up,
                 so pull up_b a little towards a/|a|   (complementary filter)

Inclination = angle between up_b and the segment's own +Y axis:
    0 deg  upright neutral,  90 deg  lying (any direction),  180 deg  upside down.

Vectorised over any leading shape, e.g. (4, 3) for one strip or (K, 4, 3) for K
recordings processed in lock-step. ESP32 port: the same 10 lines per anchor.
"""
import numpy as np

from .constants import DT_S, G

ACC_GAIN_PER_S = 2.0      # correction rate towards the accelerometer, 1/s (time constant 0.5 s)
ACC_GATE_G = 0.15         # only trust the accelerometer when | |a| - 1 g | < 0.15 g


class TiltFilter:
    def __init__(self, gain_per_s=ACC_GAIN_PER_S, gate_g=ACC_GATE_G, dt_s=DT_S):
        self.k = gain_per_s * dt_s
        self.gate = gate_g * G
        self.dt = dt_s
        self.up = None  # (..., 3) unit vectors, segment frame

    def reset(self, acc):
        """Initialise from accelerometer only (assumes the wearer is momentarily still)."""
        n = np.linalg.norm(acc, axis=-1, keepdims=True)
        self.up = np.where(n > 1e-6, acc / np.maximum(n, 1e-6), np.array([0.0, 1.0, 0.0]))

    def step(self, acc, gyro):
        """acc m/s^2, gyro rad/s, both (..., 3). Returns the updated up vector."""
        if self.up is None:
            self.reset(acc)
            return self.up
        up = self.up
        # 1) rotate with the gyro (first-order integration is fine at 100 Hz)
        up = up - np.cross(gyro, up) * self.dt
        # 2) nudge towards the measured gravity direction when it is trustworthy
        n = np.linalg.norm(acc, axis=-1, keepdims=True)
        ok = np.abs(n - G) < self.gate
        target = acc / np.maximum(n, 1e-6)
        up = np.where(ok, up + self.k * (target - up), up)
        self.up = up / np.linalg.norm(up, axis=-1, keepdims=True)
        return self.up


def run_tilt(acc, gyro, filt=None):
    """Batch helper. acc, gyro: (N, ..., 3) time-first arrays. Returns up: (N, ..., 3)."""
    filt = filt or TiltFilter()
    # initialise from the mean of the first 0.25 s
    filt.reset(acc[: max(1, int(0.25 / filt.dt))].mean(axis=0))
    out = np.empty_like(acc, dtype=np.float64)
    for i in range(acc.shape[0]):
        out[i] = filt.step(acc[i], gyro[i])
    return out


def inclination_deg(up):
    """Angle between segment +Y and true up, degrees. up: (..., 3) unit vectors."""
    return np.degrees(np.arccos(np.clip(up[..., 1], -1.0, 1.0)))
