# File: transport.py
# Purpose: Firmware-to-laptop path: encode each acquisition as the deployed raw ImuStripData
#          protobuf, pass it through a lossy BLE link model, and decode it with the backend's
#          own app.protocol.decode(). Simulated data never bypasses the real decoder unless
#          fast mode is requested explicitly.
# Contributors: Unassigned

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .site import RADIO, RadioEnv

BACKEND_DIR = Path(__file__).resolve().parents[3] / "backend"


def backend_modules():
    """Import the backend decoder and generated bindings (the code real packets go through)."""
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    from app import protocol  # noqa: E402  (path set above)
    from app.generated import telemetry_pb2  # noqa: E402

    return protocol, telemetry_pb2


class RawCodec:
    """Encode/decode firmware/proto/telemetry.proto ImuStripData exactly as deployed."""

    def __init__(self):
        self.protocol, pb = backend_modules()
        self.message = pb.ImuStripData()
        self.max_bytes = self.protocol.RAW_MAX_BYTES

    def encode(self, time_ms: int, acc: np.ndarray, gyro: np.ndarray) -> bytes:
        m = self.message
        m.Clear()
        m.time = int(time_ms) % (2**32)  # uint32 millis() wraps on the wire
        for i in range(acc.shape[0]):
            d = m.data.add()
            d.accel.x, d.accel.y, d.accel.z = (float(v) for v in acc[i])
            d.gyro.x, d.gyro.y, d.gyro.z = (float(v) for v in gyro[i])
        return m.SerializeToString()

    def decode(self, payload: bytes) -> tuple[int, np.ndarray, np.ndarray]:
        sample = self.protocol.decode(payload, "raw")
        acc = np.array([imu.accel for imu in sample.imus], float)
        gyro = np.array([imu.gyro for imu in sample.imus], float)
        return sample.sample_time_ms, acc, gyro


@dataclass
class LinkStats:
    sent: int = 0
    delivered: int = 0
    lost_radio: int = 0
    lost_outage: int = 0
    lost_throughput: int = 0
    rejected: int = 0
    bytes_sent: int = 0
    max_packet_bytes: int = 0
    latency_s: list = field(default_factory=list)

    def summary(self) -> dict:
        lat = np.array(self.latency_s) if self.latency_s else np.zeros(1)
        return {"sent": self.sent, "delivered": self.delivered, "lost_radio": self.lost_radio,
                "lost_outage": self.lost_outage, "lost_throughput": self.lost_throughput, "rejected": self.rejected,
                "loss_pct": round(100 * (1 - self.delivered / max(self.sent, 1)), 2),
                "max_packet_bytes": self.max_packet_bytes, "latency_ms_p50": round(1000 * float(np.median(lat)), 1),
                "latency_ms_p95": round(1000 * float(np.percentile(lat, 95)), 1)}


class BleLink:
    """Gilbert-Elliott loss, rare multi-packet outages, a throughput cap and latency."""

    def __init__(self, env: RadioEnv | str, seed: int = 0):
        self.env = RADIO[env] if isinstance(env, str) else env
        self.rng = np.random.default_rng(seed)
        self.bad = False
        self.tokens = 0.0
        self.last_t = None
        self.outage_until = -1.0
        self.next_outage = self._draw_outage(0.0)
        self.stats = LinkStats()

    def _draw_outage(self, now: float) -> float:
        rate = self.env.gap_rate_per_s
        return now + (self.rng.exponential(1 / rate) if rate > 0 else np.inf)

    def send(self, t: float, nbytes: int) -> float | None:
        """Return arrival time, or None if the notification is lost."""
        s = self.stats
        s.sent += 1
        s.bytes_sent += nbytes
        s.max_packet_bytes = max(s.max_packet_bytes, nbytes)
        if t >= self.next_outage:
            self.outage_until = t + self.rng.uniform(0.05, 0.30)  # SIM_SPEC 2: 50-300 ms gaps
            self.next_outage = self._draw_outage(t)
        if t < self.outage_until:
            s.lost_outage += 1
            return None
        dt = 0.0 if self.last_t is None else max(t - self.last_t, 0.0)
        self.last_t = t
        cap = 4 * nbytes  # small notification queue
        self.tokens = min(self.tokens + dt * self.env.throughput_bytes_s, cap) if s.sent > 1 else cap
        if self.tokens < nbytes:
            s.lost_throughput += 1
            return None
        self.tokens -= nbytes
        if self.bad:
            self.bad = self.rng.random() >= self.env.p_bad_to_good
        else:
            self.bad = self.rng.random() < self.env.p_good_to_bad
        if self.bad and self.rng.random() < self.env.loss_bad:
            s.lost_radio += 1
            return None
        latency = self.rng.uniform(*self.env.latency_s)
        s.delivered += 1
        s.latency_s.append(latency)
        return t + latency
