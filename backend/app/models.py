# File: models.py
# Purpose: Small decoded samples and local receipt diagnostics, independent of the dashboard.
# Contributors: Unassigned

from dataclasses import asdict, dataclass
import time
from typing import Literal

REGIONS = {1: "pelvis", 2: "lumbar", 3: "thoracic", 4: "upper_thoracic"}
STALE_AFTER_SECONDS = 0.5
STREAMING_STATES = ("subscribed", "simulated")


@dataclass(frozen=True)
class Imu:
    sensor_id: int
    accel: tuple[float, float, float] | None = None  # m/s², includes gravity
    gyro: tuple[float, float, float] | None = None  # rad/s
    quaternion: tuple[float, float, float, float] | None = None  # w,x,y,z
    valid: bool = True


@dataclass(frozen=True)
class Environment:
    temperature_c: float | None = None
    humidity_percent: float | None = None
    air_quality_index: int | None = None
    eco2_ppm: int | None = None
    tvoc_ppb: int | None = None
    noise_relative_db: float | None = None


@dataclass(frozen=True)
class Sample:
    format: Literal["raw", "quaternion"]
    sample_time_ms: int  # Device uptime; never UTC or radio-transit age.
    imus: tuple[Imu, ...]
    sequence: int | None = None
    environment: Environment | None = None
    heart_rate_bpm: float | None = None
    spo2_percent: float | None = None

    @property
    def fully_usable(self) -> bool:
        # Optional readings do not affect IMU diagnostic health.
        return all(imu.valid for imu in self.imus)


@dataclass
class Diagnostics:
    state: str = "disconnected"
    received: int = 0
    decoded: int = 0  # Structurally valid frames, possibly with unavailable orientations.
    usable: int = 0  # Fully usable frames; raw means structural validity, not sensor health.
    rejected: int = 0
    dropped: int = 0
    sequence_gaps: int = 0
    resets: int = 0
    started_at: float = 0  # Local monotonic seconds, not UTC.
    finished_at: float | None = None
    last_notification: float | None = None
    last_valid: float | None = None  # Fully usable frame, not merely a notification.
    last_error: str | None = None

    def age(self, now: float | None = None) -> float | None:
        return None if self.last_valid is None else (time.monotonic() if now is None else now) - self.last_valid

    def snapshot(self) -> dict:
        now = time.monotonic()
        result = asdict(self)
        result["last_valid_age_ms"] = None if self.last_valid is None else round((now - self.last_valid) * 1000, 1)
        result["valid_frame_hz"] = round(self.usable / max((self.finished_at or now) - self.started_at, 0.001), 1) if self.started_at else 0
        return result
