# File: dashboard.py
# Purpose: Map diagnostic samples to dashboard JSON with explicit availability.
# Contributors: Unassigned

from dataclasses import asdict
from datetime import UTC, datetime
import time

from .models import REGIONS, STALE_AFTER_SECONDS, STREAMING_STATES


def snapshot(stream, sequence: int, stale_after: float = STALE_AFTER_SECONDS, *, source="hardware", posture=None) -> dict:
    sample = stream.latest
    age_ms = None if stream.received_monotonic is None else max(0, time.monotonic() - stream.received_monotonic) * 1000
    connected = stream.stats.state in STREAMING_STATES
    current = connected and age_ms is not None and age_ms <= stale_after * 1000
    unavailable = {"status": "unavailable", "source": None, "received_at": None, "age_ms": None}
    receipt = {
        "source": source,
        "received_at": None if stream.received_at is None else stream.received_at.isoformat(),
        "age_ms": age_ms,
    }
    imus = []
    by_id = {} if sample is None or sample.format != "quaternion" else {imu.sensor_id: imu for imu in sample.imus}
    for sensor_id, region in REGIONS.items():
        imu = by_id.get(sensor_id)
        valid = imu is not None and imu.valid and imu.quaternion is not None
        status = "valid" if valid and current else "stale" if valid else "unavailable"
        freshness = dict(unavailable) if imu is None else {"status": status, **receipt}
        imus.append({"region": region, "orientation": dict(zip("wxyz", imu.quaternion)) if valid and current else None,
                     "acquired_us": None, "freshness": freshness})
    environment = None if sample is None else sample.environment
    freshness = {key: dict(unavailable) for key in ("posture", "fall_state", "heart_rate", "environment", "eeg", "overall_risk")}
    if environment is not None:
        freshness["environment"] = {"status": "valid" if current else "stale", **receipt}
    pose = None
    if posture is not None and posture.angles is not None:
        freshness["posture"] = {"status": "valid" if current else "stale", **receipt}
        if current:
            pose = {
                **posture.angles,
                "joint_rotations": {name: dict(zip("wxyz", q)) for name, q in posture.joint_deltas.items()},
            }
    esp32 = "disconnected"
    if connected:
        esp32 = {"simulated": "simulated", "replay": "replay"}.get(source, "connected")
    return {
        "schema_version": "2.0", "timestamp": datetime.now(UTC).isoformat(), "sequence": sequence,
        "device": None,  # The device packets omit boot, frame and session metadata.
        "system_status": "degraded" if connected else "offline",
        "connection": {"source": source, "esp32": esp32, "openbci": "disconnected"},
        "imus": imus,
        "posture": pose,
        "calibration": posture.calibration if posture is not None else {
            "status": "uncalibrated", "captured_at": None, "reference": None,
        },
        "alerts": posture.alerts if current and posture is not None else [],
        "fall_state": None, "heart_rate_bpm": None, "eeg": None, "overall_risk": None,
        "environment": asdict(environment) if current and environment is not None else None,
        "freshness": freshness,
    }
