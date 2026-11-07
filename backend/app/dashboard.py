# File: dashboard.py
# Purpose: Map diagnostic samples to dashboard JSON with explicit availability.
# Contributors: Unassigned

from datetime import UTC, datetime
import time

from .models import REGIONS


def snapshot(stream, sequence: int, stale_after: float = 0.5) -> dict:
    sample = stream.latest
    age_ms = None if stream.received_monotonic is None else max(0, time.monotonic() - stream.received_monotonic) * 1000
    connected = stream.stats.state == "subscribed"
    current = connected and age_ms is not None and age_ms <= stale_after * 1000
    unavailable = {"status": "unavailable", "source": None, "received_at": None, "age_ms": None}
    imus = []
    by_id = {} if sample is None or sample.format != "quaternion" else {imu.sensor_id: imu for imu in sample.imus}
    for sensor_id, region in REGIONS.items():
        imu = by_id.get(sensor_id)
        valid = imu is not None and imu.valid
        status = "valid" if valid and current else "stale" if valid else "unavailable"
        freshness = dict(unavailable) if imu is None else {
            "status": status, "source": "hardware", "received_at": stream.received_at.isoformat(), "age_ms": age_ms,
        }
        imus.append({"region": region, "orientation": dict(zip("wxyz", imu.quaternion)) if valid and current else None,
                     "acquired_us": None, "freshness": freshness})
    return {
        "schema_version": "2.0", "timestamp": datetime.now(UTC).isoformat(), "sequence": sequence,
        "device": None,  # The device packets omit boot, frame and session metadata.
        "system_status": "degraded" if connected else "offline",
        "connection": {"source": "hardware", "esp32": "connected" if connected else "disconnected", "openbci": "disconnected"},
        "imus": imus,
        "posture": None, "fall_state": None, "heart_rate_bpm": None, "environment": None, "eeg": None, "overall_risk": None,
        "freshness": {key: dict(unavailable) for key in ("posture", "fall_state", "heart_rate", "environment", "eeg", "overall_risk")},
    }
