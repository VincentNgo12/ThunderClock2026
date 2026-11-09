# File: recording.py
# Purpose: Labelled data collection sessions: every decoded sample (quaternion and raw, at the rate the
#          device sends them) plus the movement labels the dashboard's "Collect data" wizard sets.
# Contributors: Unassigned
"""One session = one JSON Lines file in RECORDINGS_DIR (default backend/recordings/, gitignored).

Lines, in time order (t = seconds since the session started, laptop monotonic clock):
  {"type": "meta", ...}                         first line: session, source, device format, client info
  {"type": "label", "t", "name"}                the wizard's current step, e.g. "forward:hold"
  {"type": "sample", "t", "origin", "format", "sample_time_ms", "sequence", "imus": [...]}
      origin: "device" (as decoded from BLE / the simulator) or "laptop_fusion" (raw_fusion output)
      imus:   {"id", "valid", "q": [w, x, y, z] | null, "a": [x, y, z] m/s^2 | null, "g": [x, y, z] rad/s | null}
  {"type": "end", "t", "samples", "labels", "aborted"}
Samples are buffered in memory (a few MB for a few minutes) and written when the session stops,
so the acquisition loop never waits on the disk.
"""
from datetime import UTC, datetime
import json
import os
import pathlib
import re
import time

DEFAULT_DIR = pathlib.Path(__file__).resolve().parent.parent / "recordings"
MAX_SAMPLES = 200_000          # ~30 min at 100 Hz raw + 20 Hz quaternions; protects memory
SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _vec(values):
    return None if values is None else [round(float(v), 6) for v in values]


class Recorder:
    def __init__(self, name: str, meta: dict):
        self.started = time.monotonic()
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        clean = SAFE_NAME.sub("_", name).strip("_")[:60] or "session"
        self.session = f"{stamp}_{clean}"
        self.meta = {"type": "meta", "session": self.session, "started_at": datetime.now(UTC).isoformat(), **meta}
        self.lines = []
        self.samples = 0
        self.labels = 0
        self.label = None
        self.truncated = False

    def elapsed(self) -> float:
        return round(time.monotonic() - self.started, 4)

    def set_label(self, name: str):
        self.label = name
        self.labels += 1
        self.lines.append({"type": "label", "t": self.elapsed(), "name": name})

    def add(self, sample, origin: str = "device"):
        if self.samples >= MAX_SAMPLES:
            self.truncated = True
            return
        imus = [{"id": imu.sensor_id, "valid": imu.valid, "q": _vec(imu.quaternion),
                 "a": _vec(imu.accel), "g": _vec(imu.gyro)} for imu in sample.imus]
        self.lines.append({"type": "sample", "t": self.elapsed(), "origin": origin, "format": sample.format,
                           "sample_time_ms": sample.sample_time_ms, "sequence": sample.sequence, "imus": imus})
        self.samples += 1

    def status(self) -> dict:
        return {"active": True, "session": self.session, "elapsed_s": self.elapsed(), "samples": self.samples,
                "label": self.label, "truncated": self.truncated}

    def save(self, aborted: bool = False) -> dict:
        folder = pathlib.Path(os.getenv("RECORDINGS_DIR", DEFAULT_DIR))
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{self.session}.jsonl"
        end = {"type": "end", "t": self.elapsed(), "samples": self.samples, "labels": self.labels,
               "aborted": aborted, "truncated": self.truncated}
        with path.open("w", encoding="utf-8") as handle:
            for line in (self.meta, *self.lines, end):
                handle.write(json.dumps(line, separators=(",", ":"), allow_nan=False) + "\n")
        return {"path": str(path), "session": self.session, "samples": self.samples,
                "duration_s": end["t"], "aborted": aborted, "truncated": self.truncated}
