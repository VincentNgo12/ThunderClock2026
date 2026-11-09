# File: replay.py
# Purpose: Play a "Collect data" recording back as the telemetry source, in real time or faster, with the
#          wizard's step labels, so the frontend, posture and detector see the session again. Started either
#          at launch (TELEMETRY_SOURCE=replay) or from the dashboard's Replay panel (POST /replay/start).
# Contributors: Unassigned
import asyncio
from datetime import UTC, datetime
import json
import os
import pathlib
import time

from .models import Diagnostics
from .recording import DEFAULT_DIR as RECORDINGS_DEFAULT_DIR
from .recording_io import read_recording


def recordings_dir():
    return pathlib.Path(os.getenv("RECORDINGS_DIR", RECORDINGS_DEFAULT_DIR))


def _last_line(path):
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - 4096))
        tail = handle.read().decode("utf-8", "replace").strip().splitlines()
    return tail[-1] if tail else ""


def list_recordings():
    """Newest first: file name, session, start time, wearer, notes, duration, format, completeness."""
    folder = recordings_dir()
    out = []
    for path in sorted(folder.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(path, encoding="utf-8") as handle:
                meta = json.loads(handle.readline())
            if meta.get("type") != "meta":
                continue
            end = json.loads(_last_line(path))
            client = meta.get("client") or {}
            out.append({"file": path.name, "session": meta.get("session"), "started_at": meta.get("started_at"),
                        "source": meta.get("source"), "device_format": meta.get("device_format"),
                        "wearer": client.get("wearer") or None, "notes": client.get("notes") or None,
                        "duration_s": end.get("t") if end.get("type") == "end" else None,
                        "aborted": end.get("aborted") if end.get("type") == "end" else None,
                        "size_mb": round(path.stat().st_size / 1e6, 1)})
        except (OSError, ValueError):
            continue
    return out


def resolve_recording(name):
    """A recording file inside RECORDINGS_DIR, by bare file name only (no paths)."""
    if not name or pathlib.Path(name).name != name or not name.endswith(".jsonl"):
        raise ValueError("give the recording's file name, e.g. 20261109T043526Z_session_1.jsonl")
    path = recordings_dir() / name
    if not path.is_file():
        raise FileNotFoundError(f"no recording {name} in {recordings_dir()}")
    return path


class ReplayStream:
    """Same attributes and updates() contract as SimulatedStream / DeviceStream."""

    packet_format = "both"

    def __init__(self, path, speed=1.0, loop=True, start_delay_s=0.0, start_t=0.0):
        if speed <= 0:
            raise ValueError("speed must be positive")
        self.recording = read_recording(path)
        self.path, self.speed, self.loop = str(path), float(speed), loop
        self.start_delay_s = float(start_delay_s)    # e.g. let a test browser load before playback starts
        self.start_t = max(0.0, float(start_t))      # recording time to start from (first pass only)
        self.paused = False
        self.stats = Diagnostics()
        self.latest = None
        self.received_monotonic = None
        self.received_at = None
        self.t = self.start_t                        # recording time of the last sample sent
        self.label = self.recording.label_at(self.start_t)
        self.pass_index = 0
        self.finished = False

    @property
    def duration_s(self):
        if self.recording.end:
            return self.recording.end["t"]
        return self.recording.samples[-1][0] if self.recording.samples else 0.0

    def status(self):
        return {"file": self.path.rsplit("/", 1)[-1], "t": round(self.t, 2), "duration_s": round(self.duration_s, 1),
                "label": self.label, "speed": self.speed, "pass": self.pass_index, "paused": self.paused,
                "loop": self.loop, "finished": self.finished,
                "steps": [[round(t, 2), name] for t, name in self.recording.labels]}

    async def updates(self):
        self.stats.state = "simulated"
        self.stats.started_at = time.monotonic()
        samples = self.recording.samples
        if self.start_delay_s > 0:
            await asyncio.sleep(self.start_delay_s)
        offset = self.start_t
        try:
            while True:
                start = time.monotonic()
                self.stats.resets += 1 if self.pass_index else 0     # a new pass is a new "boot"
                for t, origin, sample in samples:
                    if origin != "device" or t < offset:
                        continue                                    # laptop fusion output is recomputed live
                    while True:
                        if self.paused:
                            paused_at = time.monotonic()
                            while self.paused:
                                await asyncio.sleep(0.05)
                            start += time.monotonic() - paused_at     # resume where we stopped
                        delay = start + (t - offset) / self.speed - time.monotonic()
                        if delay <= 0:
                            break
                        await asyncio.sleep(min(delay, 0.1))          # stay responsive to pause
                    now = time.monotonic()
                    self.t = t
                    self.label = self.recording.label_at(t)
                    self.stats.received += 1
                    self.stats.decoded += 1
                    self.stats.usable += 1
                    self.stats.last_notification = self.stats.last_valid = now
                    if sample.format == "quaternion":
                        self.latest = sample
                        self.received_monotonic = now
                        self.received_at = datetime.now(UTC)
                    yield sample
                self.pass_index += 1
                offset = 0.0
                if not self.loop:
                    self.finished = True
                    while True:                                     # hold the last pose; the API can stop it
                        await asyncio.sleep(0.5)
                        yield None
                await asyncio.sleep(1.0)
        finally:
            self.stats.state = "disconnected"
            self.stats.finished_at = time.monotonic()
