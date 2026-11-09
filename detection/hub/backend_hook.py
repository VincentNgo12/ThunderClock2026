# File: backend_hook.py
# Purpose: Backend wiring for the hub (DETECTION_ARCHITECTURE.md 3.1, migration step B1) through the
#          documented raw handoff (backend/DIAGNOSTICS.md "Raw IMU handoff", create_app(on_raw_sample=)).
#          backend/ is not edited: this module wraps create_app and owns intake and the detector thread.
# Contributors: Unassigned
"""Run (from backend/, with detection/ on PYTHONPATH and the API dependencies installed):

    HUB_BACKEND_APP=1 TELEMETRY_SOURCE=hardware DEVICE_ID="<id>" DEVICE_FORMAT=both \\
        python -m uvicorn hub.backend_hook:app --host 127.0.0.1 --port 8000

DEVICE_FORMAT=raw or raw_fusion also deliver raw frames to the hook (f3337e3).

Threads: the callback runs on the asyncio loop and only appends (sample, receipt time) to a bounded
deque (Q2, 10 s at 100 Hz) and counts drops. ONE detector thread drains it every 100 ms, runs
RawFrames and MainModel.update_arrays and applies dismissals, so the model needs no locks.
Models are loaded at construction, never on the loop.
Not done here (needs backend changes, see proposals/): a raw queue >= 256 separate from the shared
DeviceStream queue (Q1), a loop-lag monitor, and publishing snapshot() as fall_state/detector.
"""
import os
import threading
import time
from collections import deque

os.environ.setdefault("OMP_NUM_THREADS", "1")   # one detector thread; no OpenMP fan-out

from .intake import RawFrames  # noqa: E402
from .main_model import MainModel  # noqa: E402

Q2_SIZE = 1000
PERIOD_S = 0.1
STALE_S = 3.0
EVENTS_KEPT = 20


class DetectorService:
    def __init__(self, model=None, frames=None, start_thread=True):
        self.model = model or MainModel()
        self.frames = frames or RawFrames(os.environ.get("RAW_IMU_AXES"))
        self.queue = deque(maxlen=Q2_SIZE)
        self.commands = deque(maxlen=16)
        self.dropped = 0
        self.received = 0
        self.last_receipt = None
        self.error = None
        self.events = deque(maxlen=EVENTS_KEPT)
        self.next_event_id = 1
        self.update_ms_max = 0.0
        self.reset_count = 0                # completed reset commands (detector thread)
        self.floor_id = 0                   # events with id <= floor_id belong to a stream before the last reset
        self._stop = threading.Event()
        self.thread = None
        if start_thread:
            self.thread = threading.Thread(target=self._run, name="hub-detector", daemon=True)
            self.thread.start()

    # ------------------------------------------------------------------ loop thread (must be quick)
    def on_raw_sample(self, sample, received_monotonic=None):
        if len(self.queue) == self.queue.maxlen:
            self.dropped += 1
        self.queue.append((sample, time.monotonic() if received_monotonic is None else received_monotonic))
        self.received += 1

    def dismiss(self, episode_id=None):
        """Supervisor dismissal; applied by the detector thread."""
        self.commands.append(("dismiss", episode_id))

    def calibrate(self):
        self.commands.append(("calibrate", None))

    def reset(self):
        """Start a new stream: drop queued samples and all model state (demo scenario switch).
        Applied by the detector thread before any sample that follows this call is read."""
        self.queue.clear()
        self.commands.append(("reset", None))

    # ------------------------------------------------------------------ detector thread
    def _run(self):
        while not self._stop.is_set():
            started = time.monotonic()
            self.step()
            self._stop.wait(max(0.0, PERIOD_S - (time.monotonic() - started)))

    def stop(self):
        self._stop.set()

    def step(self):
        """Drain Q2 once and run the model (called by the thread, or directly in tests)."""
        if self.error:
            return []
        try:
            while self.commands:
                cmd, arg = self.commands.popleft()
                if cmd == "dismiss":
                    self.model.dismiss(arg)
                elif cmd == "calibrate":
                    self.frames.calibrate()
                elif cmd == "reset":
                    self.model.reset()
                    self.frames.reset()
                    self.error = None
                    self.floor_id = self.next_event_id - 1
                    self.reset_count += 1
            n = len(self.queue)
            for _ in range(n):
                sample, receipt = self.queue.popleft()
                self.last_receipt = receipt
                self.frames.push(sample)
            arrays = self.frames.pop()
            if arrays is None:
                return []
            t0 = time.perf_counter()
            t, acc, gyro, mask = arrays
            new = self.model.update_arrays(t, acc, gyro, mask, None, "normal")
            self.update_ms_max = max(self.update_ms_max, (time.perf_counter() - t0) * 1000)
            for e in new:
                self.events.append({"id": self.next_event_id, "t": round(e.t, 3), "kind": e.kind,
                                    "confidence": round(e.confidence, 3), "detail": e.detail})
                self.next_event_id += 1
            return new
        except Exception as error:   # visible as detector.status = error; never crash the API
            self.error = f"{type(error).__name__}: {error}"
            return []

    # ------------------------------------------------------------------ read side
    def snapshot(self, now=None):
        """The section-5 'detector' block plus the last events (read from the loop; plain copies)."""
        now = time.monotonic() if now is None else now
        status = self.model.status()
        if self.error:
            status["status"] = "error"
        elif self.last_receipt is None or now - self.last_receipt > STALE_S:
            status["status"] = "stale"
        elif not self.frames.calibrated:
            status["status"] = "uncalibrated"
        status.update(intake_dropped=self.dropped, intake_received=self.received,
                      update_ms_max=round(self.update_ms_max, 1), error=self.error, label="prototype indicator")
        return {"detector": status, "detection_events": list(self.events)}


def make_app(service=None):
    """create_app(on_raw_sample=...) from backend/app/main.py with a running DetectorService."""
    from app.main import create_app  # backend/ on sys.path (run from backend/)

    service = service or DetectorService()

    def accept(sample):
        # The hook passes only the Sample. In raw / raw_fusion mode the stream's received_monotonic is
        # this frame's BLE receipt time; in "both" mode it tracks quaternion packets only (ble.py at
        # f3337e3), so the dequeue time is used instead (later by at most the queue latency).
        stream = getattr(app.state, "stream", None)
        receipt = None
        if getattr(stream, "packet_format", None) == "raw":
            receipt = getattr(stream, "received_monotonic", None)
        service.on_raw_sample(sample, receipt)

    app = create_app(on_raw_sample=accept)
    app.state.detector = service
    return app


if os.environ.get("HUB_BACKEND_APP") == "1":   # opt-in so importing this module stays side-effect free
    app = make_app()
