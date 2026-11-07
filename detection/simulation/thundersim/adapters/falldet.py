# File: falldet.py
# Purpose: Wrap research-95's trained two-stage detector (detection/models/falldet,
#          see its HANDOFF.md) in the simulator's FallModel API without modifying it.
# Contributors: Unassigned
"""Adapter for falldet.FallDetectorStream.

The detector expects a uniform 100 Hz stream of all four anchors and has no gap or
missing-anchor handling (HANDOFF.md 2 and 5.6). The adapter therefore:
- repeats the last frame for ticks lost on the radio (zero-order hold onto a 100 Hz grid),
  which also turns the current firmware's 0.5 Hz into held 100 Hz samples;
- holds an anchor's last valid value while slot_mask is False (HANDOFF option "send its last value");
- selects the detector's 'current' model for the current* device profiles, 'target' otherwise.
Mapping: stage1_trigger -> suspected_fall; stage2_result fall_confirmed / near_fall / normal ->
fall_confirmed / near_fall / recovered.

Set THUNDERSIM_FALLDET to the fall model directory if it is not at detection/models.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from ..model_api import GRAVITY, Event, FrameBatch, StreamContext

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "models"
MAX_FILL_S = 10.0  # never synthesise more than 10 s of held samples for one gap


def _import_stream():
    path = Path(os.environ.get("THUNDERSIM_FALLDET", DEFAULT_PATH))
    if not (path / "falldet").is_dir():
        raise ImportError(f"falldet package not found under {path}; set THUNDERSIM_FALLDET")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    from falldet import FallDetectorStream  # noqa: E402

    return FallDetectorStream


class FallDetModel:
    """FallModel wrapper around falldet.FallDetectorStream (research-95)."""

    name = "falldet"

    def __init__(self):
        self._stream_cls = _import_stream()

    def reset(self, context: StreamContext) -> None:
        self.detector_profile = "current" if context.profile.startswith("current") else "target"
        self.stream = self._stream_cls(profile=self.detector_profile)
        self.step = 0.01
        self.last_t = None
        self.hold_acc = np.tile([0.0, GRAVITY, 0.0], (4, 1))
        self.hold_gyro = np.zeros((4, 3))
        self.filled = 0

    def _push(self, t: float, acc: np.ndarray, gyro: np.ndarray) -> list[Event]:
        out = []
        for ev in self.stream.push({"t_ms": int(round(t * 1000)), "acc": acc.tolist(), "gyro": gyro.tolist()}) or []:
            if ev.get("type") == "stage1_trigger":
                out.append(Event(t, "suspected_fall", 0.5, {"source": "falldet"}))
            elif ev.get("type") == "stage2_result":
                kind = {"fall_confirmed": "fall_confirmed", "near_fall": "near_fall"}.get(ev.get("state"), "recovered")
                detail = {"state": ev.get("state"), "trigger_t_ms": ev.get("t_ms"),
                          "probabilities": {k: round(float(v), 3) for k, v in (ev.get("probabilities") or {}).items()}}
                out.append(Event(t, kind, float(ev.get("confidence", 1.0)), detail))
        return out

    def update(self, batch: FrameBatch) -> list[Event]:
        events: list[Event] = []
        for i in range(len(batch)):
            t = float(batch.t[i])
            mask = batch.slot_mask[i]
            acc = np.where(mask[:, None], batch.acc[i], self.hold_acc).astype(float)
            gyro = np.where(mask[:, None], batch.gyro[i], self.hold_gyro).astype(float)
            if self.last_t is not None:
                missing = int(round((t - self.last_t) / self.step)) - 1
                for k in range(1, min(missing, int(MAX_FILL_S / self.step)) + 1):
                    self.filled += 1
                    events.extend(self._push(self.last_t + k * self.step, self.hold_acc, self.hold_gyro))
            events.extend(self._push(t, acc, gyro))
            self.hold_acc, self.hold_gyro, self.last_t = acc, gyro, t
        return events
