# File: model_api.py
# Purpose: The contract between the simulator and any fall / man-down model (rules or learned).
#          A model receives canonical frames in time order and returns events; nothing else.
# Contributors: Unassigned
"""Model contract. See ../MODEL_API.md for the prose version.

Data flow: device emulator -> raw protobuf packets -> backend decoder -> canonicalize()
-> FrameBatch chunks -> model.update() -> Events -> check-in -> supervisor alerts.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

# Logical sensor slots in firmware order (sensor IDs 1..4, see protocol/ORIENTATION.md).
SLOTS = ("pelvis", "lumbar", "thoracic", "upper_thoracic")
GRAVITY = 9.80665  # m/s^2, standard gravity used for every g <-> m/s^2 conversion

# Every kind a model may emit. Only ALERT_KINDS start a worker check-in / supervisor alert.
EVENT_KINDS = (
    "suspected_fall",     # Stage-1 style candidate (impact, free fall or fast collapse); not an alert
    "near_fall",          # perturbation that ended without lying down
    "recovered",          # a candidate was dismissed: worker upright and moving again
    "fall_confirmed",     # a fall ended lying (or suspended/after a long drop): alert
    "man_down_warning",   # still after a fall/collapse long enough for the warning timer: alert
    "man_down_critical",  # still for the critical timer: alert, escalated
    "cleared",            # a previous alert no longer applies (worker moved or got up)
)
ALERT_KINDS = ("fall_confirmed", "man_down_warning", "man_down_critical")
WORK_MODES = ("normal", "lying_work")


@dataclass(frozen=True)
class StreamContext:
    """Fixed facts about one stream, given to reset() before the first batch."""

    fs_hz: float            # nominal canonical frame rate (the device output data rate)
    profile: str            # device profile name, see device.PROFILES
    accel_range_g: float    # full-scale range; values at the range are clipped (sat_mask)
    gyro_range_dps: float
    seed: int = 0


@dataclass(frozen=True)
class FrameBatch:
    """A chunk of consecutive canonical frames from one worker, oldest first.

    Frames are causal: a model must never use data from a later batch to explain an
    earlier one, and every Event it returns must have t <= t[-1] of the batch it was
    returned from. Batches are typically 0.25-1 s long; lost packets leave gaps in t.
    """

    t: np.ndarray          # (N,) float64 s, device time (strictly increasing, gaps allowed)
    acc: np.ndarray        # (N,4,3) float32 m/s^2 specific force, gravity included, per slot
    gyro: np.ndarray       # (N,4,3) float32 rad/s, per slot
    slot_mask: np.ndarray  # (N,4) bool, True where that slot delivered data
    sat_mask: np.ndarray   # (N,4) bool, True where any axis hit full scale (value is clipped)
    fs_hz: float           # nominal rate (same as StreamContext.fs_hz)
    work_mode: str = "normal"  # worker-declared mode for this whole batch, see WORK_MODES

    def __len__(self) -> int:
        return int(self.t.shape[0])


@dataclass
class Event:
    """One model decision. t is device time in seconds (<= the batch's last frame)."""

    t: float
    kind: str
    confidence: float = 1.0
    detail: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"Unknown event kind {self.kind!r}; use one of {EVENT_KINDS}")


@runtime_checkable
class FallModel(Protocol):
    """Streaming fall / man-down model. Construct with no arguments."""

    name: str

    def reset(self, context: StreamContext) -> None:
        """Forget all state; called once before each stream."""

    def update(self, batch: FrameBatch) -> list[Event]:
        """Consume the next batch in time order and return any new events."""


def load_model(spec: str) -> FallModel:
    """Load a model from 'baseline' or 'package.module:ClassName' (no-arg constructor)."""
    if spec == "baseline":
        from .baseline import BaselineRules

        return BaselineRules()
    if spec.startswith("baseline:"):  # baseline with a tuned config JSON, e.g. baseline:output/tuned.json
        import json

        from .baseline import BaselineConfig, BaselineRules

        with open(spec.split(":", 1)[1]) as handle:
            values = json.load(handle)
        values = {k: tuple(v) if isinstance(v, list) else v for k, v in values.items()}
        return BaselineRules(BaselineConfig(**values))
    if spec in ("falldet", "falldet-stream"):  # research-95's trained detector (detection/models)
        import os
        import sys
        from pathlib import Path

        root = Path(os.environ.get("THUNDERSIM_FALLDET", Path(__file__).resolve().parents[2] / "models"))
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        # falldet: their native FallModel (handles gaps, slot_mask, work_mode; silent below 40 Hz).
        # falldet-stream: our zero-order-hold wrapper around their FallDetectorStream, for comparison.
        spec = "falldet.sim_adapter:SpineFallModel" if spec == "falldet" else "thundersim.adapters.falldet:FallDetModel"
    if ":" not in spec:
        raise ValueError("Model spec must be 'baseline', 'falldet' or 'package.module:ClassName'")
    module_name, class_name = spec.split(":", 1)
    model = getattr(importlib.import_module(module_name), class_name)()
    if not isinstance(model, FallModel):
        raise TypeError(f"{spec} does not implement FallModel (name, reset, update)")
    return model
