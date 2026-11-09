# File: recorder.py
# Purpose: Builder subclass that logs which motion primitives a scenario called, and when,
#          without changing the simulation (it draws no random numbers of its own).
# Contributors: Unassigned
"""Recording builder for the 3D viewer.

The spine track says how the trunk moved, not why. The viewer also needs to know that the
worker was on a ladder, sitting on a bucket or carrying a bag, so it can pose the limbs and
place the props. RecordingBuilder wraps the Builder's primitives and keeps one Call per use.
Nested calls (carry -> stoop -> stand_up) are kept with their depth.

Times inside a falls.Faller integration are stale: the Faller commits its segment only at
the end, so ground_y changes made during a fall are logged at the fall's start time. Use the
ground-truth events for timing inside falls.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field

import numpy as np

from ..motions import Builder

# Primitives worth knowing about. Low-level helpers (move, seg, commit) are not logged.
WRAPPED = (
    "stand_up", "stand", "hold", "stoop", "squat", "kneel", "quadruped", "sit_ground", "lie", "get_up",
    "walk", "run", "crawl", "ladder", "impulses", "hammer", "vibrate", "jackhammer", "drill_overhead",
    "carry", "lift", "put_down", "rebar_tie", "kneel_work", "sit", "cab_ride", "jump_down", "lying_activity", "remove_strip",
    "strap_slip", "hat_off", "set_work_mode", "posture_hold", "sit_posture",
)


@dataclass
class Call:
    """One primitive call. t0/t1 in simulation seconds; pos/heading are the pelvis pose."""

    index: int
    name: str
    t0: float
    depth: int
    parent: int | None
    args: dict
    ground0: float
    pos0: np.ndarray
    heading0: float
    t1: float = 0.0
    ground1: float = 0.0
    pos1: np.ndarray = field(default=None)


def _simple(value) -> bool:
    return isinstance(value, (bool, int, float, str, np.floating, np.integer)) or value is None


class RecordingBuilder(Builder):
    """Builder that logs primitive calls, support-height changes and consciousness changes."""

    def __init__(self, worker, site, rng, fs: float = 1000.0, sink: list | None = None):
        self.calls: list[Call] = []
        self.ground_log: list[tuple[float, float]] = []
        self.conscious_log: list[tuple[float, bool]] = []
        self._stack: list[int] = []
        super().__init__(worker, site, rng, fs)
        for name in WRAPPED:
            setattr(self, name, self._wrap(name, getattr(self, name)))
        if sink is not None:
            sink.append(self)

    # Builder assigns these plain attributes; properties let us log every change.
    @property
    def ground_y(self) -> float:
        return self._ground_y

    @ground_y.setter
    def ground_y(self, value: float) -> None:
        self._ground_y = float(value)
        self.ground_log.append((self.t, float(value)))

    @property
    def conscious(self) -> bool:
        return self._conscious

    @conscious.setter
    def conscious(self, value: bool) -> None:
        self._conscious = bool(value)
        self.conscious_log.append((self.t, bool(value)))

    def _wrap(self, name: str, method):
        signature = inspect.signature(method)

        def wrapper(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            info = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in bound.arguments.items() if _simple(v)}
            if name == "impulses":
                # Strike times are relative to the (not yet committed) segment that starts now.
                info["times_abs"] = (self.t + np.asarray(bound.arguments["times"], float)).tolist()
            call = Call(len(self.calls), name, self.t, len(self._stack), self._stack[-1] if self._stack else None, info,
                        self.ground_y, self.pose.pos.copy(), float(self.pose.heading))
            self.calls.append(call)
            self._stack.append(call.index)
            try:
                return method(*args, **kwargs)
            finally:
                self._stack.pop()
                call.t1 = self.t
                call.ground1 = self.ground_y
                call.pos1 = self.pose.pos.copy()

        return wrapper

    # --- queries used by the scene and rig builders ------------------------------------
    def find(self, name: str, inside: Call | None = None) -> list[Call]:
        """All calls with this name, optionally restricted to those nested inside `inside`."""
        found = [c for c in self.calls if c.name == name]
        if inside is not None:
            found = [c for c in found if self.is_inside(c, inside)]
        return found

    def is_inside(self, call: Call, outer: Call) -> bool:
        index = call.parent
        while index is not None:
            if index == outer.index:
                return True
            index = self.calls[index].parent
        return False

    def ancestors(self, call: Call) -> list[str]:
        names = []
        index = call.parent
        while index is not None:
            names.append(self.calls[index].name)
            index = self.calls[index].parent
        return names
