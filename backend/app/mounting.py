# File: mounting.py
# Purpose: Sensor mounting calibration for the heading-free segment model: capture held poses from the live
#          stream, fit each sensor's mounting, save it per strip, re-zero "upright" when the wearer stands
#          still. Mounting is physical, so a saved one stays valid across ESP32 reboots until the vest moves.
# Contributors: Unassigned
"""Flow (POST /mounting/... in main.py, driven by the dashboard's Calibrate... wizard):

  capture stand   -> world-up per sensor while standing still (body up axis)
  capture forward -> (and optionally backward / left / right) sets each sensor's forward direction
  fit             -> SegmentModel, saved to MOUNTINGS_DIR/<device>.json and used at once
  zero            -> later: re-take only the standing up axis, keep each sensor's forward direction

Auto re-zero: after a (re)connect, the first ~2 s of standing still near the saved upright pose re-zeroes,
so small vest shifts between sessions do not tilt the whole view.
"""
import asyncio
import json
import math
import os
import pathlib
import time

from .models import REGIONS
from .segments import SegmentModel, add, cross, dot, fit_mounting, mean_vector, norm, scale, unit, world_up_in_sensor

POSES = ("stand", "forward", "backward", "left", "right")
DEFAULT_DIR = pathlib.Path(__file__).resolve().parent.parent / "mountings"
STILL_SPREAD_DEG = 4.0          # max spread of world-up during a capture to count as "held still"
AUTO_ZERO_STILL_S = 2.0
AUTO_ZERO_MAX_DEG = 25.0        # only re-zero when every sensor is within this of the saved upright pose


def mountings_dir():
    return pathlib.Path(os.getenv("MOUNTINGS_DIR", DEFAULT_DIR))


def mounting_path(device_key):
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (device_key or "default"))
    return mountings_dir() / f"{safe}.json"


def load_model(device_key):
    path = mounting_path(device_key)
    if not path.exists():
        return None
    return SegmentModel.from_json(json.loads(path.read_text(encoding="utf-8")))


def save_model(model, device_key):
    path = mounting_path(device_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_json(), indent=2), encoding="utf-8")
    return str(path)


def identity_model(source):
    """Simulated sources send body-aligned orientations (+Y up, +Z back), so no calibration is needed."""
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    return SegmentModel({region: identity for region in REGIONS.values()}, source)


def angle_deg(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, dot(a, b)))))


def rezero(model, stand_up, source):
    """Same forward directions, new upright: body up = current standing gravity, back re-orthogonalised."""
    mountings = {}
    for region, (b_x, b_y, b_z) in model.mountings.items():
        up = stand_up.get(region)
        if up is None:
            mountings[region] = (b_x, b_y, b_z)
            continue
        new_y = unit(up)
        back = add(b_z, scale(new_y, -dot(b_z, new_y)))
        if norm(back) < 1e-3:                        # degenerate: keep the old frame
            mountings[region] = (b_x, b_y, b_z)
            continue
        new_z = unit(back)
        mountings[region] = (cross(new_y, new_z), new_y, new_z)
    return SegmentModel(mountings, source, {**model.report, "rezeroed_at": time.time()})


class Capture:
    """Collects world-up per sensor from quaternion samples for a fixed time."""

    def __init__(self, seconds):
        self.until = time.monotonic() + seconds
        self.ups = {region: [] for region in REGIONS.values()}
        self.done = asyncio.Event()

    def add(self, sample):
        if sample.format != "quaternion":
            return
        if time.monotonic() >= self.until:
            self.done.set()
            return
        for imu in sample.imus:
            if imu.valid and imu.quaternion is not None:
                self.ups[REGIONS[imu.sensor_id]].append(world_up_in_sensor(imu.quaternion))

    def result(self):
        out = {}
        for region, ups in self.ups.items():
            if len(ups) < 3:
                continue
            mean = mean_vector(ups)
            spread = max(angle_deg(mean, u) for u in ups)
            out[region] = {"up": mean, "samples": len(ups), "spread_deg": round(spread, 1)}
        return out


class Calibrator:
    """Holds captured poses until they are fitted; also runs the automatic re-zero."""

    def __init__(self):
        self.captures = {}            # pose -> {region: {"up", "samples", "spread_deg"}}
        self.active = None            # Capture in progress
        self.auto_zero_pending = False
        self._still_since = None
        self._still_ups = None

    # ---- explicit calibration
    async def capture(self, pose, seconds):
        if pose not in POSES:
            raise ValueError(f"pose must be one of {', '.join(POSES)}")
        if self.active is not None:
            raise RuntimeError("a capture is already running")
        self.active = Capture(seconds)
        try:
            await asyncio.wait_for(self.active.done.wait(), seconds + 3)
        except TimeoutError:
            pass                      # a stalled stream: report what arrived
        finally:
            capture, self.active = self.active, None
        result = capture.result()
        if not result:
            raise RuntimeError("no orientation data arrived during the capture (is the strip streaming?)")
        self.captures[pose] = result
        stand = self.captures.get("stand")
        summary = {}
        for region, info in result.items():
            summary[region] = {"samples": info["samples"], "spread_deg": info["spread_deg"],
                               "still": info["spread_deg"] <= STILL_SPREAD_DEG}
            if stand and pose != "stand" and region in stand:
                summary[region]["tilt_from_stand_deg"] = round(angle_deg(stand[region]["up"], info["up"]), 1)
        return {"pose": pose, "sensors": summary, "captured_poses": sorted(self.captures)}

    def fit(self, source):
        if "stand" not in self.captures:
            raise ValueError("capture 'stand' first")
        bends = [p for p in POSES if p != "stand" and p in self.captures]
        if not bends:
            raise ValueError("capture at least one bend (forward) after 'stand'")
        mountings, report = {}, {}
        for region, info in self.captures["stand"].items():
            ups = {p: self.captures[p][region]["up"] for p in bends if region in self.captures[p]}
            try:
                mountings[region], report[region] = fit_mounting(info["up"], ups)
            except ValueError as error:
                report[region] = {"error": str(error)}
        if not mountings:
            raise ValueError("no sensor tilted enough during the bends; bend further and repeat")
        return SegmentModel(mountings, source, report)

    def reset(self):
        self.captures.clear()

    # ---- automatic re-zero (called for every quaternion sample)
    def watch(self, sample, model, now):
        """Return a re-zeroed model once the wearer has stood still and upright for AUTO_ZERO_STILL_S."""
        if self.active is not None:
            self.active.add(sample)
        if not self.auto_zero_pending or model is None or sample.format != "quaternion":
            return None
        ups = {REGIONS[i.sensor_id]: world_up_in_sensor(i.quaternion) for i in sample.imus
               if i.valid and i.quaternion is not None}
        near_upright = all(region in ups and angle_deg(m[1], ups[region]) <= AUTO_ZERO_MAX_DEG
                           for region, m in model.mountings.items())
        still = (self._still_ups is not None and set(ups) == set(self._still_ups)
                 and all(angle_deg(ups[r], self._still_ups[r]) <= STILL_SPREAD_DEG for r in ups))
        if not near_upright:
            self._still_since, self._still_ups = None, None
            return None
        if not still:
            self._still_since, self._still_ups = now, ups
            return None
        if now - self._still_since < AUTO_ZERO_STILL_S:
            return None
        self.auto_zero_pending = False
        self._still_since, self._still_ups = None, None
        return rezero(model, ups, model.source)
