# File: test_segments.py
# Purpose: Heading-free segment model (app/segments.py): maths, mounting fit, heading-drift immunity, and a
#          regression on the real 2026-11-08 "Collect data" session (fixture of per-hold world-up vectors).
# Contributors: Unassigned
import json
import math
import pathlib
import random
import unittest

from app.models import Imu, REGIONS, Sample
from app.segments import (SegmentModel, apply_mounting, fit_mounting, from_to, quat_unit, rotate, segment_state,
                          unit, world_up_in_sensor, UP)
from tools.analyze_recording import direction_check

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "session_20261109_holds.json"


def axis_angle(axis, degrees):
    axis = unit(axis)
    half = math.radians(degrees) / 2
    return (math.cos(half), *(c * math.sin(half) for c in axis))


def multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


class Sensor:
    """A simulated board strapped to a segment: arbitrary mounting, arbitrary (drifting) fusion heading."""

    def __init__(self, mounting_quat, heading_deg):
        self.mount = mounting_quat              # body -> sensor axes
        self.heading = heading_deg

    def quaternion(self, segment):
        """segment: body -> world rotation. Returns the fused sensor -> world quaternion."""
        mount_inv = (self.mount[0], -self.mount[1], -self.mount[2], -self.mount[3])
        return multiply(axis_angle(UP, self.heading), multiply(segment, mount_inv))


class MathTests(unittest.TestCase):
    def test_from_to(self):
        for a in ((1, 0, 0), (0, -1, 0), unit((1, 2, 3)), (0, 1, 0)):
            q = from_to(unit(a), UP)
            self.assertLess(math.dist(rotate(q, unit(a)), UP), 1e-9)

    def test_world_up_in_sensor(self):
        q = axis_angle((1, 0, 0), -90)          # sensor tipped forward 90 degrees
        self.assertLess(math.dist(world_up_in_sensor(q), (0, 0, 1)), 1e-9)

    def test_identity_mounting_angles(self):
        identity = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        forward = segment_state(axis_angle((1, 0, 0), -30), identity)
        self.assertAlmostEqual(forward["flexion_deg"], 30, places=6)
        self.assertAlmostEqual(forward["lateral_deg"], 0, places=6)
        right = segment_state(axis_angle((0, 0, 1), -20), identity)
        self.assertAlmostEqual(right["lateral_deg"], 20, places=6)
        self.assertAlmostEqual(right["inclination_deg"], 20, places=6)


class MountingTests(unittest.TestCase):
    def test_fit_recovers_any_mounting_and_ignores_heading(self):
        rng = random.Random(7)
        bends = {"forward": axis_angle((1, 0, 0), -40), "backward": axis_angle((1, 0, 0), 20),
                 "right": axis_angle((0, 0, 1), -25), "left": axis_angle((0, 0, 1), 25)}
        for trial in range(30):
            sensor = Sensor(quat_unit(tuple(rng.uniform(-1, 1) for _ in range(4))), rng.uniform(-180, 180))
            stand = world_up_in_sensor(sensor.quaternion((1, 0, 0, 0)))
            ups = {name: world_up_in_sensor(sensor.quaternion(q)) for name, q in bends.items()}
            mounting, report = fit_mounting(stand, ups)
            self.assertLess(report["direction_error_deg"], 1e-6, trial)
            # Heading drifts by up to 180 degrees after the fit: angles must not change.
            pose = axis_angle((1, 0, 0), -35)
            for drift in (0, 37, -120, 179):
                sensor.heading += drift
                state = segment_state(sensor.quaternion(pose), mounting)
                self.assertAlmostEqual(state["flexion_deg"], 35, places=5)
                self.assertAlmostEqual(state["lateral_deg"], 0, places=5)

    def test_forward_bend_alone_is_enough(self):
        sensor = Sensor(axis_angle((0, 1, 1), 70), 33)
        stand = world_up_in_sensor(sensor.quaternion((1, 0, 0, 0)))
        mounting, _ = fit_mounting(stand, {"forward": world_up_in_sensor(sensor.quaternion(axis_angle((1, 0, 0), -45)))})
        state = segment_state(sensor.quaternion(axis_angle((0, 0, 1), 30)), mounting)   # bend left 30
        self.assertAlmostEqual(state["lateral_deg"], -30, places=5)

    def test_no_bend_tilt_is_an_error(self):
        with self.assertRaises(ValueError):
            fit_mounting((0, 1, 0), {"forward": (0, 1, 0)})

    def test_model_compute_and_json(self):
        identity = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        model = SegmentModel({region: identity for region in REGIONS.values()}, "test")
        sample = Sample("quaternion", 0, tuple(Imu(i, quaternion=axis_angle((1, 0, 0), -10)) for i in REGIONS))
        out = model.compute(sample, REGIONS)
        self.assertEqual(set(out), set(REGIONS.values()))
        self.assertAlmostEqual(out["pelvis"]["flexion_deg"], 10, places=2)
        again = SegmentModel.from_json(json.loads(json.dumps(model.to_json())))
        self.assertEqual(again.compute(sample, REGIONS), out)
        self.assertIsNone(model.compute(Sample("raw", 0, ()), REGIONS))


class RealSessionRegression(unittest.TestCase):
    """2026-11-08 session (hardware, wearer b): fitted on stand#1 + round-1 bends, checked on everything."""

    @classmethod
    def setUpClass(cls):
        data = json.loads(FIXTURE.read_text())
        cls.holds = {name: {r: tuple(v) for r, v in ups.items()} for name, ups in data["holds"].items()}
        from app.recording_io import fit_model
        cls.model = fit_model(cls.holds)

    def test_every_sensor_gets_a_mounting(self):
        self.assertEqual(set(self.model.mountings), set(REGIONS.values()))
        for region, info in self.model.report.items():
            self.assertLess(info["direction_error_deg"], 15, region)

    def test_no_segment_bends_the_wrong_way(self):
        rows = direction_check(self.holds, self.model)
        wrong = [(name, n) for name, _, n in rows if n]
        self.assertEqual(wrong, [])
        self.assertGreaterEqual(sum(1 for _, _, n in rows if n is not None), 12)

    def test_held_out_bends_have_the_right_size(self):
        rows = {name: angles for name, angles, _ in direction_check(self.holds, self.model)}
        for region in ("lumbar", "thoracic", "upper_thoracic"):
            self.assertGreater(rows["forward#2:hold"][region][0], 30, region)       # bent well forward
            self.assertLess(rows["backward#2:hold"][region][0], -15, region)
            self.assertLess(rows["left#2:hold"][region][1], -15, region)
            self.assertGreater(rows["right#2:hold"][region][1], 15, region)

    def test_lying_is_near_horizontal(self):
        for hold in ("lie_back#1:hold", "lie_front#1:hold", "lie_left#1:hold", "lie_right#1:hold"):
            for region, up in self.holds[hold].items():
                g = unit(apply_mounting(self.model.mountings[region], up))
                inclination = math.degrees(math.acos(max(-1.0, min(1.0, g[1]))))
                self.assertGreater(inclination, 60, f"{hold} {region}")

    def test_standing_returns_to_upright(self):
        rows = {name: angles for name, angles, _ in direction_check(self.holds, self.model)}
        for region, (flex, lat) in rows["stand_end#1:hold"].items():
            self.assertLess(math.hypot(flex, lat), 15, region)


if __name__ == "__main__":
    unittest.main()
