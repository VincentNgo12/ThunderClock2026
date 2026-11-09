# File: test_viz_contact.py
# Purpose: The viewer's contact and balance pass (thundersim/viz/contact.py) and its gates
#          (contact_gates.py): the switch, the geometry helpers, ground truth unchanged, and the
#          gates on carry and ladder runs (hands on handholds, no load through the legs, no skate,
#          joint limits, no pops, static balance).
# Contributors: Unassigned

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thundersim.viz import contact, contact_gates, contact_model, rig  # noqa: E402
from thundersim.viz.export import run_payload  # noqa: E402
from thundersim.viz.geom import UP, right, unit  # noqa: E402

GROUND_TRUTH = ("spine", "segq", "anchors", "chipq", "phase", "worn")
_RUNS: dict = {}


def payload(scenario: str, seed: int, on: bool):
    """(payload, keep) of one run, built once per test session."""
    key = (scenario, seed, on)
    if key not in _RUNS:
        keep: dict = {}
        _RUNS[key] = (run_payload(scenario, seed, model="none", mocap=False, contact=on, keep=keep), keep)
    return _RUNS[key]


class TestSwitch(unittest.TestCase):
    def test_flag_env_and_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(contact.ENV, None)
            self.assertEqual(contact.requested(), contact.DEFAULT_ON)
            self.assertTrue(contact.requested(True))
            self.assertFalse(contact.requested(False))
            os.environ[contact.ENV] = "1"
            self.assertTrue(contact.requested())
            os.environ[contact.ENV] = "0"
            self.assertFalse(contact.requested())


class TestGeometry(unittest.TestCase):
    def test_support_polygon_margin(self):
        square = [np.array(p, float) for p in ((0, 0), (1, 0), (1, 1), (0, 1))]
        self.assertAlmostEqual(contact_model.support_polygon_margin(square, np.array([0.5, 0.5])), 0.5)
        self.assertAlmostEqual(contact_model.support_polygon_margin(square, np.array([1.5, 0.5])), -0.5)
        self.assertAlmostEqual(contact_model.support_polygon_margin(square[:2], np.array([0.5, 0.2])), -0.2)

    def test_box_signed_distance(self):
        R = contact_model.sack_axes(np.array([0.3]))
        centre = np.zeros((1, 3))
        half = np.array([0.1, 0.2, 0.05])
        across = right(0.3)
        pts = np.stack([np.zeros(3), across * 0.15, UP * 0.1])[None]
        d = contact_model.box_signed_distance(pts, centre, R, half)[0]
        np.testing.assert_allclose(d, [-0.05, 0.05, -0.05], atol=1e-9)

    def test_sack_handholds_are_on_the_sides_at_the_grip_height(self):
        centre = np.array([[1.0, contact_model.SACK_HALF[1], 2.0]])
        hh = contact_model.sack_handholds(centre, np.array([0.0]), 0.4)[0]
        np.testing.assert_allclose(hh[:, 1], [0.4, 0.4])
        np.testing.assert_allclose(np.linalg.norm(hh[0] - hh[1]), 2 * rig.LOAD_HALF_WIDTH_M)
        hh = contact_model.sack_handholds(centre, np.array([0.0]), np.array([[0.2, 0.45]]))[0]   # per hand
        np.testing.assert_allclose(hh[:, 1], [0.2, 0.45])

    def test_sack_regrip_moves_one_hand_after_the_other(self):
        sk = contact_model.Sack(0, 1.0, 2.0, 20.0, 21.0, 0.2, 0.4, 5.0, 20.0, np.zeros(3), np.zeros(3), 0.0, 0.0)
        g = sk.grips(np.array([4.9, 5.25, 5.5, 5.75, 6.1]))
        np.testing.assert_allclose(g[0], [0.2, 0.2])                          # the lift's grip until the re-grip
        self.assertTrue(0.2 < g[1, 0] < 0.4 and g[1, 1] == 0.2)               # left hand first
        np.testing.assert_allclose(g[2], [0.4, 0.2])
        self.assertTrue(0.2 < g[3, 1] < 0.4)                                  # then the right hand
        np.testing.assert_allclose(g[4], [0.4, 0.4])                          # the put-down's grip

    def test_swing_forward_keeps_the_arm_length(self):
        shoulder, hand = np.array([0.0, 1.4, 0.0]), np.array([0.1, 0.8, 0.05])
        out = contact.swing_forward(shoulder, hand, np.array([0.0, 0.0, -1.0]), np.radians([0.0, 20.0, 40.0]))
        np.testing.assert_allclose(out[0], hand, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(out - shoulder, axis=1), np.linalg.norm(hand - shoulder))
        self.assertTrue(np.all(np.diff(-out[:, 2]) > 0) and np.all(np.diff(out[:, 1]) > 0))   # forward and up
        np.testing.assert_allclose(out[:, 0], 0.1)                             # sideways part kept

    def test_stepladder_handhold_matches_the_drawn_carry(self):
        # scene.stepladder_motion draws the carried ladder (open 0) at hand + UP (0.03 - 0.24) + right 0.03 L.
        height, yaw = 3.1, 0.7
        length = height / np.cos(np.arctan2(1, 4))
        hand = np.array([1.0, 0.9, -2.0])
        pos = hand + UP * (0.03 - 0.24) + right(yaw) * (0.03 * length)
        hh = contact_model.stepladder_handhold(pos[None], np.array([yaw]), np.array([0.0]), height)[0]
        np.testing.assert_allclose(hh, hand, atol=1e-9)

    def test_envelope_covers_the_need_and_limits_the_slope(self):
        need = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
        out = contact.envelope(need, 0.03)
        self.assertTrue(np.all(out >= need - 1e-12))
        self.assertLessEqual(np.abs(np.diff(out)).max(), 0.03 + 1e-12)


class TestGroundTruth(unittest.TestCase):
    def test_contact_pass_changes_no_simulated_field(self):
        off, _ = payload("A-CARRY", 0, False)
        on, _ = payload("A-CARRY", 0, True)
        for name in GROUND_TRUTH:
            self.assertEqual(off["frames"][name], on["frames"][name], name)
        for name in off["signals"]:
            self.assertEqual(off["signals"][name], on["signals"][name], name)
        self.assertEqual(off["truth"], on["truth"])
        self.assertFalse(off["meta"]["contact"]["enabled"])
        self.assertTrue(on["meta"]["contact"]["enabled"])

    def test_bones_keep_their_lengths(self):
        on, keep = payload("A-CARRY", 0, True)
        J = np.array(on["frames"]["joints"]).reshape(-1, 18, 3)
        d = keep["rig"].dims
        for a, b, length in ((0, 2, d.thigh), (2, 4, d.shank), (10, 12, d.upper_arm), (12, 14, d.forearm)):
            for side in (0, 1):
                err = np.abs(np.linalg.norm(J[:, b + side] - J[:, a + side], axis=1) - length)
                self.assertLess(float(err.max()), 2e-3, (a, b, side))


class TestGates(unittest.TestCase):
    def check_run(self, scenario: str, seed: int) -> dict:
        p, keep = payload(scenario, seed, True)
        m = contact_gates.measure(p, keep)
        self.assertLessEqual(m["handhold"]["max_m"], contact_gates.HANDHOLD_M)
        self.assertEqual(m["handhold"]["unreachable_frames"], 0)
        self.assertLessEqual(m["penetration"]["max_m"], contact_gates.PENETRATION_M)
        self.assertLessEqual(m["penetration"]["floor_max_m"], contact_gates.PENETRATION_M)
        self.assertLessEqual(m["skate"]["max_drift_m"], contact_gates.SKATE_M)
        self.assertEqual(m["skate"]["lift_heel_up_frames"], 0)
        self.assertEqual(m["skate"]["floor_drag_frames"], 0)                 # no swing skims the floor
        self.assertEqual(m["joints"]["knee_frames"] + m["joints"]["elbow_frames"] + m["joints"]["hip_frames"], 0)
        self.assertEqual(m["pops"]["feet"]["frames"] + m["pops"]["hands"]["frames"], 0)
        self.assertLessEqual(m["balance"]["outside_share"], contact_gates.COM_OUT_SHARE)
        return m

    def test_carry(self):
        m = self.check_run("A-CARRY", 0)
        self.assertGreater(m["handhold"]["objects"]["sack"]["frames"], 500)   # the sack is held for real

    def test_ladder(self):
        m = self.check_run("A-LADDER", 0)
        self.assertIn("ladder_rails", m["handhold"]["objects"])
        self.assertIn("stepladder", m["handhold"]["objects"])
        p, _ = payload("A-LADDER", 0, True)
        self.assertGreater(p["meta"]["contact"]["rail_regrips"], 4)          # hand over hand

    def test_lifts_stay_on_near_straight_arms(self):
        """Step 2's lift: the load hangs on near-straight arms at the lowest posture and rises with the
        body. A-CARRY seed 1 grips low at the lift and higher at the put-down (and back), which a
        single grip height per carry turned into a 93-110 deg arm curl (review 1)."""
        p, _ = payload("A-CARRY", 1, True)
        J = np.array(p["frames"]["joints"]).reshape(-1, 18, 3)
        fps = p["meta"]["fps"]

        def elbow(k):
            return max(float(180.0 - np.degrees(np.arccos(np.clip(np.dot(
                unit(J[k, 10 + s] - J[k, 12 + s]), unit(J[k, 14 + s] - J[k, 12 + s])), -1.0, 1.0)))) for s in (0, 1))
        truth = p["truth"]
        bends = [e for e in truth if e["name"] in ("lift_grasp", "put_down_touch")]
        self.assertGreaterEqual(len(bends), 6)
        for e in bends:
            self.assertLess(elbow(int(round(e["t"] * fps))), 40.0, e)
        for e in (e for e in truth if e["name"] == "lift_off"):
            end = next(x["t"] for x in truth if x["name"] == "lift_end" and x["t"] > e["t"])
            k0, k1 = int(round(e["t"] * fps)), int(round(end * fps))
            self.assertLess(max(elbow(k) for k in range(k0, k1)), 40.0, e["t"])
            hands_y = J[k0:k1 + int(1.5 * fps), 14:16, 1].mean(axis=1)
            drops = -np.diff(hands_y)
            self.assertLess(float(drops[drops > 0].sum()), 0.05, e["t"])        # no curl, then no drop back

    def test_without_the_pass_the_old_flaws_show(self):
        # The gates measure something: the procedural rig alone slides the hands on the sack.
        p, keep = payload("A-CARRY", 0, False)
        m = contact_gates.measure(p, keep)
        self.assertGreater(m["handhold"]["max_m"], 0.05)


if __name__ == "__main__":
    unittest.main()
