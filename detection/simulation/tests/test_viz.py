# File: test_viz.py
# Purpose: The 3D viewer replays exactly the simulated run (same track, same model events),
#          and its illustrative rig keeps plausible geometry (bone lengths, feet on the ground).
# Contributors: Unassigned

import sys
import unittest
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thundersim.runner import StreamOptions, run_scenario  # noqa: E402
from thundersim.scenarios import build_run  # noqa: E402
from thundersim.viz.export import run_payload  # noqa: E402
from thundersim.viz.recorder import RecordingBuilder  # noqa: E402
from thundersim.viz.rig import two_bone  # noqa: E402

OBS = 20.0


class RecordingIsTransparent(unittest.TestCase):
    def test_recording_builder_does_not_change_the_simulation(self):
        for scenario in ("H-LAD", "STAIR", "A-EQUIP", "A-CARRY", "HAT-OFF"):
            plain = build_run(scenario, 2, obs_s=OBS)
            sink: list = []
            recorded = build_run(scenario, 2, obs_s=OBS, builder=partial(RecordingBuilder, sink=sink))
            np.testing.assert_array_equal(plain.track.root_pos, recorded.track.root_pos, err_msg=scenario)
            np.testing.assert_array_equal(plain.track.flex, recorded.track.flex, err_msg=scenario)
            self.assertEqual([e[:2] for e in plain.events], [e[:2] for e in recorded.events])
            self.assertTrue(sink[0].calls, scenario)

    def test_viewer_scores_like_the_cli(self):
        cli = run_scenario("L-TRIP", 3, "baseline", StreamOptions(profile="target", fast=True), obs_s=OBS)
        payload = run_payload("L-TRIP", 3, "baseline", "target", OBS)
        self.assertEqual(payload["result"]["output"], cli.output)
        self.assertEqual(payload["result"]["latency_s"], cli.latency_s)
        self.assertEqual([e["kind"] for e in payload["model_events"]], [e["kind"] for e in cli.events])


class RigGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payloads = {s: run_payload(s, 1, "none", "target", OBS) for s in ("H-SCAF", "A-LADDER", "DEVICE-OFF")}

    def joints(self, payload):
        n = payload["meta"]["frames"]
        return np.array(payload["frames"]["joints"]).reshape(n, 18, 3)

    def test_bone_lengths_are_constant(self):
        for scenario, payload in self.payloads.items():
            j = self.joints(payload)
            body = payload["meta"]["body"]
            thigh = np.linalg.norm(j[:, 2] - j[:, 0], axis=1)
            upper_arm = np.linalg.norm(j[:, 12] - j[:, 10], axis=1)
            self.assertLess(np.abs(thigh - body["thigh"]).max(), 1e-3, scenario)
            self.assertLess(np.abs(upper_arm - body["upper_arm"]).max(), 1e-3, scenario)

    def test_feet_rest_on_the_floor_during_the_calibration_stand(self):
        payload = self.payloads["DEVICE-OFF"]
        j = self.joints(payload)
        stand = slice(10, int(4.5 * payload["meta"]["fps"]))
        ankle_h = payload["meta"]["body"]["ankle_h"]
        self.assertLess(np.abs(j[stand, 4:6, 1] - ankle_h).max(), 0.02)  # ankles at ankle height, ground at y = 0
        self.assertGreater(j[stand, 6:10, 1].min(), -0.01)                # heels and toes not below the ground

    def test_payload_has_scene_sensors_and_events(self):
        payload = self.payloads["H-SCAF"]
        self.assertTrue(any(p["t"] == "box" for p in payload["scene"]["props"]))
        self.assertTrue(payload["scene"]["solids"], "the scaffold deck is walkable")
        n = payload["meta"]["frames"]
        self.assertEqual(len(payload["frames"]["anchors"]), n * 12)
        self.assertEqual(len(payload["signals"]["acc_g"]), len(payload["signals"]["t"]) * 4)
        self.assertIn("impact", [e["name"] for e in payload["truth"]])


class TwoBoneIK(unittest.TestCase):
    def test_reachable_target_is_hit_with_exact_bone_lengths(self):
        root = np.zeros((1, 3))
        target = np.array([[0.3, -0.6, 0.1]])
        pole = np.array([[0.0, -0.3, -1.0]])
        mid, end = two_bone(root, target, pole, 0.45, 0.42)
        np.testing.assert_allclose(end, target, atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.norm(mid - root)), 0.45, places=9)
        self.assertAlmostEqual(float(np.linalg.norm(end - mid)), 0.42, places=9)
        self.assertLess(float(mid[0, 2]), 0.0, "the joint bends toward the pole")


if __name__ == "__main__":
    unittest.main()
