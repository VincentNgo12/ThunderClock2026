# File: test_viz_mocap.py
# Purpose: Mocap limbs in the viewer (thundersim/viz/mocap.py): off by default, never change the
#          simulated ground truth, keep the rig's bone lengths, plant clip-shaped steps on the
#          simulated heel strikes without skate, hit the hammer impulses, and the warp maps
#          anchors exactly and fades the jumps where a clip loops.
# Contributors: Unassigned

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thundersim.viz import mocap, rig, server  # noqa: E402
from thundersim.viz.export import run_payload  # noqa: E402

OBS = 5.0
J = {name: i for i, name in enumerate(rig.JOINTS)}
GROUND_TRUTH = ("spine", "segq", "anchors", "chipq", "phase", "worn")
HAVE_MANIFEST = mocap.MANIFEST.exists()


def joints(payload) -> np.ndarray:
    return np.array(payload["frames"]["joints"]).reshape(-1, len(rig.JOINTS), 3)


class TestWarp(unittest.TestCase):
    def test_anchors_map_exactly_and_outside_runs_at_real_speed(self):
        w = mocap.Warp([10.0, 11.0, 12.0], [2.0, 3.0], [3.0, 3.5], 0.0, 10.0)
        _, _, tau = w.locate(np.array([10.0, 10.5, 11.0, 11.5, 12.0, 13.0, 9.0]))
        np.testing.assert_allclose(tau, [2.0, 2.5, 3.0, 3.25, 3.5, 4.5, 1.0])

    def test_a_loop_jump_fades_out(self):
        # Segment 1 starts at clip time 0 although segment 0 ended at 2: the signal f(tau) = tau
        # would jump by -2 at t = 1; the offset removes the jump and fades out.
        w = mocap.Warp([0.0, 1.0, 2.0], [1.0, 0.0], [2.0, 1.0], 0.0, 5.0)
        t = np.array([0.98, 1.0, 1.0 + mocap.JUNCTION_BLEND_S, 1.5])
        out = w.sample(lambda x: x.copy(), t)
        self.assertAlmostEqual(out[1], 2.0)                    # continuous at the junction
        self.assertAlmostEqual(out[2], mocap.JUNCTION_BLEND_S)  # offset gone after the blend
        self.assertAlmostEqual(out[3], 0.5)

    def test_foot_roll_combination(self):
        self.assertEqual(mocap.combine_pitch(0.0, 0.3), 0.3)       # clip heel strike
        self.assertEqual(mocap.combine_pitch(-0.5, -0.2), -0.5)    # the hip needs more heel lift
        self.assertEqual(mocap.combine_pitch(0.2, -0.4), 0.2)      # opposite roll: the hip wins


class TestSwitch(unittest.TestCase):
    def test_off_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(mocap.ENV_SWITCH, None)
            self.assertFalse(mocap.requested())
            self.assertTrue(mocap.requested(True))
            os.environ[mocap.ENV_SWITCH] = "1"
            self.assertTrue(mocap.requested())
            self.assertFalse(mocap.requested(False))

    def test_cache_key_follows_the_manifest(self):
        with mock.patch.object(server.mocap_lib, "manifest_bytes", return_value=b"a"):
            first = server.code_version()
        with mock.patch.object(server.mocap_lib, "manifest_bytes", return_value=b"b"):
            second = server.code_version()
        self.assertNotEqual(first, second)

    def test_a_clip_with_the_wrong_sha256_is_not_used(self):
        if not HAVE_MANIFEST:
            self.skipTest("no motionlib/manifest.json")
        import json
        manifest = json.loads(mocap.MANIFEST.read_text())
        entry = dict(manifest["clips"][0], sha256="0" * 64)
        library = mocap.Library({"clips": [entry]})
        self.assertIsNone(library.clip(entry))
        self.assertTrue(any("sha256" in p for p in library.problems))


@unittest.skipUnless(HAVE_MANIFEST, "no motionlib/manifest.json")
class TestMocapPayload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.keep = {}
        cls.off = run_payload("A-CARRY", 0, "none", "target", OBS, mocap=False)
        cls.on = run_payload("A-CARRY", 0, "none", "target", OBS, mocap=True, keep=cls.keep)
        cls.hammer_keep = {}
        cls.hammer = run_payload("A-HAMMER", 0, "none", "target", OBS, mocap=True, keep=cls.hammer_keep)

    def test_ground_truth_is_identical_with_mocap_on_and_off(self):
        for name in GROUND_TRUTH:
            self.assertEqual(self.off["frames"][name], self.on["frames"][name], name)
        self.assertEqual(self.off["signals"], self.on["signals"])
        self.assertEqual(self.off["truth"], self.on["truth"])

    def test_clips_are_used_logged_and_credited(self):
        meta = self.on["meta"]["mocap"]
        self.assertTrue(meta["enabled"])
        roles = {u["role"] for u in meta["uses"]}
        self.assertTrue({"walk", "carry_walk"} & roles, roles)
        self.assertTrue(all(u["licence_class"] == "A" for u in meta["uses"]))
        self.assertTrue(any("motion capture" in n for n in self.on["meta"]["notes"]))
        self.assertFalse(self.off["meta"]["mocap"]["enabled"])
        self.assertEqual(self.off["meta"]["mocap"]["uses"], [])

    def test_bone_lengths_stay_the_rigs(self):
        d = self.keep["rig"].dims
        j = joints(self.on)
        for s in ("l", "r"):
            for a, b, length in (("hip", "knee", d.thigh), ("knee", "ankle", d.shank),
                                 ("shoulder", "elbow", d.upper_arm), ("elbow", "hand", d.forearm)):
                err = np.abs(np.linalg.norm(j[:, J[f"{b}_{s}"]] - j[:, J[f"{a}_{s}"]], axis=1) - length)
                self.assertLess(err.max(), 1e-3, f"{a}-{b} {s}")

    def test_clip_steps_land_on_the_simulated_strikes_and_stay_planted(self):
        session = self.keep["session"]
        planted = self.keep["rig"].targets["planted"]
        j = joints(self.on)
        t = np.arange(len(j)) / self.on["meta"]["fps"]
        self.assertTrue(session.walk_plans)
        for plan in session.walk_plans.values():
            for side, t_lift, t_land in plan.step_times[1:]:
                k = int(np.searchsorted(t, t_land))
                if not planted[k:k + 3, side].all():
                    continue                            # the rig took its own step here
                # Planted from touchdown: the heel stays put for the next frames.
                heel = j[k:k + 5, J["heel_l" if side == 0 else "heel_r"]][:, [0, 2]]
                heel = heel[planted[k:k + 5, side]]
                self.assertLess(np.linalg.norm(heel - heel[0], axis=1).max(), 0.01)

    def test_hammer_hand_reaches_the_impact_point_at_each_impulse(self):
        session = self.hammer_keep["session"]
        if not any(u["role"] == "hammer" for u in self.hammer["meta"]["mocap"]["uses"]):
            self.skipTest("no hammer clip listed")
        rb = session.rb
        impact, _ = rb._hammer_points(slice(0, rb.N), overhead=False)
        hand = joints(self.hammer)[:, J["hand_r"]]
        fps = self.hammer["meta"]["fps"]
        rec = self.hammer_keep["ctx"].rec
        errors = []
        for call in rec.find("hammer"):
            for x in rec.find("impulses", inside=call):
                for s in x.args.get("times_abs", []):
                    k = int(round(s * fps))
                    near = np.linalg.norm(hand[k - 2:k + 3] - impact[k - 2:k + 3], axis=1)
                    errors.append(abs(int(np.argmin(near)) - 2) / fps)
        self.assertTrue(errors)
        self.assertLessEqual(float(np.median(errors)), 0.02)


if __name__ == "__main__":
    unittest.main()
