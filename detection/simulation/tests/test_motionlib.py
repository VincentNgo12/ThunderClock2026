# File: test_motionlib.py
# Purpose: The motion library's own readers and trunk conversion, on small synthetic inputs (no
#          downloads needed): BVH parsing, rigid resampling keeps bone lengths, and the trunk
#          angles follow the simulator's spine.root_rotation / spine.joint_rotation recipe.
#          Stages S3-S5: contact hysteresis, lift technique classes, the floor plane fit, the
#          sole direction, the rig foot used by the gates, and the cards' gate status.
#          Stage S6: the manifest lists only clips that may drive the viewer, with their sha256,
#          and none is on by default without a reviewer.
# Contributors: Unassigned

import json
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SIM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM_DIR))
sys.path.insert(0, str(SIM_DIR / "motionlib" / "tools"))

import annotate  # noqa: E402
import canonical  # noqa: E402
import features  # noqa: E402
import gates  # noqa: E402
import manifest  # noqa: E402
import payload_gates  # noqa: E402
import readers  # noqa: E402
from cliplib import CARDS, CLIPS, heading_frame, runs  # noqa: E402
from thundersim.spine import joint_rotation, root_rotation  # noqa: E402

FRAMES = 40
FRAME_TIME = 1.0 / 30.0


def two_bone_bvh() -> str:
    """Root with a 0.4 m and a 0.3 m bone swinging fast about Z (1 radian amplitude at 3 Hz)."""
    lines = ["HIERARCHY", "ROOT Hips", "{", "OFFSET 0 0 0",
             "CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation",
             "JOINT Upper", "{", "OFFSET 0 -40 0", "CHANNELS 3 Zrotation Xrotation Yrotation",
             "End Site", "{", "OFFSET 0 -30 0", "}", "}", "}",
             "MOTION", f"Frames: {FRAMES}", f"Frame Time: {FRAME_TIME:.8f}"]
    for k in range(FRAMES):
        t = k * FRAME_TIME
        hip_z = np.degrees(np.sin(2 * np.pi * 3.0 * t))
        knee_z = np.degrees(0.8 * np.sin(2 * np.pi * 3.0 * t + 1.0))
        lines.append(f"{10 * t:.5f} 90 0 {hip_z:.5f} 0 0 {knee_z:.5f} 0 0")
    return "\n".join(lines) + "\n"


class ReaderTests(unittest.TestCase):
    def test_bvh_reads_hierarchy_and_frames(self):
        motion = readers.read_bvh(two_bone_bvh())
        self.assertEqual(motion.names, ["Hips", "Upper"])
        self.assertEqual(motion.parent, [-1, 0])
        self.assertEqual(motion.start.shape, (FRAMES, 2, 3))
        self.assertAlmostEqual(motion.fps, 30.0, places=3)
        lengths = np.linalg.norm(motion.end[:, 1] - motion.start[:, 1], axis=1)
        np.testing.assert_allclose(lengths, 30.0, atol=1e-9)

    def test_rigid_resampling_keeps_bone_lengths(self):
        motion = readers.resample_motion(readers.read_bvh(two_bone_bvh()), 100.0)
        self.assertEqual(motion.fps, 100.0)
        self.assertEqual(len(motion.start), int(np.floor((FRAMES - 1) * FRAME_TIME * 100 + 1e-3)) + 1)
        upper = np.linalg.norm(motion.start[:, 1] - motion.start[:, 0], axis=1)
        lower = np.linalg.norm(motion.end[:, 1] - motion.start[:, 1], axis=1)
        self.assertLess(np.abs(upper - 40.0).max(), 1e-9)
        self.assertLess(np.abs(lower - 30.0).max(), 1e-9)

    def test_bad_bvh_is_rejected(self):
        text = two_bone_bvh().replace(f"Frames: {FRAMES}", f"Frames: {FRAMES + 5}")
        with self.assertRaises(ValueError):
            readers.read_bvh(text)


class TrunkConventionTests(unittest.TestCase):
    def test_sim_angles_inverts_the_simulator_builders(self):
        rng = np.random.default_rng(3)
        n = 200
        heading = rng.uniform(-np.pi, np.pi, n)
        pitch, roll = rng.uniform(-0.6, 1.2, n), rng.uniform(-0.3, 0.3, n)
        flex, lat, twist = (rng.uniform(-0.3, 0.5, (n, 3)), rng.uniform(-0.2, 0.2, (n, 3)),
                            rng.uniform(-0.3, 0.3, (n, 3)))
        seg = [root_rotation(heading, pitch, roll)]
        for j in range(3):
            seg.append(seg[-1] * joint_rotation(flex[:, j], lat[:, j], twist[:, j]))
        angles = canonical.sim_angles(seg)
        np.testing.assert_allclose(angles[:, 0], heading, atol=1e-9)
        np.testing.assert_allclose(angles[:, 1], pitch, atol=1e-9)
        np.testing.assert_allclose(angles[:, 2], roll, atol=1e-9)
        np.testing.assert_allclose(angles[:, 3:6], flex, atol=1e-9)
        np.testing.assert_allclose(angles[:, 6:9], lat, atol=1e-9)
        np.testing.assert_allclose(angles[:, 9:12], twist, atol=1e-9)

    def test_calibration_makes_the_neutral_stand_read_heading_only(self):
        rng = np.random.default_rng(5)
        frames = [Rotation.random(50, random_state=rng.integers(1 << 30)).as_matrix() for _ in range(4)]
        neutral = canonical.window_means(frames, 10, 20)
        seg = canonical.calibrate(frames, neutral, heading=0.7)
        for s in seg:      # the window's mean rotation of every segment becomes Ry(heading)
            mid = s[10:20].mean()
            self.assertLess((mid.inv() * Rotation.from_rotvec([0.0, 0.7, 0.0])).magnitude(), 1e-6)

    def test_heading_of_right_matches_root_rotation(self):
        for heading in (-2.5, -0.4, 0.0, 1.1, 3.0):
            right = Rotation.from_rotvec([0.0, heading, 0.0]).apply([1.0, 0.0, 0.0])
            self.assertAlmostEqual(canonical.heading_of_right(right), heading, places=9)



class FakeClip:
    """The few Clip attributes estimate_floor needs, for synthetic foot paths."""

    def __init__(self, points: dict, fs: float = 100.0):
        self.points, self.fs = points, fs
        self.n = len(next(iter(points.values())))

    def has(self, *names):
        return all(n in self.points for n in names)

    def j(self, name):
        return self.points[name]


class AnnotateTests(unittest.TestCase):
    def test_hysteresis_fills_short_gaps_and_drops_short_runs(self):
        enter = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0], bool)
        stay = np.array([0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0], bool)
        out = annotate.hysteresis(enter, stay, min_frames=3, gap=1)
        # 1-4 stay on (stay holds at 4), the one-frame gap at 5 is filled, 11 alone is too short.
        self.assertEqual(runs(out), [(1, 9)])

    def test_lift_technique_follows_knee_flexion_classes(self):
        self.assertEqual(annotate.technique_of(30.0), "stoop")
        self.assertEqual(annotate.technique_of(44.9), "stoop")
        self.assertEqual(annotate.technique_of(45.0), "semi_squat")
        self.assertEqual(annotate.technique_of(90.0), "semi_squat")
        self.assertEqual(annotate.technique_of(112.5), "squat")
        self.assertEqual(annotate.LIFT_TECHNIQUES, ("stoop", "semi_squat", "squat"))

    def test_floor_plane_recovers_a_tilted_floor_and_ladders_stay_level(self):
        # Feet that stand still in turn while walking 4 m along -Z over a floor rising 3 cm per metre.
        n = 800
        z = -np.linspace(0.0, 4.0, n)
        floor_y = 0.05 - 0.03 * z
        points = {}
        for side, phase in (("l", 0.0), ("r", np.pi)):
            stance = np.sin(np.linspace(0, 16 * np.pi, n) + phase) > 0
            zz = np.where(stance, np.round(z / 0.5) * 0.5, z)    # parked on a spot while in stance
            lift = np.where(stance, 0.0, 0.08)
            for part, dz in (("heel", 0.0), ("toe", -0.2)):
                y = 0.05 - 0.03 * (zz + dz) + lift
                points[f"{part}_{side}"] = np.stack([np.zeros(n), y, zz + dz], axis=1)
        plane = annotate.estimate_floor(FakeClip(points), root_relative=False, ladder=False)
        heights = annotate.above_floor(plane, np.stack([np.zeros(n), floor_y, z], axis=1))
        self.assertLess(np.abs(heights).max(), 0.01)
        self.assertAlmostEqual(annotate.floor_slope_deg(plane), np.degrees(np.arctan(0.03)), delta=0.3)
        level = annotate.estimate_floor(FakeClip(points), root_relative=False, ladder=True)
        self.assertEqual(level[1], 0.0)
        self.assertEqual(level[2], 0.0)
        self.assertTrue(np.isnan(annotate.estimate_floor(FakeClip(points), root_relative=True, ladder=False)).all())


class FeatureAndGateTests(unittest.TestCase):
    def test_sole_direction_reads_flat_and_stays_continuous_when_pointing_down(self):
        n = 50
        R = heading_frame(np.zeros(n))
        ankle = np.zeros((n, 3))
        knee = np.tile([0.0, 0.45, 0.0], (n, 1))
        # Skeleton toe 6 cm below the ankle-height line when flat; then the foot points down by up
        # to 60 deg (toe-off, kneeling on the toes), past the point where d x up would flip.
        angle = np.linspace(0.0, np.deg2rad(60.0), n)
        flat_pitch = np.arctan2(-0.06, 0.15)
        pitch = flat_pitch - angle
        toe = np.stack([np.zeros(n), np.sin(pitch), -np.cos(pitch)], axis=1) * 0.16
        u, nrm, yaw, sole_pitch = features.sole_direction(ankle, toe, knee, flat_pitch, R)
        np.testing.assert_allclose(u[0], [0.0, 0.0, 1.0], atol=1e-9)          # forward in the heading frame
        np.testing.assert_allclose(nrm[0], [0.0, 1.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(sole_pitch, -angle, atol=1e-9)
        np.testing.assert_allclose(yaw, 0.0, atol=1e-9)
        np.testing.assert_allclose(np.sum(u * nrm, axis=1), 0.0, atol=1e-12)

    def test_gate_foot_matches_the_rig_foot(self):
        d = gates.Dims(1.75, 80.0)
        ankle = np.array([[0.1, d.ankle_h, -0.2]])
        u = np.array([[0.0, 0.0, -1.0]])
        nrm = np.array([[0.0, 1.0, 0.0]])
        heel, ball, toe = gates.foot_points(ankle, u, nrm, np.zeros(1), d)
        np.testing.assert_allclose(heel, [[0.1, 0.0, -0.2 + d.heel]], atol=1e-12)
        np.testing.assert_allclose(ball, [[0.1, 0.0, -0.2 - d.ball]], atol=1e-12)
        np.testing.assert_allclose(toe, [[0.1, 0.0, -0.2 - d.toe]], atol=1e-12)
        # Toes bent up by the heel lift keep the tip off the floor.
        _, _, bent = gates.foot_points(ankle, u, nrm, np.array([0.5]), d)
        self.assertGreater(bent[0, 1], 0.0)

    def test_toes_straighten_after_lift_off(self):
        contact = np.array([[True], [True], [False], [False], [False], [False], [False], [False], [False], [False]])
        pitch = np.full((10, 1), -0.6)
        bend = gates.toe_bends(contact, pitch)
        self.assertAlmostEqual(bend[1, 0], 0.6)
        self.assertTrue(np.all(np.diff(bend[1:, 0]) <= 0))
        self.assertEqual(bend[-1, 0], 0.0)

    def test_cards_status_follows_the_gates(self):
        checked = 0
        for path in sorted(CARDS.glob("*.json")):
            card = json.loads(path.read_text())
            gate_block = card.get("gates") or {}
            if "failed" not in gate_block:
                continue
            checked += 1
            failed = [k for k in gates.GATE_ORDER if gate_block[k]["result"] == "FAIL"]
            self.assertEqual(failed, gate_block["failed"], card["clip"])
            # Stage S6 measures on the real payload (feet locked) and replaces S5's skate result.
            s6 = card.get("s6", {}).get("result")
            skate_only = all(f == "skate" for f in failed)
            if card.get("problems") or s6 == "FAIL" or not skate_only:
                expected = "rejected"
            elif s6 == "PASS" or not failed:
                expected = "gated"
            else:
                expected = "rejected"
            self.assertEqual(card["status"], expected, card["clip"])
            self.assertEqual(gate_block["sync"]["result"], "PENDING", card["clip"])
            self.assertTrue((CLIPS / f"{card['clip']}.npz").exists(), card["clip"])
        if not checked:
            self.skipTest("no S5 gate results in the cards")


class TestManifest(unittest.TestCase):
    def test_listed_clips_are_eligible_verified_and_not_on_by_default(self):
        path = manifest.MANIFEST
        if not path.exists():
            self.skipTest("no motionlib/manifest.json")
        data = json.loads(path.read_text())
        for entry in data["clips"]:
            card = json.loads((CARDS / f"{entry['clip']}.json").read_text())
            self.assertEqual(entry["licence_class"], "A", entry["clip"])
            self.assertEqual(manifest.sha256(SIM_DIR / "motionlib" / entry["file"]), entry["sha256"], entry["clip"])
            self.assertTrue(all(f == "skate" for f in card["gates"]["failed"]), entry["clip"])
            self.assertNotEqual(card.get("s6", {}).get("result"), "FAIL", entry["clip"])
            self.assertEqual(entry["default_on"], bool(card.get("reviewed_by")), entry["clip"])
            self.assertTrue(entry["roles"], entry["clip"])

    def test_s6_status_rule(self):
        base = {"gates": {"failed": ["skate"]}}
        self.assertEqual(payload_gates.s6_status({**base, "s6": {"result": "PASS"}}), "gated")
        self.assertEqual(payload_gates.s6_status({**base, "s6": {"result": "FAIL"}}), "rejected")
        self.assertEqual(payload_gates.s6_status(base), "rejected")              # not measured on the payload yet
        self.assertEqual(payload_gates.s6_status({"gates": {"failed": ["skate", "hip"]}, "s6": {"result": "PASS"}}),
                         "rejected")
        self.assertEqual(payload_gates.s6_status({"gates": {"failed": []}}), "gated")


if __name__ == "__main__":
    unittest.main()
