# File: test_physics.py
# Purpose: Physical sanity of the spine model and fall physics (gravity, free fall, harness
#          limits, surface ordering, no artefacts at motion joins, spine curve).
# Contributors: Unassigned

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thundersim import falls  # noqa: E402
from thundersim.model_api import GRAVITY  # noqa: E402
from thundersim.motions import Builder, knee_class  # noqa: E402
from thundersim.scenarios import begin  # noqa: E402
from thundersim.site import HARNESS_MAX_DECEL_DISTANCE_M, HARNESS_MAX_FORCE_N, Site  # noqa: E402
from thundersim.spine import Worker, random_mounts, spine_curve, strip_kinematics, virtual_imu  # noqa: E402


def imu_for(builder: Builder, worker: Worker, seed: int = 0):
    track = builder.build()
    kin = strip_kinematics(track, worker, random_mounts(worker, np.random.default_rng(seed)))
    acc, gyro = virtual_imu(kin, track.fs)
    return track, kin, acc, gyro


class PhysicsTests(unittest.TestCase):
    def test_standing_reads_one_g_and_no_rotation(self):
        worker = Worker()
        b = Builder(worker, Site(), np.random.default_rng(1))
        begin(b)
        b.stand(5.0)
        _, _, acc, gyro = imu_for(b, worker)
        mid = slice(500, -500)
        self.assertLess(np.abs(np.linalg.norm(acc[mid], axis=2) / GRAVITY - 1).max(), 0.05)
        self.assertLess(np.linalg.norm(gyro[mid], axis=2).max(), 0.1)

    def test_walking_has_no_join_artefacts(self):
        worker = Worker()
        b = Builder(worker, Site(), np.random.default_rng(2))
        begin(b)
        b.walk(8.0)
        b.stoop(70)
        b.kneel(30)
        b.stand_up()
        b.hammer(5.0)
        _, _, acc, _ = imu_for(b, worker)
        self.assertLess(np.linalg.norm(acc[50:-50], axis=2).max() / GRAVITY, 3.5)

    def test_free_fall_time_matches_sqrt_2h_over_g(self):
        worker = Worker()
        b = Builder(worker, Site(), np.random.default_rng(3))
        begin(b, height=4.0)
        f = falls.Faller(b, "forward")
        start_y = f.com[1]
        f.flight(landing_y=0.0, pelvis_stop=worker.pelvis_height - 0.05)
        start, end = (t for t, n, _ in f.events if n.startswith("free_fall"))
        drop = start_y - f.com[1]
        self.assertAlmostEqual(end - start, np.sqrt(2 * drop / GRAVITY), delta=0.01)

    def test_harness_arrest_respects_osha_limits(self):
        for seed in range(4):
            b = Builder(Worker(mass_kg=120, load_kg=15), Site(), np.random.default_rng(seed))
            begin(b, height=6.0)
            info = falls.height_fall(b, 6.0, start="edge", protection="harness")
            arrest = [e for e in b.events if e[1] == "arrest_end"][0][2]
            self.assertLessEqual(info["free_fall_m"], 1.8)
            self.assertLessEqual(arrest["decel_distance_m"], HARNESS_MAX_DECEL_DISTANCE_M + 1e-6)
            self.assertLessEqual(arrest["force_n"], HARNESS_MAX_FORCE_N)

    def test_harder_surface_gives_higher_impact(self):
        peaks = {}
        for surface in ("concrete", "crash_mat"):
            b = Builder(Worker(), Site(surface=surface), np.random.default_rng(11))
            begin(b)
            b.stand(1.0)
            peaks[surface] = falls.same_level_fall(b, "left", surface=surface)["peak_pelvis_g"]
        self.assertGreater(peaks["concrete"], peaks["crash_mat"] * 1.2)

    def test_same_level_hip_speed_near_consensus(self):
        speeds = []
        for seed in range(8):
            b = Builder(Worker(), Site(), np.random.default_rng(100 + seed))
            begin(b)
            b.stand(1.0)
            speeds.append(falls.same_level_fall(b, "backward", slip=True)["hip_speed_ms"])
        self.assertTrue(2.8 < np.mean(speeds) < 4.2, speeds)  # consensus 3.4 +/- 0.5 m/s

    def test_hermite_spine_curve_passes_through_anchors(self):
        worker = Worker()
        b = Builder(worker, Site(), np.random.default_rng(4))
        begin(b)
        b.stoop(60, hold_s=0.5)
        _, kin, _, _ = imu_for(b, worker)
        i = 5600
        anchors = kin.anchors[i]
        axes = np.stack([kin.seg_rot[s][i].apply([0, 1, 0]) for s in range(4)])
        curve = spine_curve(anchors, axes, points=301)
        for anchor in anchors:
            self.assertLess(np.linalg.norm(curve - anchor, axis=1).min(), 0.01)


class LiftTests(unittest.TestCase):
    """Builder.lift / put_down: technique sets the pelvis drop, the rise is leg-driven, joins stay
    smooth and carry() keeps the run's main random stream and timing."""

    def lift(self, technique: str, object_height: float = 0.15):
        worker = Worker()
        b = Builder(worker, Site(), np.random.default_rng(6))
        begin(b)
        b.lift(object_height, 20.0, technique=technique)
        b.stand(1.0)
        events = {n: t for t, n, _ in b.events}
        return b, worker, events

    def test_technique_sets_the_pelvis_drop(self):
        drops = {}
        for technique in ("stoop", "semi_squat", "squat"):
            b, _, events = self.lift(technique)
            track = b.build()
            k = int(events["lift_grasp"] * track.fs)
            drops[technique] = track.root_pos[100, 1] - track.root_pos[k, 1]
        self.assertLess(drops["stoop"], drops["semi_squat"])
        self.assertLess(drops["semi_squat"], drops["squat"])
        self.assertLess(drops["stoop"], 0.15)
        self.assertTrue(0.3 < drops["squat"] < 0.5, drops)

    def test_semi_squat_floor_lift_matches_self_selected_lifts(self):
        """Lowest posture of a semi-squat to a 0.15 m grip, against self-selected floor lifts
        (Saraceni 2021, no-LBP group, loaded box): knee and pelvis inside the 95 % CIs, thorax
        within 10 deg of the mean."""
        b = Builder(Worker(), Site(), np.random.default_rng(0))
        low = b.lift_posture(0.15, "semi_squat", 24.2, 0.38)
        self.assertTrue(73.2 <= np.degrees(low["knee"]) <= 98.6, np.degrees(low["knee"]))
        self.assertTrue(45.0 <= np.degrees(low["pitch"]) <= 57.6, np.degrees(low["pitch"]))
        self.assertLess(abs(np.degrees(low["trunk"]) - 102.8), 10.0)

    def test_events_name_the_knee_class_reached(self):
        """A planned stoop to a low grip reaches the trunk cap and bends the knees past 45 deg: the
        events keep the planned technique and name the knee class reached (Straker 2003 classes)."""
        b, _, _ = self.lift("stoop", object_height=0.15)
        info = next(i for _, n, i in b.events if n == "lift_grasp")
        self.assertEqual(info["technique"], "stoop")
        self.assertGreaterEqual(info["knee_deg"], 45.0)
        self.assertEqual(info["knee_class"], "semi_squat")
        for knee, name in ((44.9, "stoop"), (45.0, "semi_squat"), (112.4, "semi_squat"), (112.5, "squat")):
            self.assertEqual(knee_class(knee), name)

    def test_rise_is_leg_driven(self):
        """Knees extend before the trunk (Burgess-Limerick 1995): the pelvis is half way up before
        the upper trunk is half way back to upright."""
        b, _, events = self.lift("semi_squat")
        track = b.build()
        k0, k1 = int(events["lift_off"] * track.fs), int(events["lift_end"] * track.fs)
        y = track.root_pos[k0:k1, 1]
        trunk = track.pitch[k0:k1] + track.flex[k0:k1].sum(axis=1)
        half_y = np.argmax(y - y[0] >= 0.5 * (y.max() - y[0]))
        half_trunk = np.argmax(trunk[0] - trunk >= 0.5 * (trunk[0] - trunk.min()))
        self.assertLess(half_y, half_trunk - 0.1 * track.fs)

    def test_lift_has_no_join_artefacts(self):
        for technique in ("stoop", "squat"):
            b, worker, _ = self.lift(technique)
            b.put_down(0.3, 20.0, technique=technique)
            _, _, acc, gyro = imu_for(b, worker)
            self.assertLess(np.linalg.norm(acc[50:-50], axis=2).max() / GRAVITY, 2.0, technique)
            self.assertLess(np.linalg.norm(gyro[50:-50], axis=2).max(), 3.5, technique)

    def test_carry_keeps_the_main_random_stream_and_timing(self):
        """carry() draws from the main stream exactly what the old stoop-walk-stoop did, and a
        lift with total_s lasts exactly total_s."""
        worker = Worker()
        new = Builder(worker, Site(), np.random.default_rng(8))
        begin(new)
        new.carry(6.0, 20.0)
        old = Builder(worker, Site(), np.random.default_rng(8))
        begin(old)
        for walk in (True, False):                       # the old stoop: 1.2 s down, hold, 1.2 s up
            old.rng.uniform(60, 90)
            old.hold(1.2)
            old.hold(old.rng.uniform(1, 3))
            old.hold(1.2)
            if walk:
                old.walk(6.0, load_kg=20.0)
        self.assertEqual(new.n, old.n)
        self.assertEqual(new.rng.bit_generator.state, old.rng.bit_generator.state)


if __name__ == "__main__":
    unittest.main()
