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
from thundersim.motions import Builder  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
