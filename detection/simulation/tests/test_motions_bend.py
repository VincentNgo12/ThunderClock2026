# File: test_motions_bend.py
# Purpose: Bending primitives tuned against real motion: the rounded turn of window_ramp, the quick
#          turn at the bottom of a lift, and the squat's amplitudes and timing.
# Contributors: Unassigned

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thundersim.model_api import GRAVITY  # noqa: E402
from thundersim.motions import (SQUAT_DOWN_S, SQUAT_FLEX_DEG, SQUAT_PELVIS_DEG, SQUAT_UP_S, Builder,  # noqa: E402
                                window_ramp)
from thundersim.scenarios import begin  # noqa: E402
from thundersim.site import Site  # noqa: E402
from thundersim.spine import Worker, random_mounts, strip_kinematics, virtual_imu  # noqa: E402


def upper_trunk_deg(track) -> np.ndarray:
    """Sagittal inclination of the upper trunk: pelvis pitch plus the spinal flexion (deg)."""
    return np.degrees(track.pitch + track.flex.sum(axis=1))


class WindowRampTests(unittest.TestCase):
    def test_zero_turn_is_minimum_jerk(self):
        u = np.linspace(0.0, 1.0, 101)
        np.testing.assert_allclose(window_ramp(u, (0.0, 1.0)), 10 * u**3 - 15 * u**4 + 6 * u**5)

    def test_turn_keeps_the_ends_and_sets_the_end_acceleration(self):
        u = np.linspace(0.0, 1.0, 20001)
        du = u[1] - u[0]
        for turn in (0.0, 5.0, 10.0, 20.0):
            for key in ("arrive", "depart"):
                ramp = window_ramp(u, (0.0, 1.0), **{key: turn})
                self.assertAlmostEqual(ramp[0], 0.0)
                self.assertAlmostEqual(ramp[-1], 1.0)
                self.assertGreaterEqual(np.diff(ramp).min(), -1e-12, (key, turn))   # monotonic
                accel = np.diff(ramp, 2) / du**2
                if key == "arrive":
                    self.assertAlmostEqual(accel[-1], -turn, delta=0.01 + 0.002 * turn)
                    self.assertAlmostEqual(accel[0], 0.0, delta=0.01)
                else:
                    self.assertAlmostEqual(accel[0], turn, delta=0.01 + 0.002 * turn)
                    self.assertAlmostEqual(accel[-1], 0.0, delta=0.01)


class LiftTurnTests(unittest.TestCase):
    def test_lift_turns_round_quickly_at_the_grip(self):
        """Real bends spend 0.44 s (0.23-0.81) above 90 % of their deepest trunk angle (motionval
        mocap, low-passed). Unfiltered, this lift spends 0.79-0.93 s there; the old minimum-jerk stop
        with a 0.1-0.4 s hold spent 1.36-1.57 s."""
        for technique in ("stoop", "semi_squat", "squat"):
            b = Builder(Worker(), Site(), np.random.default_rng(6))
            begin(b)
            b.lift(0.15, 20.0, technique=technique)
            b.stand(1.0)
            track = b.build()
            trunk = upper_trunk_deg(track)
            above = np.flatnonzero(trunk >= 0.9 * trunk.max())
            self.assertLess((above[-1] - above[0]) / track.fs, 1.0, technique)


class SquatTests(unittest.TestCase):
    def squat(self, hold_s: float = 1.5):
        worker = Worker()
        b = Builder(worker, Site(), np.random.default_rng(3))
        begin(b)
        t0 = b.t
        b.squat(hold_s)
        b.stand(1.0)
        return b, worker, t0

    def test_squat_depth_and_timing(self):
        """Erciyes 804 squats: waist 20 deg, chest 34 deg; the rise is slower than the descent."""
        b, _, t0 = self.squat()
        track = b.build()
        self.assertAlmostEqual(np.degrees(track.pitch.max()), SQUAT_PELVIS_DEG, delta=0.5)
        self.assertAlmostEqual(upper_trunk_deg(track).max(), SQUAT_PELVIS_DEG + sum(SQUAT_FLEX_DEG), delta=1.5)
        self.assertGreater(SQUAT_UP_S, SQUAT_DOWN_S)
        self.assertAlmostEqual(b.t - t0, SQUAT_DOWN_S + 1.5 + SQUAT_UP_S + 1.0, delta=0.01)

    def test_squat_has_no_join_artefacts(self):
        b, worker, _ = self.squat()
        track = b.build()
        kin = strip_kinematics(track, worker, random_mounts(worker, np.random.default_rng(0)))
        acc, gyro = virtual_imu(kin, track.fs)
        self.assertLess(np.linalg.norm(acc[50:-50], axis=2).max() / GRAVITY, 1.6)
        self.assertLess(np.linalg.norm(gyro[50:-50], axis=2).max(), 2.0)


if __name__ == "__main__":
    unittest.main()
