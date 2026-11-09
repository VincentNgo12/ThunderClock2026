# File: test_transitions.py
# Purpose: Checks for the voluntary posture transitions tuned against Erciyes (lying down on
#          the back, sitting down hard): end postures, the voluntary rate cap and contact shapes.
# Contributors: Unassigned
"""Simulated motion only; nothing here is a medical claim."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thundersim.motions import (SIT_LEAN_DEG, SIT_SEATED_PITCH_DEG, Builder, lowering,  # noqa: E402
                                minjerk)
from thundersim.scenarios import begin  # noqa: E402
from thundersim.site import Site  # noqa: E402
from thundersim.spine import MAX_VOLUNTARY_RATE, Worker  # noqa: E402


class LoweringProfile(unittest.TestCase):
    def test_zero_end_rate_is_minimum_jerk(self):
        np.testing.assert_allclose(lowering(500, 0.0), minjerk(500), atol=1e-12)

    def test_ends_still_moving_and_never_turns_back(self):
        k = 1000
        for end_rate in (0.4, 1.0, 1.6):
            u = lowering(k, end_rate)
            rate = np.diff(u) * (k - 1)
            self.assertAlmostEqual(u[-1], 1.0, places=9)
            self.assertGreaterEqual(rate.min(), -1e-9)
            self.assertAlmostEqual(rate[-1], end_rate, delta=0.01)


class LieDown(unittest.TestCase):
    def build_lie(self, seed):
        b = Builder(Worker(), Site(), np.random.default_rng(seed))
        begin(b)
        b.stand(2.0)
        start = b.n
        b.lie("supine", 3.0)
        return b, b.build(), start

    def test_ends_supine_on_the_ground(self):
        b, track, _ = self.build_lie(0)
        self.assertAlmostEqual(np.rad2deg(track.pitch[-1]), -88.0, delta=1.0)
        self.assertAlmostEqual(track.root_pos[-1, 1], b.ground_y + 0.03, delta=0.01)

    def test_lowering_respects_the_voluntary_rate_cap(self):
        for seed in range(3):
            _, track, start = self.build_lie(seed)
            rate = np.abs(np.diff(track.pitch[start:])) * track.fs
            self.assertLess(rate.max(), MAX_VOLUNTARY_RATE * 1.05)

    def test_back_touches_the_ground_still_moving(self):
        # A min-jerk lie-back arrives at rest; the new one reaches the ground at a few tenths of
        # a m/s and stops within a few contact time constants.
        b, track, start = self.build_lie(1)
        y = track.root_pos[start:, 1]
        vy = np.gradient(y, 1.0 / track.fs)
        near_ground = y < b.ground_y + 0.06       # only the last few cm of the lowering
        self.assertLess(vy[near_ground].min(), -0.2)
        self.assertLess(np.abs(vy[-200:]).max(), 0.02)


class HardSit(unittest.TestCase):
    def test_leans_forward_then_sits_back(self):
        b = Builder(Worker(), Site(), np.random.default_rng(5))
        begin(b)
        b.stand(2.0)
        start = b.n
        b.sit(0.45, hard=True)
        track = b.build()
        trunk = np.rad2deg(track.pitch[start:] + track.flex[start:].sum(axis=1))
        self.assertGreater(trunk.max(), 0.8 * SIT_LEAN_DEG)
        self.assertAlmostEqual(np.rad2deg(track.pitch[-1]), SIT_SEATED_PITCH_DEG, delta=1.0)


if __name__ == "__main__":
    unittest.main()
