# File: tests/test_walk_gait.py
# Purpose: Checks of the walking gait in motions.Builder.walk against the values it was tuned to
#          (eval/walk: mocap harmonic fits and Erciyes walking windows). Simulated data only.
"""Walking gait: pelvis surge (off by default), thorax counter-rotation and side bend, step-to-step variation."""

import unittest

import numpy as np

from thundersim.motions import Builder
from thundersim.site import Site
from thundersim.spine import Worker

FS = 1000.0


def walk_track(seed: int = 3, seconds: float = 12.0):
    """A steady straight walk at 1.3 m/s on concrete; returns (builder, segment, its middle slice)."""
    site = Site(surface="concrete", condition="dry", clothing="light", radio="open_site")
    b = Builder(Worker(), site, np.random.default_rng(seed), FS)
    b.hold(1.0)
    b.walk(seconds, speed=1.3)
    s = b.segs[-1]
    middle = slice(int(2 * FS), s.k - int(2 * FS))
    return b, s, middle


def stride_phase(x: np.ndarray, ref: np.ndarray) -> float:
    """Phase (deg) of x relative to ref at ref's dominant frequency; > 0 means x leads."""
    spec_ref = np.fft.rfft(ref - ref.mean())
    i = int(np.argmax(np.abs(spec_ref[1:]))) + 1
    spec_x = np.fft.rfft(x - x.mean())
    return float(np.rad2deg(np.angle(spec_x[i] / spec_ref[i])))


class WalkGait(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b, cls.s, cls.mid = walk_track()

    def test_pelvis_surges_fore_aft(self):
        """With the measured surge on, the pelvis speed is not constant: its fore-aft acceleration is
        about 0.1-0.2 g (mocap 0.16, Erciyes waist 0.15 g). The surge is off by default (WALK_SURGE,
        pending the viz rig), so this turns it on for one walk."""
        from thundersim import motions

        saved = motions.WALK_SURGE
        try:
            motions.WALK_SURGE = motions.WALK_SURGE_MOCAP
            _, s, mid = walk_track()
        finally:
            motions.WALK_SURGE = saved
        heading = s.heading[mid]
        fwd = np.stack([-np.sin(heading), -np.cos(heading)], axis=1)
        acc = np.gradient(np.gradient(s.pos[mid][:, [0, 2]], 1 / FS, axis=0), 1 / FS, axis=0)
        fore_aft = np.sum(acc * fwd, axis=1)
        smooth = np.convolve(fore_aft, np.ones(25) / 25, mode="same")[50:-50] / 9.80665
        self.assertGreater(np.std(smooth), 0.08)
        self.assertLess(np.std(smooth), 0.25)

    def test_thorax_turns_behind_the_pelvis(self):
        """Thorax world yaw (pelvis heading + spinal twist) about 0.5-0.8x the pelvis rotation
        and 70-130 deg behind it (mocap 0.65x, ~95-130 deg), not anti-phase and cancelled."""
        s, mid = self.s, self.mid
        pelvis = np.unwrap(s.heading[mid])
        pelvis = pelvis - np.polyval(np.polyfit(np.arange(len(pelvis)), pelvis, 1), np.arange(len(pelvis)))
        thorax = pelvis + s.twist[mid].sum(axis=1)
        ratio = np.std(thorax) / np.std(pelvis)
        self.assertGreater(ratio, 0.5)
        self.assertLess(ratio, 0.8)
        lag = -stride_phase(thorax, pelvis)
        self.assertGreater(lag, 70.0)
        self.assertLess(lag, 130.0)

    def test_thorax_rolls_less_than_the_pelvis(self):
        """The spine bends sideways against the pelvic obliquity: thorax roll about half the pelvis's."""
        s, mid = self.s, self.mid
        pelvis = s.roll[mid] - s.roll[mid].mean()
        thorax = pelvis + s.lat[mid].sum(axis=1)
        ratio = np.std(thorax) / np.std(pelvis)
        self.assertGreater(ratio, 0.35)
        self.assertLess(ratio, 0.7)

    def test_steps_vary_without_touching_the_main_random_stream(self):
        """Step-to-step variation comes from a keyed generator: the main stream ends where it would
        without it, so every later random draw of a run is unchanged."""
        from thundersim import motions

        saved = motions.WALK_STEP_VARIATION
        try:
            motions.WALK_STEP_VARIATION = 0.0
            b0, s0, _ = walk_track(seed=11)
            motions.WALK_STEP_VARIATION = saved
            b1, s1, _ = walk_track(seed=11)
        finally:
            motions.WALK_STEP_VARIATION = saved
        self.assertEqual(b0.rng.random(), b1.rng.random())
        self.assertFalse(np.allclose(s0.pos[:, 1], s1.pos[:, 1]))


if __name__ == "__main__":
    unittest.main()
