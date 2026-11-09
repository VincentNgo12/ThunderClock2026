"""Motion validation (thundersim/motion_validation.py) and the step-1 motion tuning it backs."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thundersim import motion_validation as mv  # noqa: E402
from thundersim.motions import Builder  # noqa: E402
from thundersim.scenarios import random_worker  # noqa: E402
from thundersim.site import random_site  # noqa: E402


def builder(seed: int = 3) -> Builder:
    rng = np.random.default_rng(seed)
    site = random_site(np.random.default_rng(seed + 7_919))
    b = Builder(random_worker(rng, site.clothing), site, rng)
    b.hold(5.0, "calibration_stand")
    return b


class Geometry(unittest.TestCase):
    def test_inclination_of_a_pure_pelvis_pitch(self):
        angles = np.zeros((5, 12))
        angles[:, 0] = 1.2          # any heading
        angles[:, 1] = np.deg2rad(30.0)
        incl = mv.segment_inclinations(angles)
        np.testing.assert_allclose(incl, 30.0, atol=1e-6)

    def test_inclination_adds_joint_flexion(self):
        angles = np.zeros((1, 12))
        angles[0, 1] = np.deg2rad(20.0)
        angles[0, 3:6] = np.deg2rad([10.0, 5.0, 5.0])
        np.testing.assert_allclose(mv.segment_inclinations(angles)[0], [20.0, 30.0, 35.0, 40.0], atol=1e-6)


class SimulatorWindows(unittest.TestCase):
    def test_carry_run_gives_every_window_kind_and_sane_gait(self):
        windows = mv.sim_windows("A-CARRY", 0)
        kinds = {w.activity for w in windows}
        self.assertTrue({"walk", "carry_walk", "lift", "idle"} <= kinds)
        walk = mv.gait_metrics(next(w for w in windows if w.activity == "walk"))
        self.assertTrue(1.5 < walk["step_hz"] < 2.2)
        self.assertTrue(0.5 < walk["imu_step_peak_10hz_g"] < 2.0)

    def test_ladder_rolls_to_alternate_sides(self):
        ladder = next(w for w in mv.sim_windows("A-LADDER", 0) if w.activity == "ladder")
        self.assertAlmostEqual(mv.ladder_metrics(ladder)["roll_to_rung_freq"], 0.5, delta=0.1)

    def test_kneeling_trunk_moves(self):
        kneel = next(w for w in mv.sim_windows("A-KNEEL", 0) if w.activity == "kneel")
        m = mv.kneel_metrics(kneel)
        self.assertGreater(m["upper_incl_p90_deg"] - m["upper_incl_p10_deg"], 15.0)
        self.assertLess(m["upper_incl_rate_p95_dps"], 180.0)


class LiftTiming(unittest.TestCase):
    def test_short_hold_and_exact_total(self):
        b = builder()
        start = b.n
        b.lift(0.2, 20.0, technique="semi_squat", total_s=4.4)
        self.assertEqual(b.n - start, 4400)          # a carry keeps its old length
        grasp = next(t for t, name, _ in b.events if name == "lift_grasp")
        off = next(t for t, name, _ in b.events if name == "lift_off")
        self.assertLessEqual(off - grasp, 0.41)      # LIFT_HOLD_S; the rest is spent upright


class VestRecordings(unittest.TestCase):
    def test_round_trip_of_a_synthetic_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = mv.write_example_session(Path(tmp) / "20261108_SYN_01", "A-CARRY", 0)
            self.assertEqual(mv.recording_sessions(Path(tmp)), [])     # synthetic sessions are skipped
            self.assertEqual(mv.recording_sessions(Path(tmp), include_synthetic=True), [folder])
            session = mv.load_vest_session(folder)
            neutral = session["t"] < 5.0
            np.testing.assert_allclose(np.mean(session["acc"][neutral], axis=0) / mv.GRAVITY, [0, 1, 0], atol=0.03)  # device scale error
            windows = mv.vest_windows(folder)
            kinds = {w.activity for w in windows}
            self.assertTrue({"walk", "lift", "idle"} <= kinds)
            # Trunk angles recovered from the fused quaternions match the simulated bends.
            sim_peaks = sorted(r["peak_upper_deg"] for w in mv.sim_windows("A-CARRY", 0) if w.activity == "lift"
                               for r in mv.window_metrics(w))
            vest_peaks = sorted(r["peak_upper_deg"] for w in windows if w.activity == "lift" for r in mv.window_metrics(w))
            self.assertEqual(len(vest_peaks), len(sim_peaks))
            np.testing.assert_allclose(vest_peaks, sim_peaks, atol=3.0)


if __name__ == "__main__":
    unittest.main()
