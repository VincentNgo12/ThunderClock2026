# File: test_posture.py
# Purpose: Posture-detection regression gates on fixed seeds (simulated data): trunk angle
#          accuracy, slumped-curve detection, and twist being reported as not measurable.
# Contributors: Unassigned

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thundersim.posture import evaluate_run  # noqa: E402


def variant(rows, method, cal):
    return next(r for r in rows if r["method"] == method and r["functional_cal"] == cal)


class PostureTests(unittest.TestCase):
    def test_trunk_angle_accuracy_on_target_firmware(self):
        row = variant(evaluate_run("P-BENCH", 2, "target"), "fusion", True)
        self.assertLess(row["trunk_mae"], 3.0)
        self.assertGreater(row["zone_agreement"], 0.9)

    def test_slumped_spinal_curve_is_detected(self):
        row = variant(evaluate_run("P-SLOUCH", 1, "target"), "fusion", True)
        curve = row["episodes"]["flexed_curve"]
        self.assertEqual(curve["hit"], curve["truth"])
        self.assertGreaterEqual(curve["truth"], 1)

    def test_twist_needs_the_gyro(self):
        rows = evaluate_run("P-TWIST", 1, "target")
        self.assertIsNone(variant(rows, "accel", True)["episodes"]["twist_hold"]["hit"])
        self.assertIsNotNone(variant(rows, "fusion", True)["twist_mae"])


if __name__ == "__main__":
    unittest.main()
