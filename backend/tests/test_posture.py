# File: test_posture.py
# Purpose: Verify neutral-relative angles, sustained warnings, and unavailable input.
# Contributors: Unassigned

from datetime import UTC, datetime
import math
import unittest

from app.models import Imu, Sample
from app.posture import Posture, multiply


IDENTITY = (1.0, 0.0, 0.0, 0.0)
RECEIPT = datetime(2026, 1, 1, tzinfo=UTC)


def rotation(axis, degrees):
    half_angle = math.radians(degrees) / 2
    return (math.cos(half_angle), *(value * math.sin(half_angle) for value in axis))


def frame(pelvis=IDENTITY, lumbar=IDENTITY, thoracic=IDENTITY, upper=IDENTITY):
    return Sample("quaternion", 0, tuple(
        Imu(sensor_id, quaternion=q, valid=q is not None)
        for sensor_id, q in enumerate((pelvis, lumbar, thoracic, upper), 1)
    ))


class PostureTests(unittest.TestCase):
    def setUp(self):
        self.posture = Posture(warning_deg=25, warning_seconds=0.5)
        self.forward = rotation((1, 0, 0), -30)
        self.bent = frame(IDENTITY, self.forward, self.forward, self.forward)

    def capture(self):
        self.posture.capture(frame(), 0.0, RECEIPT, 0.0)

    def test_upright_and_common_motion(self):
        self.posture.update(frame(), 0.0)
        self.assertIsNone(self.posture.angles)
        self.capture()
        self.assertTrue(all(angle == 0 for angle in self.posture.angles.values()))

        tilt = rotation((1, 0, 0), -45)
        self.posture.update(frame(tilt, tilt, tilt, tilt), 0.1)
        for angle in self.posture.angles.values():
            self.assertAlmostEqual(angle, 0)

    def test_signed_relative_rotations(self):
        self.capture()
        left = rotation((0, 0, 1), 15)
        upper = multiply(self.forward, left)
        self.posture.update(frame(IDENTITY, self.forward, upper, upper), 0.1)
        angles = self.posture.angles
        self.assertAlmostEqual(angles["lumbar_flexion_deg"], 30)
        self.assertAlmostEqual(angles["thoracic_flexion_deg"], 0)
        self.assertAlmostEqual(angles["thoracic_lateral_bend_deg"], 15)
        self.assertAlmostEqual(angles["upper_thoracic_flexion_deg"], 0)

        # Capture a non-identity reference, then measure extension from it.
        self.posture.capture(self.bent, 0.1, RECEIPT, 0.1)
        self.posture.update(frame(), 0.2)
        self.assertAlmostEqual(self.posture.angles["lumbar_flexion_deg"], -30)

    def test_sustained_bend_warns_and_clears(self):
        self.capture()
        for receipt in (0.25, 0.5):
            self.posture.update(self.bent, receipt)
            self.assertEqual(self.posture.alerts, [])
        self.posture.update(self.bent, 0.75)
        self.assertEqual(len(self.posture.alerts), 1)
        self.assertEqual(self.posture.alerts[0]["region"], "lumbar")
        self.assertEqual(self.posture.alerts[0]["sustained_seconds"], 0.5)

        self.posture.update(self.bent, 1.5)
        self.assertEqual(self.posture.alerts, [])  # Silence breaks the hold.
        self.posture.update(frame(), 1.75)
        self.posture.update(self.bent, 2.0)
        self.assertEqual(self.posture.alerts, [])

    def test_missing_sensor_clears_posture_and_hold(self):
        self.capture()
        self.posture.update(self.bent, 0.25)
        self.posture.update(frame(upper=None), 0.5)
        self.assertIsNone(self.posture.angles)
        self.assertEqual(self.posture.alerts, [])
        self.posture.update(self.bent, 0.75)
        self.assertEqual(self.posture.alerts, [])
        self.assertEqual(self.posture.calibration["status"], "captured")

    def test_failed_calibration_preserves_reference(self):
        with self.assertRaises(ValueError):
            self.posture.capture(None, None, None, 0.0)
        self.assertEqual(self.posture.calibration["status"], "uncalibrated")
        self.capture()
        reference = self.posture.neutral
        for sample, now in ((frame(upper=None), 0.0), (Sample("raw", 0, ()), 0.0), (self.bent, 1.0)):
            with self.subTest(sample=sample, now=now), self.assertRaises(ValueError):
                self.posture.capture(sample, 0.0, RECEIPT, now)
            self.assertIs(self.posture.neutral, reference)
