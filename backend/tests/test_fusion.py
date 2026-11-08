# File: test_fusion.py
# Purpose: Verify raw-axis mapping, gyro integration and timestamp discontinuities.
# Contributors: Unassigned

import math
import unittest

from app.fusion import GRAVITY_MS2, RawFusion, parse_axes
from app.models import Diagnostics, Imu, Sample


def raw_frame(time_ms, angle=0.0, rate=0.0, missing_id=None):
    # Build board-frame readings for a rotation about application +X.
    accel = (math.sin(angle) * GRAVITY_MS2, 0.0, math.cos(angle) * GRAVITY_MS2)
    gyro = (0.0, -rate, 0.04)  # Fixed board Z gyro bias.
    imus = tuple(
        Imu(sensor_id, accel=(0, 0, 0), gyro=(0, 0, 0))
        if sensor_id == missing_id else Imu(sensor_id, accel=accel, gyro=gyro)
        for sensor_id in range(1, 5)
    )
    return Sample("raw", time_ms & 0xFFFFFFFF, imus)


class FusionTests(unittest.TestCase):
    def test_initializes_still_then_tracks_known_bend(self):
        fusion = RawFusion(Diagnostics())
        fusion.update(raw_frame(0), 0, None)
        self.assertFalse(fusion.latest.fully_usable)
        for time_ms in range(10, 501, 10):
            fusion.update(raw_frame(time_ms), time_ms / 1000, None)
        self.assertTrue(fusion.latest.fully_usable)
        self.assertEqual(fusion.latest.imus[0].quaternion, (1, 0, 0, 0))

        rate = -math.pi / 3
        for time_ms in range(510, 1001, 10):
            angle = rate * (time_ms - 500) / 1000
            fusion.update(raw_frame(time_ms, angle, rate), time_ms / 1000, None)
        for imu in fusion.latest.imus:
            w, x, y, z = imu.quaternion
            self.assertAlmostEqual(math.hypot(w, x, y, z), 1)
            self.assertAlmostEqual(math.degrees(2 * math.atan2(x, w)), -30, delta=0.5)
            self.assertAlmostEqual(y, 0)
            self.assertAlmostEqual(z, 0)

    def test_gravity_feedback_corrects_tilt_with_no_gyro_motion(self):
        fusion = RawFusion(Diagnostics())
        for time_ms in range(0, 501, 10):
            fusion.update(raw_frame(time_ms), time_ms / 1000, None)
        for time_ms in range(510, 2501, 10):
            fusion.update(raw_frame(time_ms, angle=-math.pi / 6), time_ms / 1000, None)
        w, x, _, _ = fusion.latest.imus[0].quaternion
        self.assertAlmostEqual(math.degrees(2 * math.atan2(x, w)), -30, delta=1)

    def test_wrap_duplicate_and_missing_slot(self):
        fusion = RawFusion(Diagnostics())
        start = 2**32 - 200
        for elapsed in range(0, 501, 50):
            fusion.update(raw_frame(start + elapsed), elapsed / 1000, None)
        self.assertTrue(fusion.latest.fully_usable)
        self.assertFalse(fusion.update(raw_frame(start + 500), 0.55, None))
        self.assertEqual(fusion.received_monotonic, 0.5)
        self.assertEqual(fusion.stats.resets, 0)

        fusion.update(raw_frame(start + 550, missing_id=2), 0.55, None)
        self.assertFalse(fusion.latest.imus[1].valid)
        self.assertIsNone(fusion.latest.imus[1].quaternion)
        self.assertTrue(fusion.latest.imus[0].valid)
        self.assertEqual(fusion.stats.resets, 1)

        fusion.update(raw_frame(start + 1000), 1.0, None)
        self.assertFalse(any(imu.valid for imu in fusion.latest.imus))
        self.assertEqual(fusion.stats.resets, 2)

    def test_axis_maps_must_be_right_handed(self):
        self.assertEqual(parse_axes("x,y,z;-y,z,-x;x,-z,y;x,y,z")[1], (-2, 3, -1))
        with self.assertRaisesRegex(ValueError, "right-handed"):
            parse_axes("-x,y,z;x,y,z;x,y,z;x,y,z")
