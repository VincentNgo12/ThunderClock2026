# File: test_device_transport.py
# Purpose: MPU-6050 emulation, raw protobuf round trip through the backend decoder, and the
#          canonicalizer's neutral alignment and de-duplication.
# Contributors: Unassigned

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thundersim.canonical import Canonicalizer, neutral_alignment  # noqa: E402
from thundersim.device import ACCEL_LSB_PER_G, INTERNAL_HZ, PROFILES, DeviceErrors, ImuStrip  # noqa: E402
from thundersim.model_api import GRAVITY  # noqa: E402
from thundersim.transport import BleLink, RawCodec  # noqa: E402

QUIET = DeviceErrors(accel_bias_g=(0, 0, 0), accel_scale=0.0, gyro_bias_dps=0.0, gyro_tempco_dps_per_c=0.0,
                     read_skew_s=0.0008, timing_jitter_s=0.0)


def run_strip(profile, acc_value, seconds=4.0):
    n = int(seconds * INTERNAL_HZ)
    t = np.arange(n) / INTERNAL_HZ
    acc = np.tile(acc_value, (n, 4, 1)).astype(float)
    strip = ImuStrip(PROFILES[profile], 4, QUIET, seed=0)
    return strip.process(t, acc, np.zeros((n, 4, 3)), 20.0)


class DeviceTests(unittest.TestCase):
    def test_current_firmware_samples_every_two_seconds(self):
        out = run_strip("current", [0, GRAVITY, 0], seconds=10.0)
        self.assertEqual(len(out.t_true), 5)
        self.assertTrue(np.allclose(np.diff(out.t_true), 2.0, atol=1e-3))

    def test_two_g_range_clips_a_four_g_impact(self):
        out = run_strip("current_100hz", [0, 4 * GRAVITY, 0])
        self.assertTrue(out.sat[-10:].all())
        self.assertAlmostEqual(out.acc[-1, 0, 1] / GRAVITY, 2.0, delta=0.01)
        target = run_strip("target", [0, 4 * GRAVITY, 0])
        self.assertFalse(target.sat[-10:].any())

    def test_values_are_quantized_to_the_range_lsb(self):
        out = run_strip("target", [0, GRAVITY, 0])
        counts = out.acc * ACCEL_LSB_PER_G[16] / GRAVITY
        self.assertTrue(np.allclose(counts, np.round(counts), atol=1e-6))


class TransportTests(unittest.TestCase):
    def test_raw_packet_round_trips_through_backend_decoder(self):
        codec = RawCodec()
        acc = np.random.default_rng(0).normal(0, 10, (4, 3))
        gyro = np.random.default_rng(1).normal(0, 2, (4, 3))
        payload = codec.encode(2**32 + 1234, acc, gyro)  # millis() wraps on the wire
        self.assertLessEqual(len(payload), codec.max_bytes)
        time_ms, acc2, gyro2 = codec.decode(payload)
        self.assertEqual(time_ms, 1234)
        self.assertTrue(np.allclose(acc2, acc.astype(np.float32)))
        self.assertTrue(np.allclose(gyro2, gyro.astype(np.float32)))

    def test_constrained_link_drops_100hz_raw_stream(self):
        link = BleLink("constrained", seed=0)
        delivered = sum(link.send(i * 0.01, 148) is not None for i in range(2000))
        self.assertLess(delivered / 2000, 0.6)  # 14.8 kB/s offered into ~6 kB/s


class CanonicalTests(unittest.TestCase):
    def test_neutral_alignment_maps_mean_to_up(self):
        tilted = np.array([0.3, 0.9, -0.2]) * GRAVITY
        rot = neutral_alignment(tilted)
        self.assertTrue(np.allclose(rot.apply(tilted) / np.linalg.norm(tilted), [0, 1, 0], atol=1e-9))

    def test_duplicate_times_are_dropped_and_batches_are_aligned(self):
        canon = Canonicalizer(100.0, 16, 2000, calibration_s=1.0, batch_s=0.5)
        acc = np.tile([0.2, 9.7, 0.5], (4, 1))
        for k in range(300):
            canon.push(1000 + 10 * k, acc, np.zeros((4, 3)))
            canon.push(1000 + 10 * k, acc, np.zeros((4, 3)))  # repeat must be ignored
        batches = canon.pop_batches(final=True)
        frames = sum(len(b) for b in batches)
        self.assertEqual(frames, 300)
        self.assertTrue(np.allclose(batches[0].acc[0, :, 1], np.linalg.norm([0.2, 9.7, 0.5]), atol=1e-3))


if __name__ == "__main__":
    unittest.main()
