# File: test_backend_hook.py
# Purpose: DetectorService (backend hook) with fake decoded raw samples: no BLE, no FastAPI.
#          Checks intake -> MainModel -> events, drop counting, scoped dismissal and status.
# Contributors: Unassigned
import unittest
from types import SimpleNamespace

import numpy as np

import synth
from hub.backend_hook import DetectorService


def to_samples(t, acc, gyro):
    """Segment-frame arrays -> backend Samples in board axes (inverse of the default -y,z,-x map)."""
    out = []
    for k in range(len(t)):
        imus = []
        for i in range(4):
            r, u, b = acc[k, i]
            gr, gu, gb = gyro[k, i]
            # board (x, y, z) with right=-y, up=z, back=-x  ->  x=-back, y=-right, z=up
            imus.append(SimpleNamespace(sensor_id=i + 1, accel=(-b, -r, u), gyro=(-gb, -gr, gu), valid=True))
        out.append(SimpleNamespace(sample_time_ms=int(round(t[k] * 1000)) + 5000, imus=imus))
    return out


class TestDetectorService(unittest.TestCase):
    def test_fake_ble_collapse_gives_events(self):
        t, acc, gyro, _ = synth.stream(synth.topple_script(pre_s=10, after_s=80), noise=0.002)
        service = DetectorService(start_thread=False)
        events = []
        for k, sample in enumerate(to_samples(t, acc, gyro)):
            service.on_raw_sample(sample, received_monotonic=float(k) / 100)
            if k % 10 == 9:                       # detector thread every 100 ms
                events += service.step()
        kinds = [e.kind for e in events]
        self.assertIn("fall_confirmed", kinds)
        self.assertIn("man_down_critical", kinds)
        snap = service.snapshot(now=len(t) / 100)
        self.assertEqual(snap["detector"]["status"], "ok")
        self.assertEqual(snap["detector"]["intake_dropped"], 0)
        self.assertTrue(all(a["id"] < b["id"] for a, b in zip(snap["detection_events"], snap["detection_events"][1:])))

    def test_drops_counted_and_stale(self):
        service = DetectorService(start_thread=False)
        t, acc, gyro, _ = synth.stream(synth.Script().hold(12))
        for sample in to_samples(t, acc, gyro):
            service.on_raw_sample(sample, received_monotonic=0.0)
        self.assertEqual(service.dropped, len(t) - 1000)
        service.step()
        self.assertEqual(service.snapshot(now=10.0)["detector"]["status"], "stale")

    def test_dismiss_is_scoped(self):
        t, acc, gyro, _ = synth.stream(synth.topple_script(pre_s=10, after_s=80))
        service = DetectorService(start_thread=False)
        kinds = []
        for k, sample in enumerate(to_samples(t, acc, gyro)):
            service.on_raw_sample(sample, received_monotonic=float(k) / 100)
            if k % 10 == 9:
                new = service.step()
                kinds += [e.kind for e in new]
                if any(e.kind == "fall_confirmed" for e in new):
                    service.dismiss()
        self.assertIn("man_down_critical", kinds)    # a dismissal never covers man-down


if __name__ == "__main__":
    unittest.main()
