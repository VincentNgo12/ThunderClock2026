# File: test_demo.py
# Purpose: Spine-demo plumbing: DetectorService.reset, RawFrames restart while calibrating, and the
#          Demo state machine (events -> fall_state, dismissal scope). No FastAPI, no simulator.
# Contributors: Unassigned
import unittest
from types import SimpleNamespace

import numpy as np

import synth
from hub.backend_hook import DetectorService
from hub.demo_app import Demo
from hub.intake import RawFrames
from test_backend_hook import to_samples


def sample(ms, up=9.8):
    imus = [SimpleNamespace(sensor_id=i + 1, accel=(0.0, up, 0.0), gyro=(0.0, 0.0, 0.0), valid=True) for i in range(4)]
    return SimpleNamespace(sample_time_ms=ms, imus=imus)


class TestIntakeRestart(unittest.TestCase):
    def test_restart_while_learning_forgets_the_old_boot(self):
        frames = RawFrames(axes=";".join(["x,y,z"] * 4))
        for k in range(100):                      # 1 s of quiet standing at boot A, around t = 60 s
            frames.push(sample(60000 + 10 * k))
        self.assertFalse(frames.calibrated)
        frames.push(sample(5000))                 # device restarted
        self.assertEqual(len(frames.quiet), 1)    # only the new boot's sample counts
        for k in range(1, 260):
            frames.push(sample(5000 + 10 * k))
        self.assertTrue(frames.calibrated)


class TestServiceReset(unittest.TestCase):
    def test_reset_clears_state_and_marks_old_events(self):
        t, acc, gyro, _ = synth.stream(synth.topple_script(pre_s=10, after_s=20), noise=0.002)
        service = DetectorService(start_thread=False)
        for k, s in enumerate(to_samples(t, acc, gyro)):
            service.on_raw_sample(s, float(k) / 100)
            if k % 10 == 9:
                service.step()
        self.assertTrue(service.events)
        last_id = service.next_event_id - 1
        service.on_raw_sample(sample(1), 0.0)
        service.reset()
        self.assertEqual(len(service.queue), 0)   # stale samples are dropped at once
        service.step()
        self.assertEqual(service.reset_count, 1)
        self.assertEqual(service.floor_id, last_id)
        self.assertFalse(service.frames.calibrated)


class FakeService:
    def __init__(self):
        self.events, self.next_event_id, self.reset_count, self.floor_id = [], 1, 1, 0
        self.dismissed = []

    def add(self, kind, episode=None):
        self.events.append({"id": self.next_event_id, "t": 1.0, "kind": kind, "confidence": 0.9,
                            "detail": {"episode_id": episode, "path": "impact", "rule": "model"}})
        self.next_event_id += 1

    def dismiss(self, episode=None):
        self.dismissed.append(episode)

    def reset(self):
        self.reset_count += 1

    def calibrate(self):
        pass


class TestDemoState(unittest.TestCase):
    def make(self):
        service = FakeService()
        demo = Demo(service)
        demo.begin_playback(SimpleNamespace(), 0.0)
        service.reset_count = demo.expect_reset   # the hub applied the reset
        return service, demo

    def test_states_follow_events(self):
        service, demo = self.make()
        self.assertEqual(demo.fall_state()["state"], "normal")
        service.add("suspected_fall")
        self.assertEqual(demo.fall_state()["state"], "suspected")
        service.add("fall_confirmed", episode=7)
        state = demo.fall_state()
        self.assertEqual((state["state"], state["episode_id"], state["dismissable"]), ("fall_confirmed", 7, True))
        service.add("man_down_warning", episode=7)
        state = demo.fall_state()
        self.assertEqual((state["state"], state["dismissable"]), ("man_down_warning", False))

    def test_dismiss_covers_fall_but_not_man_down(self):
        service, demo = self.make()
        self.assertIsNone(demo.dismiss())
        service.add("fall_confirmed", episode=3)
        demo.fall_state()
        self.assertEqual(demo.dismiss(), 3)
        self.assertEqual(service.dismissed, [3])
        self.assertEqual(demo.fall_state()["state"], "dismissed")
        service.add("man_down_critical", episode=3)
        self.assertEqual(demo.fall_state()["state"], "man_down_critical")

    def test_events_before_the_reset_are_ignored(self):
        service, demo = self.make()
        service.add("fall_confirmed", episode=1)
        service.floor_id = 1                       # hub reset after that event
        self.assertEqual(demo.fall_state()["state"], "normal")
        service.reset_count = demo.expect_reset - 1   # reset not applied yet: nothing is consumed
        service.add("suspected_fall")
        self.assertEqual(demo.fall_state()["state"], "normal")


if __name__ == "__main__":
    unittest.main()
