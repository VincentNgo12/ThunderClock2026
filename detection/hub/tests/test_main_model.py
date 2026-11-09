# File: test_main_model.py
# Purpose: Tests for the arbiter, MainModel time handling (event t <= batch end, gaps, restart,
#          rate-latch recovery), T1 + T3 = one alert, RawFrames intake, the falldet profile choice
#          and the ThunderSim adapter.
# Contributors: Unassigned
import os
import unittest
from types import SimpleNamespace

import numpy as np

import synth
from hub.arbiter import Arbiter
from hub.config import config_hash, load_config
from hub.falldet_path import choose_profile
from hub.intake import RawFrames
from hub.main_model import MainModel
from hub.posture_track import PostureTrack

CFG = load_config()


class FakeTrack:
    def __init__(self, upright_s=0.0, gait=False):
        self.u, self.g = upright_s, gait

    def upright_for(self, t):
        return self.u

    def gait(self, t):
        return self.g


class TestArbiter(unittest.TestCase):
    def test_t1_plus_t3_is_one_alert(self):
        a = Arbiter(CFG["arbiter"])
        out = a.suspected(10.0, "topple", 10.0) + a.suspected(10.3, "impact", 10.2)
        self.assertEqual([p["kind"] for p in out], ["suspected_fall"])
        out = a.confirmed(18.0, "topple", "topple_still_down", 10.0, 0.8)
        out += a.confirmed(18.2, "impact", "model", 10.2, 0.9)
        self.assertEqual([p["kind"] for p in out], ["fall_confirmed"])
        self.assertEqual(a.episode["paths"], ["topple", "impact"])
        self.assertEqual(a.recovered(40.0, "topple", "moving_or_up", 10.0, 0.6), [])

    def test_man_down_once_per_episode_and_not_dismissable(self):
        a = Arbiter(CFG["arbiter"])
        a.confirmed(18.0, "impact", "model", 10.0, 0.9)
        self.assertTrue(a.dismiss(30.0))
        out = a.man_down(41.0, "man_down_warning", "down_mostly_still", 10.0)
        out += a.man_down(42.0, "man_down_warning", "down_mostly_still", 10.0)
        out += a.man_down(71.0, "man_down_critical", "down_mostly_still", 10.0)
        self.assertEqual([p["kind"] for p in out], ["man_down_warning", "man_down_critical"])
        self.assertEqual({p["episode_id"] for p in out}, {1})

    def test_new_trigger_after_dismissal_opens_new_episode(self):
        a = Arbiter(CFG["arbiter"])
        a.confirmed(18.0, "impact", "model", 10.0, 0.9)
        a.dismiss(30.0)
        a.suspected(100.0, "topple", 100.0)
        out = a.confirmed(108.0, "topple", "topple_still_down", 100.0, 0.8)
        self.assertEqual(out[0]["kind"], "fall_confirmed")
        self.assertEqual(out[0]["episode_id"], 2)

    def test_dismissed_episode_closes_after_quiet_period(self):
        a = Arbiter(CFG["arbiter"])
        a.confirmed(18.0, "impact", "model", 10.0, 0.9)
        a.dismiss(30.0)
        a.on_sample(149.0, FakeTrack())
        self.assertIsNotNone(a.episode)
        a.on_sample(151.0, FakeTrack())
        self.assertIsNone(a.episode)

    def test_cleared_needs_upright_and_gait(self):
        a = Arbiter(CFG["arbiter"])
        a.confirmed(18.0, "impact", "model", 10.0, 0.9)
        self.assertEqual(a.on_sample(40.0, FakeTrack(upright_s=20.0, gait=False)), [])   # suspended in a harness
        self.assertEqual(a.on_sample(15.0, FakeTrack(upright_s=5.0, gait=True)), [])     # < 8 s after the event
        out = a.on_sample(40.0, FakeTrack(upright_s=6.0, gait=True))
        self.assertEqual(out[0]["kind"], "cleared")


class TestMainModel(unittest.TestCase):
    def collapse_stream(self, after_s=80.0, fs=100.0):
        return synth.stream(synth.topple_script(after_s=after_s), fs=fs, noise=0.002)

    def test_events_never_after_batch_end_and_full_escalation(self):
        t, acc, gyro, _ = self.collapse_stream()
        model = MainModel()
        rng = np.random.default_rng(1)
        events, i = [], 0
        while i < len(t):
            j = min(len(t), i + int(rng.integers(5, 80)))
            for e in model.update_arrays(t[i:j], acc[i:j], gyro[i:j]):
                self.assertLessEqual(e.t, t[j - 1] + 1e-9)
                events.append(e)
            i = j
        kinds = [e.kind for e in events]
        self.assertEqual(kinds, ["suspected_fall", "fall_confirmed", "man_down_warning", "man_down_critical"])
        self.assertEqual(events[1].detail["path"], "topple")
        self.assertEqual(events[1].detail["data"]["model_profile"], "current")
        self.assertEqual(len({e.detail["episode_id"] for e in events[1:]}), 1)

    def test_t1_and_t3_give_one_alert(self):
        t, acc, gyro, _ = self.collapse_stream()
        model = MainModel()
        real_push = model.falldet.push
        sent = {"done": False}

        def push(tt, *args):
            ev, diags = real_push(tt, *args)
            if not sent["done"] and tt[-1] >= 18.6:
                sent["done"] = True
                ev = ev + [{"type": "stage1_trigger", "t_s": 10.5, "svm_g": 2.2, "rot_rad_s": 4.0},
                           {"type": "stage2_result", "state": "fall_confirmed", "confidence": 0.9, "rule": "model",
                            "t_s": 10.5, "decided_t_s": float(tt[-1]), "gap_in_window": False, "features": {}}]
            return ev, diags

        model.falldet.push = push
        events = [e for e, _ in synth.feed(model, t, acc, gyro)]
        alerts = [e for e in events if e.kind == "fall_confirmed"]
        self.assertEqual(len(alerts), 1)
        self.assertEqual(sum(e.kind == "suspected_fall" for e in events), 1)

    def test_gap_at_event(self):
        """Gaps centred on the end of the descent (10.6 s). A 1.5 s gap that hides the whole descent is
        a known miss for T3 (no rotation is observed; the T5 gap trigger is a stretch goal)."""
        t, acc, gyro, _ = self.collapse_stream()
        for gap, expect in ((0.3, True), (0.7, True), (1.5, False)):
            keep = ~((t >= 10.6 - gap / 2) & (t < 10.6 + gap / 2))
            model = MainModel()
            kinds = [e.kind for e, _ in synth.feed(model, t[keep], acc[keep], gyro[keep])]
            self.assertEqual("fall_confirmed" in kinds, expect, f"gap {gap}")
            self.assertEqual("man_down_critical" in kinds, expect, f"gap {gap}")

    def test_restart_keeps_arming(self):
        t, acc, gyro, _ = self.collapse_stream()
        boot = 13.0
        keep = ~((t >= boot) & (t < boot + 0.5))
        t2 = np.where(t >= boot, t - boot, t)[keep]
        model = MainModel()
        events = synth.feed(model, t2, acc[keep], gyro[keep])
        for e, end in events:
            self.assertLessEqual(e.t, end + 1e-9)
        kinds = [e.kind for e, _ in events]
        self.assertEqual(model.counters["restarts"], 1)
        self.assertIn("man_down_critical", kinds)
        self.assertTrue(any(d["code"] == "restart" for d in model.diagnostics))

    def test_backwards_jitter_is_dropped_not_a_restart(self):
        t, acc, gyro, _ = synth.stream(synth.Script().hold(5))
        model = MainModel()
        model.update_arrays(t[:300], acc[:300], gyro[:300])
        model.update_arrays(t[250:400], acc[250:400], gyro[250:400])   # 0.5 s overlap
        self.assertEqual(model.counters["restarts"], 0)
        self.assertEqual(model.counters["duplicates"], 50)

    def test_slow_start_latch_recovers(self):
        t, acc, gyro, _ = self.collapse_stream()
        keep = ~((t < 2.0) & (np.arange(len(t)) % 5 != 0))    # 20 Hz for 2 s, then 100 Hz
        model = MainModel()
        synth.feed(model, t[keep][:300], acc[keep][:300], gyro[keep][:300])
        self.assertTrue(model.falldet.latched)
        self.assertEqual(model.status()["status"], "insufficient_rate")
        synth.feed(model, t[keep][300:], acc[keep][300:], gyro[keep][300:])
        self.assertFalse(model.falldet.latched)
        self.assertEqual(model.counters["rate_recoveries"], 1)
        self.assertEqual(model.status()["status"], "ok")

    def test_20hz_topple_still_triggers(self):
        t, acc, gyro, _ = synth.stream(synth.topple_script(after_s=80), fs=20.0)
        model = MainModel()
        kinds = [e.kind for e, _ in synth.feed(model, t, acc, gyro)]
        self.assertEqual(model.status()["status"], "insufficient_rate")
        self.assertIn("fall_confirmed", kinds)

    def test_dead_slot_nan(self):
        t, acc, gyro, _ = self.collapse_stream()
        acc = acc.copy()
        acc[:, 2] = np.nan
        mask = np.ones((len(t), 4), bool)
        mask[:, 2] = False
        kinds = [e.kind for e, _ in synth.feed(MainModel(), t, acc, gyro, slot_mask=mask)]
        self.assertIn("fall_confirmed", kinds)

    def test_slow_lie_down_and_table_are_silent(self):
        for script, breathing in ((synth.topple_script(fall_s=3.0, after_s=100), True),
                                  (synth.Script((90,) * 4).hold(120), False)):
            t, acc, gyro, _ = synth.stream(script, breathing=breathing)
            kinds = [e.kind for e, _ in synth.feed(MainModel(), t, acc, gyro)]
            self.assertEqual([k for k in kinds if k != "near_fall"], [])

    def test_config_hash_stable(self):
        self.assertEqual(config_hash(load_config()), config_hash(load_config()))
        self.assertNotEqual(config_hash(load_config()), config_hash(load_config(overrides={"topple": {"rot_rad_s": 2.9}})))


class TestProfileChoice(unittest.TestCase):
    def test_bandwidth_rule(self):
        f = CFG["falldet"]
        self.assertEqual(choose_profile(f, 10, 8)[0], "current")        # fw_main
        self.assertEqual(choose_profile(f, 44, 8)[0], "target")         # target_8g
        self.assertEqual(choose_profile(f, 44, 2)[0], "current")        # current_100hz-like range
        self.assertEqual(choose_profile(f, 10, 8, explicit="target")[0], "target")
        os.environ["FALLDET_PROFILE"] = "target"
        try:
            self.assertEqual(choose_profile(f, 10, 8)[0], "target")
        finally:
            del os.environ["FALLDET_PROFILE"]
        self.assertEqual(choose_profile(f)[0], "current")               # backend default: firmware 10 Hz


class TestIntake(unittest.TestCase):
    def sample(self, ms, accel=(0.0, 0.0, 9.80665), gyro=(0.0, 0.0, 0.0), dead=()):
        imus = [SimpleNamespace(sensor_id=i + 1, accel=(0, 0, 0) if i in dead else accel,
                                gyro=(0, 0, 0) if i in dead else gyro, valid=True) for i in range(4)]
        return SimpleNamespace(sample_time_ms=ms, imus=imus)

    def test_unwrap_duplicates_dead_slot_and_alignment(self):
        frames = RawFrames()
        start = 2 ** 32 - 3000
        for k in range(400):
            frames.push(self.sample((start + 10 * k) & 0xFFFFFFFF, dead=(2,)))
            if k == 10:
                frames.push(self.sample((start + 10 * k) & 0xFFFFFFFF))     # duplicate
        self.assertTrue(frames.calibrated)
        t, acc, gyro, mask = frames.pop()
        self.assertTrue(np.all(np.diff(t) > 0))
        self.assertGreater(t[-1], 2 ** 32 / 1000.0)                          # unwrapped
        self.assertEqual(frames.duplicates, 1)
        self.assertFalse(mask[:, 2].any())
        self.assertTrue(np.isnan(acc[:, 2]).all())
        np.testing.assert_allclose(acc[-1, 0], [0, 9.80665, 0], atol=1e-6)   # board +Z up -> segment +Y

    def test_restart_passes_through(self):
        frames = RawFrames(align=False)
        for ms in (50000, 50010, 120, 130):
            frames.push(self.sample(ms))
        t, *_ = frames.pop()
        self.assertEqual(list(t), [50.0, 50.01, 0.12, 0.13])
        model = MainModel()
        model.update_arrays(t[:2], np.tile([0, 9.8, 0], (2, 4, 1)), np.zeros((2, 4, 3)))
        model.update_arrays(t[2:], np.tile([0, 9.8, 0], (2, 4, 1)), np.zeros((2, 4, 3)))
        self.assertEqual(model.counters["restarts"], 1)


class TestThunderSimAdapter(unittest.TestCase):
    def test_hub_loads_and_runs_a_collapse(self):
        try:
            from thundersim.model_api import FallModel, load_model
            from thundersim.runner import StreamOptions, run_scenario
        except Exception as error:  # simulator not importable in this environment
            self.skipTest(f"thundersim unavailable: {error}")
        model = load_model("hub.sim_models:HubFull")
        self.assertIsInstance(model, FallModel)
        result = run_scenario("SYNC", 0, model, StreamOptions(profile="target_8g", fast=True), obs_s=70)
        kinds = [e["kind"] for e in result.events]
        self.assertIn("fall_confirmed", kinds)
        self.assertEqual(model.status()["model_profile"], "target")    # 44 Hz DLPF -> target


if __name__ == "__main__":
    unittest.main()
