# File: test_triggers.py
# Purpose: Unit tests for PostureTrack, the T3 topple trigger, the C3 collapse confirmer and the
#          T4 man-down timer on synthetic signals (every branch named in DETECTION_ARCHITECTURE.md).
# Contributors: Unassigned
import unittest

import numpy as np

import synth  # noqa: F401  (sets sys.path)
from hub.collapse import CollapseConfirmer
from hub.config import load_config
from hub.man_down import ManDownTimer
from hub.posture_track import PostureTrack
from hub.topple import ToppleTrigger, run_topple

CFG = load_config()


def run_track(t, acc, gyro, valid=None, gaps_reset=True):
    """PostureTrack + T3 over a stream; returns (track, infos, fire times)."""
    track = PostureTrack(CFG["posture"])
    trig = ToppleTrigger(CFG["topple"])
    infos, fires = [], []
    last = None
    for i in range(len(t)):
        if gaps_reset and last is not None and t[i] - last > CFG["time"]["tilt_reset_gap_s"]:
            track.reset_tilt()
            if t[i] - last >= CFG["time"]["long_gap_s"]:
                trig.reset_timing()
        last = t[i]
        v = [True] * 4 if valid is None else list(valid[i])
        info, _ = track.step(float(t[i]), acc[i].tolist(), gyro[i].tolist(), v)
        infos.append(info)
        if trig.step(float(t[i]), info["rot_pl"], info["down"], info["not_down"]):
            fires.append(float(t[i]))
    return track, infos, fires


class TestPostureTrack(unittest.TestCase):
    def test_inclination_follows_script(self):
        t, acc, gyro, th = synth.stream(synth.Script().hold(2).move(2.0, (60, 60, 60, 60)).hold(3))
        _, infos, _ = run_track(t, acc, gyro)
        self.assertAlmostEqual(infos[-1]["incl"][0], 60.0, delta=2.0)
        self.assertTrue(infos[100]["upright"])

    def test_real_dt_at_50_hz(self):
        script = synth.Script().hold(2).move(0.6, (90, 90, 90, 90)).hold(2)
        t, acc, gyro, _ = synth.stream(script, fs=50.0)
        _, infos, _ = run_track(t, acc, gyro)
        # just after the move the gyro has carried the estimate (accel gated during motion)
        i = int(np.searchsorted(t, 2.7))
        self.assertGreater(infos[i]["incl"][0], 75.0)

    def test_dead_slots(self):
        t, acc, gyro, _ = synth.stream(synth.Script().hold(3))
        valid = np.ones((len(t), 4), bool)
        valid[:, 2:] = False                     # two dead slots: posture unknown
        _, infos, _ = run_track(t, acc, gyro, valid)
        self.assertFalse(infos[-1]["upright"] or infos[-1]["down"] or infos[-1]["not_down"])


class TestTopple(unittest.TestCase):
    def test_fires_once_on_topple(self):
        t, acc, gyro, _ = synth.stream(synth.topple_script())
        _, _, fires = run_track(t, acc, gyro)
        self.assertEqual(len(fires), 1)
        self.assertTrue(10.0 < fires[0] < 11.5)

    def test_silent_on_slow_lie_down(self):
        t, acc, gyro, _ = synth.stream(synth.topple_script(fall_s=3.0))
        _, _, fires = run_track(t, acc, gyro)
        self.assertEqual(fires, [])

    def test_silent_on_fast_stoop_or_kneel(self):
        # upper trunk goes to 80 deg fast, pelvis stays upright: not "down"
        t, acc, gyro, _ = synth.stream(synth.Script().hold(5).move(0.5, (15, 70, 80, 85)).hold(10))
        _, _, fires = run_track(t, acc, gyro)
        self.assertEqual(fires, [])

    def test_silent_on_strip_on_table(self):
        t, acc, gyro, _ = synth.stream(synth.Script((90, 90, 90, 90)).hold(30), breathing=False)
        _, _, fires = run_track(t, acc, gyro)
        self.assertEqual(fires, [])

    def test_short_gap_at_event_keeps_history(self):
        t, acc, gyro, _ = synth.stream(synth.topple_script())
        keep = ~((t >= 10.2) & (t < 10.9))       # 0.7 s gap in the middle of the descent
        _, _, fires = run_track(t[keep], acc[keep], gyro[keep])
        self.assertEqual(len(fires), 1)

    def test_long_gap_resets_timing(self):
        t, acc, gyro, _ = synth.stream(synth.topple_script())
        keep = ~((t >= 10.1) & (t < 13.0))       # 2.9 s gap: no descent can be claimed
        _, _, fires = run_track(t[keep], acc[keep], gyro[keep])
        self.assertEqual(fires, [])

    def test_one_dead_slot_three_live_anchors(self):
        t, acc, gyro, _ = synth.stream(synth.topple_script())
        for dead in range(4):
            valid = np.ones((len(t), 4), bool)
            valid[:, dead] = False
            _, _, fires = run_track(t, acc, gyro, valid)
            self.assertEqual(len(fires), 1, f"dead slot {dead}")

    def test_batch_helper_matches_streaming(self):
        t, acc, gyro, _ = synth.stream(synth.Script().hold(5).move(0.6, (90,) * 4).hold(5).move(1.0, (0,) * 4)
                                       .hold(5).move(0.5, (95,) * 4).hold(5))
        _, infos, fires = run_track(t, acc, gyro)
        idx = run_topple(t, [i["rot_pl"] for i in infos], [i["down"] for i in infos], [i["not_down"] for i in infos],
                         CFG["topple"])
        self.assertEqual([float(t[i]) for i in idx], fires)
        self.assertEqual(len(fires), 2)          # once per descent

    def test_threshold_is_respected(self):
        trig = ToppleTrigger(CFG["topple"])
        self.assertFalse(trig.step(0.0, 0.0, False, True))
        self.assertFalse(trig.step(0.5, 2.9, True, False))     # below 3.0 rad/s
        trig = ToppleTrigger(CFG["topple"])
        trig.step(0.0, 0.0, False, True)
        self.assertTrue(trig.step(0.5, 3.1, True, False))
        trig = ToppleTrigger(CFG["topple"])
        trig.step(0.0, 0.0, False, True)
        self.assertFalse(trig.step(3.5, 5.0, True, False))     # not-down seen > 3 s ago


def c3_run(script, fs=100.0, breathing=True, t3=None):
    t, acc, gyro, _ = synth.stream(script, fs=fs, breathing=breathing)
    track = PostureTrack(CFG["posture"])
    trig = ToppleTrigger(CFG["topple"])
    c3 = CollapseConfirmer(CFG["collapse"])
    decisions = []
    for i in range(len(t)):
        info, closed = track.step(float(t[i]), acc[i].tolist(), gyro[i].tolist(), [True] * 4)
        if trig.step(float(t[i]), info["rot_pl"], info["down"], info["not_down"]):
            c3.open(float(t[i]))
        d = c3.on_sample(float(t[i]), track)
        if d:
            decisions.append(d)
        for b in closed:
            d = c3.on_bin(b, track, float(t[i]))
            if d:
                decisions.append(d)
    return decisions


class TestCollapse(unittest.TestCase):
    def test_fast_confirm(self):
        d = c3_run(synth.topple_script())
        self.assertEqual([x["rule"] for x in d], ["topple_still_down"])
        self.assertLess(d[0]["t"] - d[0]["trigger_t"], 9.5)

    def test_brief_movement_tolerated(self):
        s = synth.Script().hold(10).move(0.6, (90, 90, 95, 100)).hold(2.5)
        for _ in range(5):
            s.jerk(0.2, 20.0).hold(1.5)
        s.hold(30)
        d = c3_run(s)
        self.assertEqual(d[0]["kind"], "fall_confirmed")
        self.assertIn(d[0]["rule"], ("topple_still_down", "topple_mostly_still"))

    def test_return_to_upright_rejects(self):
        s = synth.Script().hold(10).move(0.6, (90,) * 4).hold(1.0).move(1.5, (0,) * 4).hold(30)
        d = c3_run(s)
        self.assertEqual(d[0]["kind"], "recovered")
        self.assertIn("ended_upright", d[0]["tags"])

    def test_moving_on_the_ground_rejects_at_end(self):
        s = synth.Script().hold(10).move(0.6, (90,) * 4)
        for _ in range(45):
            s.jerk(0.5, 25.0).hold(0.3)
        d = c3_run(s)
        self.assertEqual(d[0]["kind"], "recovered")
        self.assertEqual(d[0]["rule"], "moving_or_up")

    def test_no_motion_tags_but_confirms(self):
        d = c3_run(synth.topple_script(), breathing=False)
        self.assertEqual(d[0]["kind"], "fall_confirmed")
        self.assertIn("no_motion", d[0]["tags"])


def t4_run(script, arm_at=None, breathing=True, work_mode="normal"):
    t, acc, gyro, _ = synth.stream(script, breathing=breathing)
    track = PostureTrack(CFG["posture"])
    t4 = ManDownTimer(CFG["man_down"])
    out = []
    for i in range(len(t)):
        ti = float(t[i])
        _, closed = track.step(ti, acc[i].tolist(), gyro[i].tolist(), [True] * 4)
        if arm_at is not None and abs(ti - arm_at) < 0.005:
            t4.arm(ti, "test")
        t4.on_sample(ti, track)
        for b in closed:
            out += t4.on_bin(b, track, ti, work_mode)
    return out


class TestManDown(unittest.TestCase):
    def test_escalates_when_armed(self):
        out = t4_run(synth.topple_script(after_s=80), arm_at=10.6)
        kinds = {d["kind"]: d["t"] for d in out}
        self.assertIn("man_down_warning", kinds)
        self.assertIn("man_down_critical", kinds)
        self.assertTrue(40 <= kinds["man_down_warning"] <= 46)
        self.assertTrue(70 <= kinds["man_down_critical"] <= 76)

    def test_uncalibrated_bias_still_escalates(self):
        """MPU-6050 worst case without calibration: 20 dps gyro offset, 3 % accel scale."""
        t, acc, gyro, _ = synth.stream(synth.topple_script(after_s=80), accel_scale=1.03)
        gyro = gyro + 0.35
        track = PostureTrack(CFG["posture"])
        t4 = ManDownTimer(CFG["man_down"])
        kinds = []
        for i in range(len(t)):
            ti = float(t[i])
            _, closed = track.step(ti, acc[i].tolist(), gyro[i].tolist(), [True] * 4)
            if abs(ti - 10.6) < 0.005:
                t4.arm(ti, "test")
            t4.on_sample(ti, track)
            for b in closed:
                kinds += [d["kind"] for d in t4.on_bin(b, track, ti)]
        self.assertIn("man_down_critical", kinds)

    def test_not_armed_stays_silent(self):
        self.assertEqual(t4_run(synth.topple_script(after_s=80), arm_at=None), [])

    def test_brief_movement_does_not_reset(self):
        s = synth.Script().hold(10).move(0.6, (90,) * 4).hold(20).jerk(0.3, 20).hold(60)
        out = t4_run(s, arm_at=10.6)
        crit = [d["t"] for d in out if d["kind"] == "man_down_critical"]
        self.assertTrue(crit and crit[0] <= 76)

    def test_upright_resets(self):
        s = synth.Script().hold(10).move(0.6, (90,) * 4).hold(25).move(1.5, (0,) * 4).hold(8) \
            .move(1.5, (90,) * 4).hold(20)
        self.assertEqual(t4_run(s, arm_at=10.6), [])

    def test_no_motion_after_trigger_escalates(self):
        out = t4_run(synth.topple_script(after_s=80), arm_at=10.6, breathing=False)
        self.assertTrue(any(d["kind"] == "man_down_critical" and d["rule"] == "no_motion_after_trigger" for d in out))

    def test_fix_flags_on_by_default(self):
        self.assertTrue(CFG["man_down"]["disarm_on_upright"])
        self.assertTrue(CFG["man_down"]["lying_work_quiet"])

    @staticmethod
    def _t4_cfg(script, cfg, arm_at=None, work_mode="normal"):
        t, acc, gyro, _ = synth.stream(script)
        track, t4, out = PostureTrack(CFG["posture"]), ManDownTimer(cfg), []
        for i in range(len(t)):
            ti = float(t[i])
            _, closed = track.step(ti, acc[i].tolist(), gyro[i].tolist(), [True] * 4)
            if arm_at is not None and abs(ti - arm_at) < 0.005:
                t4.arm(ti, "test")
            t4.on_sample(ti, track)
            for b in closed:
                out += t4.on_bin(b, track, ti, work_mode)
        return out

    def test_upright_disarms_before_arm_s(self):
        """Default: 5 s upright cancels arming; a later stoop or lie-down cannot escalate."""
        s = synth.Script().hold(10).move(0.6, (90,) * 4).hold(5).move(1.5, (0,) * 4).hold(8) \
            .move(1.0, (90,) * 4).hold(80)
        self.assertEqual(t4_run(s, arm_at=10.6), [])
        # opt-out (old reading of 3.6): armed for >= 120 s, so the second lie-down escalates
        old = dict(CFG["man_down"], disarm_on_upright=False)
        self.assertTrue(any(d["kind"] == "man_down_critical" for d in self._t4_cfg(s, old, arm_at=10.6)))

    def test_lying_work_quiet_rejects_slow_motion(self):
        s = synth.Script((90,) * 4)
        for _ in range(40):
            s.move(2.0, (95, 95, 95, 95)).move(2.0, (90,) * 4)   # slow 5 deg lying-work motion
        self.assertEqual(t4_run(s, arm_at=None, work_mode="lying_work"), [])            # default
        old = dict(CFG["man_down"], lying_work_quiet=False)
        self.assertTrue(self._t4_cfg(s, old, work_mode="lying_work"))                    # opt-out fires

    def test_lying_work_timer(self):
        s = synth.Script((90,) * 4).hold(200)
        out = t4_run(s, arm_at=None, work_mode="lying_work")
        kinds = {d["kind"]: d["t"] for d in out}
        self.assertTrue(150 <= kinds["man_down_warning"] <= 153)
        self.assertTrue(180 <= kinds["man_down_critical"] <= 183)
        self.assertEqual(t4_run(s, arm_at=None, work_mode="normal"), [])


if __name__ == "__main__":
    unittest.main()
