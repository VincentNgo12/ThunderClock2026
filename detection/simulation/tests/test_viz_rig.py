# File: test_viz_rig.py
# Purpose: The viewer's procedural rig moves like a body: feet never teleport, knees and
#          elbows never flip, ladder feet stand on rungs, footsteps land on the simulated heel
#          strikes, slip and collapse feet behave, and the head stays out of the floor without
#          snapping. Uses physics=False so only the procedural rig is tested.
# Contributors: Unassigned

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scipy.signal import argrelextrema
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thundersim.viz import export, rig  # noqa: E402

OBS = 5.0                       # short observation tail keeps each payload under ~2 s
FALLING = (2, 3, 10)            # fall_descent, impact, free_fall: the trunk itself flies
LYING = (4, 8)                  # lying, lying_work
J = {name: i for i, name in enumerate(rig.JOINTS)}
FEET = [J[n] for n in ("ankle_l", "ankle_r", "heel_l", "heel_r", "toe_l", "toe_r")]
SOLES = [J[n] for n in ("heel_l", "heel_r", "toe_l", "toe_r")]


def build(scenario: str, seed: int) -> dict:
    """Payload plus the rig, scene, context and leg IK targets behind it."""
    seen: dict = {}
    real_build, real_two_bone = export.build_rig, rig.two_bone

    def capture_rig(ctx, scene, frames, body_idx, strip, shift=None):
        seen["rig"] = real_build(ctx, scene, frames, body_idx, strip, shift)
        seen.update(ctx=ctx, scene=scene, frames=frames)
        return seen["rig"]

    def capture_ik(root, target, *args, **kwargs):
        seen.setdefault("ik_targets", []).append(target.copy())
        return real_two_bone(root, target, *args, **kwargs)

    with mock.patch.object(export, "build_rig", capture_rig), mock.patch.object(rig, "two_bone", capture_ik):
        seen["payload"] = export.run_payload(scenario, seed, "none", "target", OBS, physics=False)
    r, ctx = seen["rig"], seen["ctx"]
    seen["t"] = seen["frames"] / ctx.fs
    seen["phase"] = ctx.track.phase[seen["frames"]]
    seen["ankle_target"] = seen["ik_targets"][0]          # legs are solved first
    seen["joints"] = r.joints
    return seen


def mode_at(modes, t: float) -> str:
    current = modes[0][2]
    for t0, _t1, name in modes:
        if t0 <= t:
            current = name
    return current


def support(scene, points, below) -> np.ndarray:
    return np.array([scene.support(p[0], p[2], b) for p, b in zip(points, below, strict=True)])


class RigMotion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = {key: build(*key) for key in (("A-RUN", 2), ("A-KNEEL", 1), ("A-JUMP", 2), ("A-HAMMER", 2),
                                                  ("H-LAD", 1), ("A-LADDER", 2), ("L-TRIP", 3), ("L-SLIP", 2),
                                                  ("SYNC", 1), ("ARR", 1), ("STRUCK-KD", 1))}

    def test_feet_never_teleport(self):
        """Ankle, heel and toe move < 12 cm per 20 ms frame relative to the pelvis while the
        trunk is not falling (walking, running, jumping, kneeling, ladders, collapse)."""
        for key in (("A-RUN", 2), ("A-KNEEL", 1), ("A-JUMP", 2), ("H-LAD", 1), ("A-LADDER", 2), ("L-TRIP", 3),
                    ("SYNC", 1), ("ARR", 1)):
            run = self.runs[key]
            rel = run["joints"][:, FEET] - run["rig"].pelvis[:, None, :]
            step = np.linalg.norm(np.diff(rel, axis=0), axis=2).max(axis=1)
            calm = ~np.isin(run["phase"][1:], FALLING) & ~np.isin(run["phase"][:-1], FALLING)
            worst = int(np.argmax(np.where(calm, step, 0.0)))
            self.assertLess(step[calm].max(), 0.12, f"{key} at t={run['t'][worst + 1]:.2f}")

    def test_knees_and_elbows_never_flip(self):
        """No knee or elbow jumps > 12 cm in one frame while its hand or foot barely moves
        (a bend-plane flip), in any phase, falls included."""
        for key, run in self.runs.items():
            j = run["joints"]
            for mid, root, end in (("knee_l", "hip_l", "ankle_l"), ("knee_r", "hip_r", "ankle_r"),
                                   ("elbow_l", "shoulder_l", "hand_l"), ("elbow_r", "shoulder_r", "hand_r")):
                jump_mid = np.linalg.norm(np.diff(j[:, J[mid]] - j[:, J[root]], axis=0), axis=1)
                jump_end = np.linalg.norm(np.diff(j[:, J[end]] - j[:, J[root]], axis=0), axis=1)
                flips = np.flatnonzero((jump_mid > 0.12) & (jump_mid > 3 * jump_end))
                self.assertEqual(len(flips), 0, f"{key} {mid} flips at t={run['t'][flips[:3] + 1]}")

    def test_bone_lengths_stay_exact(self):
        for key, run in self.runs.items():
            j, dims = run["joints"], run["rig"].dims
            for a, b, length in (("hip_l", "knee_l", dims.thigh), ("knee_r", "ankle_r", dims.shank),
                                 ("shoulder_l", "elbow_l", dims.upper_arm), ("elbow_r", "hand_r", dims.forearm)):
                err = np.abs(np.linalg.norm(j[:, J[a]] - j[:, J[b]], axis=1) - length)
                self.assertLess(err.max(), 1e-3, f"{key} {a}-{b}")

    def test_ladder_feet_stand_on_rungs(self):
        """On ladders the leg IK reaches its rung target (no foot left hanging short of a rung),
        and a foot at rest sits on a rung top or the ground at the ladder's foot."""
        for key in (("H-LAD", 1), ("A-LADDER", 2)):
            run = self.runs[key]
            r, scene, t = run["rig"], run["scene"], run["t"]
            ankles = run["joints"][:, [J["ankle_l"], J["ankle_r"]]]
            on_ladder = np.array([mode_at(r.leg_modes, x) == "ladder" for x in t])
            self.assertGreater(on_ladder.sum(), 100, key)
            miss = np.linalg.norm(ankles - run["ankle_target"], axis=-1).max(axis=1)
            self.assertLess(np.mean(miss[on_ladder] > 0.02), 0.02, f"{key}: feet short of their rungs")
            speed = np.r_[np.zeros((1, 2)), np.linalg.norm(np.diff(ankles, axis=0), axis=2) / 0.02]
            for i, s in zip(*np.nonzero(on_ladder[:, None] & (speed < 0.05)), strict=True):
                sole = ankles[i, s, 1] - r.dims.ankle_h
                gaps = [np.abs(sole - (lad.base[1] + 0.3 * np.arange(lad.rungs + 1)
                                       + (0.0 if lad.kind == "steps" else 0.03) * (np.arange(lad.rungs + 1) > 0))).min()
                        for lad in scene.ladders]
                self.assertLess(min(gaps), 0.02, f"{key} t={t[i]:.2f} foot {s} off the rungs")

    def test_footsteps_land_on_the_simulated_heel_strikes(self):
        """Rig touchdowns (a foot stopping after a swing) coincide with the simulated heel
        strikes, taken as the minima of the 1 kHz pelvis bob during walking."""
        run = self.runs[("A-RUN", 2)]
        ctx, t = run["ctx"], run["t"]
        y = ctx.track.root_pos[:, 1]
        strikes = []
        for call in ctx.rec.find("walk"):
            a, b = int(call.t0 * ctx.fs), int(call.t1 * ctx.fs)
            minima = argrelextrema(y[a:b], np.less_equal, order=int(0.18 * ctx.fs))[0]
            strikes += [(a + m) / ctx.fs for m in minima if 0.8 * ctx.fs < m < (b - a) - 0.8 * ctx.fs]
        self.assertGreater(len(strikes), 6)
        touchdowns = []
        for j in (J["ankle_l"], J["ankle_r"]):
            speed = np.r_[0.0, np.linalg.norm(np.diff(run["joints"][:, j], axis=0), axis=1) / 0.02]
            for i in range(6, len(t)):
                if speed[i] < 0.12 <= speed[i - 1] and (speed[i - 6:i] > 0.4).any():
                    touchdowns.append(t[i])
        touchdowns = np.array(touchdowns)
        offsets = np.array([np.abs(touchdowns - s).min() for s in strikes])
        self.assertLess(np.median(offsets), 0.06)
        self.assertLess(np.percentile(offsets, 90), 0.12)

    def test_slip_feet_slide_out_on_the_floor(self):
        run = self.runs[("L-SLIP", 2)]
        ctx, t, j = run["ctx"], run["t"], run["joints"]
        t_on = ctx.t_ev("fall_onset")
        t_hit = ctx.first_after(("impact",), t_on)
        sel = np.flatnonzero((t >= t_on) & (t < t_hit))
        for a in (J["ankle_l"], J["ankle_r"]):
            slide = np.linalg.norm((j[sel[-1], a] - j[sel[0], a])[[0, 2]])
            self.assertGreater(slide, 0.15, "the feet slide out with the slip")
        self.assertGreater(j[sel][:, SOLES, 1].min() - run["scene"].ground_y, -0.02, "slipping feet stay out of the floor")

    def test_collapsing_feet_stay_planted(self):
        run = self.runs[("SYNC", 1)]
        ctx, t, j = run["ctx"], run["t"], run["joints"]
        t_on = ctx.t_ev("fall_onset")
        sel = (t >= t_on) & (t < ctx.first_after(("impact",), t_on))
        lowest = j[sel][:, SOLES, 1].min(axis=1) - run["scene"].ground_y
        self.assertGreater(lowest.min(), -0.02)
        self.assertLess(np.mean(lowest > 0.05), 0.05, "feet hover during the collapse")

    def test_head_stays_out_of_the_floor_when_lying(self):
        for key in (("L-TRIP", 3), ("SYNC", 1)):
            run = self.runs[key]
            r, j = run["rig"], run["joints"]
            lying = np.isin(run["phase"], LYING)
            self.assertGreater(lying.sum(), 50, key)
            head, neck = j[lying, J["head"]], j[lying, J["neck"]]
            floor = support(run["scene"], head, neck[:, 1] + 0.02)
            clearance = head[:, 1] - floor - 0.08 * r.dims.s
            self.assertGreater(clearance.min(), -0.01, key)

    def test_head_does_not_snap(self):
        """Head rotation relative to the upper thorax stays below 7 deg per frame outside falls,
        including when consciousness is lost (ARR, SYNC) and while lying."""
        for key in (("ARR", 1), ("SYNC", 1), ("L-TRIP", 3), ("STRUCK-KD", 1)):
            run = self.runs[key]
            n = run["payload"]["meta"]["frames"]
            upper = Rotation.from_quat(np.asarray(run["payload"]["frames"]["segq"]).reshape(n, 4, 4)[:, 3])
            rel = (upper.inv() * Rotation.from_quat(run["rig"].head_q)).as_quat()
            step = 2 * np.degrees(np.arccos(np.clip(np.abs(np.sum(rel[1:] * rel[:-1], axis=1)), -1, 1)))
            calm = ~np.isin(run["phase"][1:], FALLING)
            self.assertLess(step[calm].max(), 7.0, key)

    def test_rig_is_deterministic(self):
        again = build("L-TRIP", 3)
        np.testing.assert_array_equal(again["joints"], self.runs[("L-TRIP", 3)]["joints"])
        np.testing.assert_array_equal(again["rig"].head_q, self.runs[("L-TRIP", 3)]["rig"].head_q)


class LiftingRig(unittest.TestCase):
    """A-CARRY lifts and put-downs (motions.Builder.lift / put_down): planted feet with the heels
    down while the hips drop, knees forward, hands on the load at the simulated grip, and the
    load raised by the body, not by an arm curl."""

    @classmethod
    def setUpClass(cls):
        cls.carry = build("A-CARRY", 0)
        ctx = cls.carry["ctx"]
        cls.calls = [c for c in ctx.rec.calls if c.name in ("lift", "put_down")]
        cls.events = {name: [(t, info) for t, n, info in ctx.events if n == name]
                      for name in ("lift_grasp", "lift_off", "lift_end", "put_down_touch", "put_down_release")}

    def frame(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.carry["t"], t), 0, len(self.carry["t"]) - 1))

    def test_feet_stay_planted_with_heels_down(self):
        """Through each lift and put-down the toes stay put and every foot keeps a sole point on
        the floor (a trailing foot may roll over its ball while upright); from grasp to lift-off
        and from touchdown to release, with the hips down, both heels are on the floor."""
        j, floor = self.carry["joints"], self.carry["scene"].ground_y
        self.assertGreaterEqual(len(self.calls), 4)
        for c in self.calls:
            sel = slice(self.frame(c.t0) + 1, self.frame(c.t1))
            for s in ("l", "r"):
                toe = j[sel, J[f"toe_{s}"]]
                drift = np.linalg.norm((toe - toe[0])[:, [0, 2]], axis=1).max()
                self.assertLess(drift, 0.01, f"{c.name} at {c.t0:.1f}: toe_{s} slides {drift:.3f} m")
                lowest = j[sel][:, [J[f"heel_{s}"], J[f"toe_{s}"]], 1].min(axis=1) - floor
                self.assertLess(np.abs(lowest).max(), 0.01, f"{c.name} at {c.t0:.1f}: foot {s} leaves the floor")
        for low, up in (("lift_grasp", "lift_off"), ("put_down_touch", "put_down_release")):
            for (t0, _), (t1, _) in zip(self.events[low], self.events[up], strict=True):
                sel = slice(self.frame(t0), self.frame(t1))
                heels = j[sel][:, [J["heel_l"], J["heel_r"]], 1] - floor
                self.assertLess(heels.max(), 0.01, f"{low} at {t0:.1f}: a heel is up")

    def test_knees_bend_forward_as_the_hips_drop(self):
        j = self.carry["joints"]
        heading = self.carry["ctx"].track.heading[self.carry["frames"]]
        for name in ("lift_grasp", "put_down_touch"):
            for t, info in self.events[name]:
                k = self.frame(t + 0.1)
                forward = rig.fwd(heading[k])
                for s in ("l", "r"):
                    hip, knee, ankle = (j[k, J[f"{x}_{s}"]] for x in ("hip", "knee", "ankle"))
                    u, v = hip - knee, ankle - knee
                    flexion = 180 - np.degrees(np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v)))
                    self.assertAlmostEqual(flexion, info["knee_deg"], delta=15.0, msg=f"{name} at {t:.1f}")
                    line = (hip + ankle) / 2
                    self.assertGreater(float((knee - line) @ forward), 0.02, f"{name} at {t:.1f}: knee not forward")

    def test_hands_reach_the_grip(self):
        j = self.carry["joints"]
        for name in ("lift_grasp", "put_down_touch"):
            for t, info in self.events[name]:
                k = self.frame(t + 0.1)
                grip = np.array([info["gx"], info["gy"], info["gz"]])
                hands = (j[k, J["hand_l"]] + j[k, J["hand_r"]]) / 2
                self.assertLess(np.linalg.norm(hands - grip), 0.02, f"{name} at {t:.1f}")

    def test_load_rises_with_the_body_not_an_arm_curl(self):
        """From lift-off to upright the hands rise by at most 4 cm relative to the shoulders."""
        j = self.carry["joints"]
        for (t_off, _), (t_end, _) in zip(self.events["lift_off"], self.events["lift_end"], strict=True):
            sel = slice(self.frame(t_off), self.frame(t_end) + 1)
            sh = (j[sel, J["shoulder_l"]] + j[sel, J["shoulder_r"]]) / 2
            hands = (j[sel, J["hand_l"]] + j[sel, J["hand_r"]]) / 2
            rel = hands[:, 1] - sh[:, 1]
            self.assertLess(rel.max() - rel[0], 0.04, f"lift at {t_off:.1f}")


class ContinuousIK(unittest.TestCase):
    def test_bend_plane_does_not_flip_when_the_pole_crosses_the_limb(self):
        """A pole that swings through the limb line flips the knee with plain IK; the
        continuous solve turns it gradually instead."""
        n = 50
        root = np.zeros((n, 3))
        target = np.tile([0.0, -0.6, 0.2], (n, 1))
        sweep = np.linspace(-0.4, 0.4, n)                       # pole sweeps sideways through the limb line
        pole = root + (target - root) * 1.5 + np.column_stack([np.sin(sweep) * 0.3, np.zeros(n), np.zeros(n)])
        plain, _ = rig.two_bone(root, target, pole, 0.42, 0.41)
        smooth, end = rig.two_bone(root, target, pole, 0.42, 0.41, continuous=True)
        self.assertGreater(np.linalg.norm(np.diff(plain, axis=0), axis=1).max(), 0.3, "plain IK flips here")
        self.assertLess(np.linalg.norm(np.diff(smooth, axis=0), axis=1).max(), rig.MAX_BEND_STEP_M + 1e-6)
        np.testing.assert_allclose(np.linalg.norm(smooth - root, axis=1), 0.42, atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(end - smooth, axis=1), 0.41, atol=1e-9)

    def test_swing_profile_is_smooth_and_gentle(self):
        u = np.linspace(0.0, 1.0, 2001)
        w = np.array([rig.swing_profile(x) for x in u])
        speed = np.diff(w) / np.diff(u)
        self.assertAlmostEqual(w[0], 0.0)
        self.assertAlmostEqual(w[-1], 1.0)
        self.assertTrue(np.all(speed >= -1e-9))
        self.assertLess(speed.max(), 1.45)                      # peak speed / mean speed
        self.assertLess(max(speed[0], speed[-1]), 1e-3)         # starts and ends at rest


if __name__ == "__main__":
    unittest.main()
