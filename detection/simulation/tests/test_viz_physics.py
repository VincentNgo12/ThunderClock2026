# File: test_viz_physics.py
# Purpose: The viewer's small physics engine (thundersim/viz/physics.py) obeys gravity and
#          contact, its pose drive and joint limits do not overshoot or launch limbs, a slack
#          lanyard falls at g, and the limb ragdoll changes only the drawn limbs: sensor data
#          and the simulated spine stay exactly as simulated, bones keep their lengths, the
#          engine itself keeps limbs on the support (the final pass only touches up), the
#          drawn body falls at g in free flight and limbs move continuously (no teleports,
#          knee/elbow flips, head snaps or limb snaps while lying) with physics on.
# Contributors: Unassigned

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thundersim.model_api import GRAVITY  # noqa: E402
from thundersim.viz import physics  # noqa: E402
from thundersim.viz.export import run_payload  # noqa: E402
from thundersim.viz.physics import Colliders, World, body_com, support_points  # noqa: E402
from thundersim.viz.scene import Scene, Walkable, rope_step  # noqa: E402

FRAME_DT = 1 / 50                 # viewer output frame (s)
OBS = 5.0                         # short observation after the fall keeps the payloads fast
SCENARIOS = ("H-SCAF", "L-TRIP", "ARR", "NET", "SYNC", "A-JUMP", "A-LIE")
LIMB_JOINTS = (2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 17)          # rig.JOINTS knees .. head
LIMB_ROOT = {2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1, 8: 0, 9: 1, 12: 10, 13: 11, 14: 10, 15: 11, 17: 16}
FEET = (4, 5, 6, 7, 8, 9)
FALLING = (2, 3, 10)              # fall_descent, impact, free_fall: the trunk itself flies or hits
LYING = (4, 5, 6)                 # lying, suspended, recovery


def run_world(world: World, seconds: float) -> np.ndarray:
    """Step a world at the viewer frame rate; positions per frame (frames, particles, 3)."""
    out = []
    for _ in range(int(round(seconds / FRAME_DT))):
        world.step(FRAME_DT)
        out.append(world.x.copy())
    return np.array(out)


def fitted_acceleration(t: np.ndarray, y: np.ndarray) -> float:
    return float(2 * np.polyfit(t - t[0], y, 2)[0])


class EngineBasics(unittest.TestCase):
    def test_free_particle_falls_at_g(self):
        world = World(Colliders(ground_y=None))
        world.add_particles([[0.0, 10.0, 0.0]], mass=1.0)
        y = run_world(world, 0.6)[:, 0, 1]
        t = np.arange(1, len(y) + 1) * FRAME_DT
        self.assertAlmostEqual(fitted_acceleration(t, y), -GRAVITY, delta=0.05)

    def test_resting_particle_does_not_penetrate_or_bounce(self):
        radius = 0.05
        world = World(Colliders(ground_y=0.0))
        world.add_particles([[0.0, 1.0, 0.0]], mass=1.0, radius=radius)
        y = run_world(world, 2.0)[:, 0, 1]
        self.assertGreaterEqual(y.min(), radius - 1e-9)
        landed = int(np.argmax(y <= radius + 1e-6))
        self.assertGreater(landed, 0, "the particle reaches the floor")
        self.assertLess(y[landed:].max() - radius, 1e-3, "restitution ~0: no bounce after touchdown")
        self.assertLess(abs(float(world.velocities[0, 1])), 1e-6)

    def test_pinned_chain_keeps_link_lengths(self):
        # A rope: 10 links of 0.1 m hanging from a pin; 4 solver iterations keep the stretch
        # under 1 mm (1 iteration, the ragdoll's setting, allows ~2.5 mm on this chain).
        world = World(Colliders(ground_y=0.0), iterations=4)
        points = np.column_stack([np.zeros(11), np.full(11, 1.2), np.linspace(0.0, 1.0, 11)])
        ids = world.add_particles(points, mass=0.05, radius=0.01)
        world.add_distance(ids[:-1], ids[1:])
        world.add_pin(ids[:1])
        x = run_world(world, 3.0)
        links = np.linalg.norm(np.diff(x[-1], axis=0), axis=1)
        self.assertLess(np.abs(links - 0.1).max(), 1e-3)
        np.testing.assert_allclose(x[:, 0], np.tile(points[0], (len(x), 1)), atol=1e-12, err_msg="the pin holds")
        self.assertGreaterEqual(x[..., 1].min(), 0.01 - 1e-6)

    def test_drive_step_response_does_not_overshoot(self):
        # A driven particle (no gravity) told to move 0.5 m settles on its target without
        # overshooting at any muscle tone the ragdoll uses (an undamped drive overshoots 23-92 %).
        f_limp = GRAVITY * (1 / physics.SUBSTEP_HZ) ** 2 / physics.LIMP_SAG_M
        for ctrl in (0.2, 0.3, 0.5, 0.6, 0.75, 0.9):
            world = World(Colliders(ground_y=None), gravity=0.0)
            world.add_particles([[0.0, 0.0, 0.0]], mass=1.0)
            x = []
            for _ in range(150):
                world.set_drive([0], [[0.5, 0.0, 0.0]], f_limp ** (1.0 - ctrl))
                world.step(FRAME_DT)
                x.append(world.x[0, 0])
            self.assertLess(max(x) - 0.5, 0.05 * 0.5, f"ctrl {ctrl}: overshoot")
            self.assertAlmostEqual(x[-1], 0.5, delta=0.01, msg=f"ctrl {ctrl}: settles")

    def test_drive_speed_cap_limits_the_approach(self):
        world = World(Colliders(ground_y=None), gravity=0.0)
        world.add_particles([[0.0, 0.0, 0.0]], mass=1.0)
        x = []
        for _ in range(50):
            world.set_drive([0], [[1.0, 0.0, 0.0]], 1.0, max_speed=2.0)
            world.step(FRAME_DT)
            x.append(world.x[0, 0])
        self.assertLess(np.diff(np.r_[0.0, x]).max(), 2.0 * FRAME_DT + 1e-9)
        self.assertAlmostEqual(x[-1], 1.0, delta=1e-6)

    def test_cone_limit_turns_a_limb_back_without_launching_it(self):
        # A child 120 deg off a 55 deg cone is turned back over several substeps, inelastically.
        world = World(Colliders(ground_y=None), gravity=0.0)
        ids = world.add_particles([[0.0, 0.0, 0.0], [0.4 * np.sin(2.09), 0.4 * np.cos(2.09), 0.0]], mass=[np.inf, 1.0])
        world.add_distance(ids[:1], ids[1:])
        cone = world.add_cone(ids[:1], ids[1:], np.radians(55))
        world.set_cone_axis(cone, [[0.0, 1.0, 0.0]])
        x = run_world(world, 0.5)
        speed = np.linalg.norm(np.diff(x[:, 1], axis=0), axis=1) / FRAME_DT
        angle = np.degrees(np.arccos(np.clip(x[-1, 1, 1] / np.linalg.norm(x[-1, 1]), -1, 1)))
        self.assertLessEqual(angle, 55.5, "back inside the cone")
        self.assertLess(speed[5:].max(), 0.05, "no velocity left over from the correction")

    def test_slack_lanyard_falls_at_g(self):
        # scene.rope_step: a slack rope (links longer than their spacing) keeps its own state,
        # so it falls at g; resetting it to the frame-average velocity every frame gave ~0.6 g.
        world = World(Colliders(ground_y=None), iterations=2)
        points = np.column_stack([np.linspace(0.0, 1.2, 13), np.full(13, 10.0), np.zeros(13)])
        ids = world.add_particles(points, mass=0.1, radius=0.01)
        world.add_distance(ids[:-1], ids[1:], lo=0.0, hi=0.15)
        prev, ys = world.x.copy(), []
        for _ in range(int(round(0.5 / FRAME_DT))):
            prev = rope_step(world, ids, 13 * 0.15, prev, FRAME_DT)
            ys.append(prev[:, 1].mean())
        t = np.arange(1, len(ys) + 1) * FRAME_DT
        self.assertAlmostEqual(fitted_acceleration(t, np.array(ys)), -GRAVITY, delta=0.1)

    def test_particles_rest_on_a_rotated_box(self):
        col = Colliders(ground_y=0.0)
        col.add_box([0.0, 0.25, 0.0], [1.0, 0.5, 0.6], yaw=0.7)
        world = World(col)
        drops = np.array([[0.0, 1.0, 0.0], [0.3, 1.5, 0.1], [-0.2, 0.8, -0.1]])
        world.add_particles(drops, mass=0.2, radius=0.02)
        x = run_world(world, 2.0)
        np.testing.assert_allclose(x[-1, :, 1], 0.52, atol=2e-3, err_msg="resting on the top face")
        self.assertGreaterEqual(x[..., 1].min(), 0.52 - 0.02, "never deep inside the box")


def scene_from_payload(payload: dict) -> Scene:
    """Support surfaces as the viewer draws them (ground and walkable solids)."""
    scene = Scene(ground_y=payload["scene"]["ground_y"])
    for s in payload["scene"]["solids"]:
        scene.walkables.append(Walkable(np.array([s["c"][0], 0.0, s["c"][1]]), s["y"], tuple(s["h"]), s["yaw"]))
    return scene


def frames_of(payload: dict, key: str, width: int) -> np.ndarray:
    return np.array(payload["frames"][key]).reshape(payload["meta"]["frames"], -1, width)


def drawn_com(payload: dict) -> np.ndarray:
    """Whole-body COM of the drawn figure (physics.body_com) from the payload alone."""
    segq = frames_of(payload, "segq", 4)
    M = np.stack([Rotation.from_quat(segq[:, k]).as_matrix() for k in range(4)], axis=1)
    body = payload["meta"]["body"]
    return body_com(frames_of(payload, "joints", 3), frames_of(payload, "spine", 3), M, body["H"], body["s"], body["w"])


class RagdollPayloads(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.off = {s: run_payload(s, 0, "none", "target", OBS, physics=False) for s in SCENARIOS}
        cls.on, cls.diag = {}, {}
        for s in SCENARIOS:
            cls.on[s] = run_payload(s, 0, "none", "target", OBS, physics=True)
            cls.diag[s] = {k: physics.LAST_RUN[k] for k in ("final_move", "blend")}

    def test_sensor_data_and_simulated_spine_are_unchanged(self):
        for s in SCENARIOS:
            on, off = self.on[s], self.off[s]
            self.assertEqual(on["signals"], off["signals"], s)
            for key in ("segq", "chipq", "phase", "worn"):
                self.assertEqual(on["frames"][key], off["frames"][key], f"{s} {key}")
            self.assertEqual(on["truth"], off["truth"], s)
            # Spine and strip anchors: at most a common render-only vertical lift (rest lift).
            spine_on, spine_off = frames_of(on, "spine", 3), frames_of(off, "spine", 3)
            np.testing.assert_allclose(spine_on[..., [0, 2]], spine_off[..., [0, 2]], atol=1e-9, err_msg=s)
            lift = spine_on[..., 1] - spine_off[..., 1]
            self.assertLess(np.abs(lift - lift[:, :1]).max(), 2e-4, f"{s}: one lift for the whole spine")
            self.assertGreaterEqual(lift.min(), -1e-4, s)
            self.assertLessEqual(lift.max(), 0.15 + 1e-4, s)
            anchors_on, anchors_off = frames_of(on, "anchors", 3), frames_of(off, "anchors", 3)
            np.testing.assert_allclose(anchors_on[..., [0, 2]], anchors_off[..., [0, 2]], atol=1e-9, err_msg=s)
            worn = np.array(on["frames"]["worn"], bool)
            np.testing.assert_allclose((anchors_on - anchors_off)[worn, :, 1], np.repeat(lift[worn, :1], 4, axis=1),
                                       atol=2e-4, err_msg=s)

    def test_physics_metadata(self):
        for s in SCENARIOS:
            meta = self.on[s]["meta"]["physics"]
            self.assertEqual(meta["engine"], "thundersim-pbd")
            self.assertTrue(meta["enabled"], s)
            self.assertIn("version", meta)
            self.assertEqual(meta["substep_hz"], 250.0)
            self.assertFalse(self.off[s]["meta"]["physics"]["enabled"])
            self.assertEqual(self.off[s]["meta"]["physics"]["windows"], [])
        for s in ("H-SCAF", "L-TRIP", "NET", "SYNC", "ARR"):
            self.assertTrue(self.on[s]["meta"]["physics"]["windows"], f"{s}: the fall is simulated")

    def test_bone_lengths_are_exact(self):
        bones = {"thigh": (0, 2), "shank": (2, 4), "upper_arm": (10, 12), "forearm": (12, 14)}
        for s in SCENARIOS:
            joints, body = frames_of(self.on[s], "joints", 3), self.on[s]["meta"]["body"]
            for name, (a, b) in bones.items():
                for side in (0, 1):
                    length = np.linalg.norm(joints[:, b + side] - joints[:, a + side], axis=1)
                    self.assertLess(np.abs(length - body[name]).max(), 1e-3, f"{s} {name}")

    def test_engine_keeps_limbs_on_the_support(self):
        """The final pass (exact bones, support clamp) only touches up the simulated limbs: where
        the ragdoll runs unblended it moves a joint more than 2 cm in under 2 % of the frames and
        never more than 10 cm, so the engine's own contacts keep the limbs out of the floor."""
        for s in SCENARIOS:
            move, full = self.diag[s]["final_move"], self.diag[s]["blend"] >= 1.0 - 1e-9
            if not full.any():
                continue
            self.assertLess(np.mean(move[full] > 0.02), 0.02, s)
            self.assertLess(move[full].max(), 0.10, s)

    def test_limbs_rest_on_the_support_after_impacts(self):
        """After the impacts, no limb joint centre is more than 2 cm below the support under it
        (surfaces within 5 cm above it), except where its root (hip, shoulder, neck) is itself
        in the floor: the simulated trunk below the support (ground truth), which the limb
        follows."""
        checked = 0
        for s in SCENARIOS:
            payload = self.on[s]
            impacts = [e["t"] for e in payload["truth"]
                       if e["name"] == "impact" and e["info"].get("mode") != "feet_first"]
            if not impacts:
                continue
            joints, scene = frames_of(payload, "joints", 3), scene_from_payload(payload)
            settled = np.arange(len(joints)) * FRAME_DT > impacts[-1] + 1.5
            for j in LIMB_JOINTS:
                pos, root = joints[settled, j], joints[settled, LIMB_ROOT[j]]
                below = support_points(scene, pos[:, 0], pos[:, 2], pos[:, 1] + 0.05)
                ok = root[:, 1] >= below
                if ok.any():
                    self.assertGreater((pos[ok, 1] - below[ok]).min(), -0.02, f"{s} joint {j}")
            checked += 1
        self.assertGreaterEqual(checked, 3)

    def test_drawn_body_falls_closer_to_g_in_free_flight(self):
        errors = {True: [], False: []}
        for s in ("H-SCAF", "ARR", "NET", "A-JUMP"):
            truth = self.on[s]["truth"]
            starts = [e["t"] for e in truth if e["name"] == "free_fall_start"]
            for t0 in starts:
                t1 = min(e["t"] for e in truth if e["name"] == "free_fall_end" and e["t"] > t0)
                for physics, payload in ((True, self.on[s]), (False, self.off[s])):
                    com = drawn_com(payload)
                    t = np.arange(len(com)) * FRAME_DT
                    sel = (t > t0 + 0.04) & (t < t1 - 0.04)       # trim the takeoff and touchdown frames
                    errors[physics].append(abs(fitted_acceleration(t[sel], com[sel, 1]) + GRAVITY))
                with self.subTest(scenario=s, t0=t0):
                    on, off = errors[True][-1], errors[False][-1]
                    self.assertLessEqual(on, max(off, 0.25), f"{s}: |a + g| {on:.2f} with physics vs {off:.2f} without")
        self.assertLess(np.mean(errors[True]), np.mean(errors[False]))

    def test_deterministic(self):
        again = run_payload("L-TRIP", 0, "none", "target", OBS, physics=True)
        self.assertEqual(again["frames"]["joints"], self.on["L-TRIP"]["frames"]["joints"])
        self.assertEqual(again["frames"]["headq"], self.on["L-TRIP"]["frames"]["headq"])
        self.assertEqual(again["meta"]["physics"], self.on["L-TRIP"]["meta"]["physics"])


class RagdollContinuity(unittest.TestCase):
    """Physics on (the viewer default): the rig's continuity checks (tests/test_viz_rig.py runs
    them with physics off) hold for the ragdoll too, on the runs where the verification found
    foot teleports, knee flips, head snaps and limbs whipping when tone comes back."""

    CASES = (("L-SLIP", 0, 20.0), ("A-JUMP", 1, 20.0), ("FALL-GETUP", 1, 20.0), ("H-HOLE", 0, 20.0),
             ("H-SCAF", 0, 20.0), ("L-TRIP", 1, 60.0))

    @classmethod
    def setUpClass(cls):
        cls.runs = {}
        for scenario, seed, obs in cls.CASES:
            p = run_payload(scenario, seed, "none", "target", obs, physics=True)
            cls.runs[(scenario, seed)] = {
                "joints": frames_of(p, "joints", 3), "spine": frames_of(p, "spine", 3),
                "target": physics.LAST_RUN["target"],
                "segq": frames_of(p, "segq", 4), "headq": np.array(p["frames"]["headq"]).reshape(-1, 4),
                "phase": np.array(p["frames"]["phase"])}

    def steps(self, run, key: str = "joints"):
        """Joint steps per frame relative to the pelvis (frames - 1, 18) and the calm mask
        (neither frame in a falling phase)."""
        rel = run[key] - run[key][:, :2].mean(axis=1, keepdims=True)
        step = np.linalg.norm(np.diff(rel, axis=0), axis=2)
        calm = ~np.isin(run["phase"][1:], FALLING) & ~np.isin(run["phase"][:-1], FALLING)
        return step, calm

    def test_feet_never_teleport(self):
        """Outside the falling phases no ankle, heel or toe jumps: a step of more than 12 cm in a
        frame (relative to the hips) that is over twice the steps before and after it, unless
        the procedural pose itself moves nearly as far. (A limp leg may swing fast but smoothly
        while the trunk still bounces or rolls after the impact.)"""
        for key, run in self.runs.items():
            step, calm = self.steps(run)
            posed, _ = self.steps(run, "target")
            feet = step[:, list(FEET)]
            around = np.maximum(np.r_[feet[:1], feet[:-1]], np.r_[feet[1:], feet[-1:]])
            spike = (feet > 0.12) & (feet > 2 * around) & (feet > posed[:, list(FEET)] + 0.05)
            jump = spike.any(axis=1) & calm
            self.assertEqual(jump.sum(), 0, f"{key} at t={(np.flatnonzero(jump)[:3] + 1) * FRAME_DT}")

    def test_knees_and_elbows_never_flip(self):
        for key, run in self.runs.items():
            j = run["joints"]
            _, calm = self.steps(run)
            for mid, root, end in ((2, 0, 4), (3, 1, 5), (12, 10, 14), (13, 11, 15)):
                jump_mid = np.linalg.norm(np.diff(j[:, mid] - j[:, root], axis=0), axis=1)
                jump_end = np.linalg.norm(np.diff(j[:, end] - j[:, root], axis=0), axis=1)
                flips = np.flatnonzero(calm & (jump_mid > 0.12) & (jump_mid > 3 * jump_end))
                self.assertEqual(len(flips), 0, f"{key} joint {mid} flips at t={(flips[:3] + 1) * FRAME_DT}")

    def test_head_does_not_snap(self):
        """Head rotation relative to the upper thorax: at most two rate-limit steps per frame
        outside the falling phases (physics.HEAD_STEP_DEG, plus as much again to lift it off the
        floor)."""
        for key, run in self.runs.items():
            upper = Rotation.from_quat(run["segq"][:, 3])
            rel = (upper.inv() * Rotation.from_quat(run["headq"])).as_quat()
            step = 2 * np.degrees(np.arccos(np.clip(np.abs(np.sum(rel[1:] * rel[:-1], axis=1)), -1, 1)))
            _, calm = self.steps(run)
            self.assertLess(step[calm].max(), 2 * physics.HEAD_STEP_DEG + 0.5, key)

    def test_limbs_do_not_snap_while_lying(self):
        """While lying, suspended or recovering with a slow trunk (< 0.5 m/s), no limb joint
        moves more than 0.2 m in a frame relative to the pelvis, including when tone comes back
        (waking up, struggling, getting up)."""
        for key, run in self.runs.items():
            step, _ = self.steps(run)
            trunk = np.linalg.norm(np.diff(run["spine"], axis=0), axis=2).max(axis=1) / FRAME_DT
            still = np.isin(run["phase"][1:], LYING) & (trunk < 0.5)
            if key[0] != "A-JUMP":                         # the jumps never lie down
                self.assertGreater(still.sum(), 50, key)
            worst = step[:, list(LIMB_JOINTS)].max(axis=1)
            self.assertLess(worst[still].max(initial=0.0), 0.2,
                            f"{key} at t={(np.argmax(np.where(still, worst, 0)) + 1) * FRAME_DT:.2f}")


if __name__ == "__main__":
    unittest.main()
