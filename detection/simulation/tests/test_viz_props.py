# File: test_viz_props.py
# Purpose: The viewer's props are physically consistent with the simulated body: moving props
#          never teleport, the crane load is one pendulum (drawn rope = swing length), the loader
#          drives on the ground, sacks and stairs sit under what rests on them, the lanyard is a
#          rope of fixed length, the safety net is a cloth under the body and the knocked-off
#          hard hat falls under gravity until its cable catches it.
# Contributors: Unassigned

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scipy.special import ellipk  # noqa: E402

from thundersim.model_api import GRAVITY  # noqa: E402
from thundersim.viz.export import run_payload  # noqa: E402
from thundersim.viz.scene import body_scale, torso_surface  # noqa: E402

OBS = 20.0
FPS = 50.0
RUNS = {"crane": ("STRUCK-KD", 0), "vehicle": ("A-EQUIP", 0), "ladder": ("A-LADDER", 0), "bags": ("A-CARRY", 0),
        "creeper": ("A-LIEWORK", 0), "hat": ("HAT-OFF", 0), "stairs": ("STAIR", 9), "opening": ("H-SCAF", 3),
        "roof": ("H-ROOF", 1), "lanyard": ("ARR", 2), "net": ("NET", 0)}
PAYLOADS: dict = {}


def setUpModule():
    for key, (scenario, seed) in RUNS.items():
        PAYLOADS[key] = run_payload(scenario, seed, "none", "target", OBS)


# --- payload helpers ------------------------------------------------------------------------
def channel(payload, key):
    data = np.asarray(payload["dyn"][key], float)
    return data.reshape(payload["meta"]["frames"], -1)


def joints(payload):
    return np.asarray(payload["frames"]["joints"], float).reshape(-1, 18, 3)


def drawn_torso(payload, frames):
    """Points on the drawn torso tube (n, k, 3) at the given output frames."""
    n = payload["meta"]["frames"]
    spine = np.asarray(payload["frames"]["spine"], float).reshape(n, 4, 3)[frames]
    segq = np.asarray(payload["frames"]["segq"], float).reshape(n, 4, 4)[frames]
    M = np.stack([Rotation.from_quat(segq[:, i]).as_matrix() for i in range(4)], axis=1)
    body = payload["meta"]["body"]
    return torso_surface(spine, M, body["H"], body["s"], body["w"]).reshape(len(frames), -1, 3)


def support(payload, x, z, below):
    """Top of the highest drawn walkable surface under (x, z) not above `below` (scene.support)."""
    best = float(payload["scene"]["ground_y"])
    for s in payload["scene"]["solids"]:
        dx, dz = x - s["c"][0], z - s["c"][1]
        a = dx * np.cos(s["yaw"]) - dz * np.sin(s["yaw"])
        b = -dx * np.sin(s["yaw"]) - dz * np.cos(s["yaw"])
        if abs(a) <= s["h"][0] + 1e-3 and abs(b) <= s["h"][1] + 1e-3 and best < s["y"] <= below + 1e-6:
            best = s["y"]
    return best


def event_time(payload, name, nth=0):
    return [e["t"] for e in payload["truth"] if e["name"] == name][nth]


# --- tests --------------------------------------------------------------------------------
class NoTeleports(unittest.TestCase):
    """Every moving prop moves continuously: no more than 5 m/s between 50 Hz frames."""

    def test_moving_props_never_jump(self):
        cases = [("crane", "crane", 3), ("vehicle", "vehicle", 3), ("ladder", "stepladder", 3), ("creeper", "creeper", 3),
                 ("hat", "hat", 3)] + [("bags", key, 3) for key in PAYLOADS["bags"]["dyn"] if key.startswith("bag") and key[3:].isdigit()]
        for run, key, dims in cases:
            pos = channel(PAYLOADS[run], key)[:, :dims]
            step = np.linalg.norm(np.diff(pos, axis=0), axis=1).max()
            self.assertLess(step, 5.0 / FPS, f"{run}/{key} jumps {step:.3f} m in one frame")

    def test_hat_leaves_the_head_from_its_on_head_pose(self):
        payload = PAYLOADS["hat"]
        hat, on = channel(payload, "hat"), np.asarray(payload["dyn"]["hat_on"]) > 0.5
        head = joints(payload)[:, 17]
        headq = np.asarray(payload["frames"]["headq"], float).reshape(-1, 4)
        s = payload["meta"]["body"]["s"]
        on_head = head + Rotation.from_quat(headq).apply([0.0, 0.045 * s, 0.005])   # body.js hatOffset
        np.testing.assert_allclose(hat[on, :3], on_head[on], atol=2e-3)
        k0 = int(np.argmax(~on))
        self.assertLess(np.linalg.norm(hat[k0, :3] - on_head[k0 - 1]), 0.06, "no pop at the release")


class CranePendulum(unittest.TestCase):
    def setUp(self):
        self.p = PAYLOADS["crane"]
        self.rig = self.p["dyn"]["crane_rig"]
        self.load = channel(self.p, "crane")
        self.pivot = np.asarray(self.rig["pivot"])
        self.t = np.arange(len(self.load)) / FPS
        self.t_s = event_time(self.p, "struck")

    def test_drawn_rope_is_the_swing_length(self):
        rope = np.linalg.norm(self.load[:, :3] - self.pivot, axis=1)
        np.testing.assert_allclose(rope, self.rig["rope_m"], atol=2e-3)
        # The rope's angle is the swing angle the beam is tilted by.
        rel = self.load[:, :3] - self.pivot
        dvec = np.asarray(self.rig["dvec"])
        angle = np.arctan2(rel @ dvec, -rel[:, 1])
        np.testing.assert_allclose(angle, self.load[:, 3], atol=2e-3)

    def test_swing_obeys_gravity_on_that_rope(self):
        theta = self.load[:, 3]
        rope = self.rig["rope_m"]
        theta0 = -theta[0]                                   # released from rest at -theta0
        # Released from rest, it reaches the bottom of the arc exactly at the strike: an elliptic
        # quarter period of a pendulum as long as the drawn rope.
        quarter = ellipk(np.sin(theta0 / 2) ** 2) / np.sqrt(GRAVITY / rope)
        self.assertAlmostEqual(self.t_s - self.rig["t_release"], quarter, delta=0.01)
        self.assertLess(abs(np.interp(self.t_s, self.t, theta)), 0.003)
        # Energy on that rope: angular speed sqrt(2 g / L (cos theta - cos theta0)) during the swing.
        swing = np.flatnonzero((self.t > self.rig["t_release"] + 0.3) & (self.t < self.t_s - 0.04))
        omega = np.diff(theta)[swing] * FPS
        mid = (theta[swing] + theta[swing + 1]) / 2
        expected = np.sqrt(2 * GRAVITY / rope * (np.cos(mid) - np.cos(theta0)))
        self.assertAlmostEqual(float(np.median(omega / expected)), 1.0, delta=0.05)

    def test_load_rests_on_the_crib_until_the_release(self):
        before = self.t < self.rig["t_release"]
        self.assertTrue(before.any())
        self.assertLess(np.ptp(self.load[before, :3], axis=0).max(), 1e-6, "motionless before the release")
        bottom = self.load[before, 1] - self.rig["half_h"]
        self.assertLess(np.abs(bottom - self.rig["crib"]["top"][1]).max(), 2e-3, "beam on the crib, not hanging off plumb")

    def test_beam_touches_the_worker_at_the_strike(self):
        k = int(round(self.t_s * FPS))
        dvec = np.asarray(self.rig["dvec"])
        beam = self.load[k, :3]
        pts = drawn_torso(self.p, [k])[0]
        j = joints(self.p)[k]
        limbs = np.concatenate([j[a] + (j[b] - j[a]) * np.linspace(0, 1, 9)[:, None] for a, b in ((10, 12), (11, 13), (12, 14), (13, 15))])
        s = self.p["meta"]["body"]["s"]
        band = lambda p, r: np.abs(p[:, 1] - beam[1]) <= self.rig["half_h"] + r  # noqa: E731
        fronts = [pts[band(pts, 0.0)] @ dvec, limbs[band(limbs, 0.05 * s)] @ dvec - 0.045 * s]
        body_front = min(float(f.min()) for f in fronts if len(f))
        gap = body_front - (float(beam @ dvec) + self.rig["half_depth"])
        self.assertLess(abs(gap), 0.03, f"beam face {gap:+.3f} m from the body at the strike")


class Loader(unittest.TestCase):
    def test_wheels_stay_on_the_ground_and_the_seat_takes_the_bounce(self):
        payload = PAYLOADS["vehicle"]
        v = channel(payload, "vehicle")
        ground = np.array([support(payload, x, z, 50.0) for x, z in v[::25, [0, 2]]])
        self.assertLess(np.abs(v[::25, 1] - ground).max(), 0.02, "tyre bottoms within 2 cm of the ground")
        self.assertGreater(np.ptp(v[:, 4]), 0.02, "the suspension seat moves with the operator")
        self.assertTrue(np.all(np.diff(v[:, 5]) >= -1e-9), "odometer never runs backwards")


class RestingProps(unittest.TestCase):
    def test_sacks_stand_on_the_ground(self):
        payload = PAYLOADS["bags"]
        size = next(p["size"] for p in payload["scene"]["props"] if p.get("k") == "bags")
        keys = [k for k in payload["dyn"] if k.startswith("bag") and k[3:].isdigit()]
        self.assertTrue(keys)
        for key in keys:
            bag = channel(payload, key)
            # Resting: no step into or out of the frame. The payload is rounded to 0.1 mm, so a held
            # sack moving slowly (a quiet stand, the start of a put-down) can show one zero step.
            zero = np.linalg.norm(np.diff(bag[:, :3], axis=0), axis=1) < 1e-6
            still = np.r_[True, zero] & np.r_[zero, True]
            for x, y, z, _ in bag[still][::50]:
                self.assertAlmostEqual(y - size[1] / 2, support(payload, x, z, y + 0.3), delta=0.005)

    def test_creeper_board_is_under_the_back(self):
        payload = PAYLOADS["creeper"]
        top = next(p["top"] for p in payload["scene"]["props"] if p.get("k") == "creeper")
        board = channel(payload, "creeper")
        lying = np.flatnonzero(np.asarray(payload["frames"]["phase"]) == 8)[::10]
        pts = drawn_torso(payload, lying)
        worst = 0.0
        for k, p in zip(lying, pts, strict=True):
            rel = p - board[k, :3]
            yaw = board[k, 3]
            across = rel[:, 0] * np.cos(yaw) - rel[:, 2] * np.sin(yaw)
            along = -rel[:, 0] * np.sin(yaw) - rel[:, 2] * np.cos(yaw)
            over = (np.abs(across) <= 0.21) & (np.abs(along) <= 0.5)
            if over.any():
                worst = max(worst, board[k, 1] + top - float(p[over, 1].min()))
        self.assertLess(worst, 0.015, f"board top {worst:.3f} m inside the torso")


class FittedScenes(unittest.TestCase):
    def test_stairs_hold_the_body_at_rest_and_the_walk(self):
        # STAIR seed 7 -> 9: the realism-scorecard contact changes (simulator falls.py) moved seed 7's
        # fall so its lead-in walk now crosses the stairwell; seed 9 is a dev seed that again fits one
        # consistent scene (the property this test pins).
        payload = PAYLOADS["stairs"]
        params = payload["meta"]["params"]
        final_level = 0.18 * params["steps"] / params["impacts"]     # falls.stairs_fall's last support
        t_last = [e["t"] for e in payload["truth"] if e["name"] == "impact"][-1]
        rest = np.arange(int((t_last + 2.0) * FPS), int((t_last + 3.0) * FPS), 5)   # settled (after the render lift)
        for p in drawn_torso(payload, rest):
            low = p[np.argmin(p[:, 1])]
            under = support(payload, low[0], low[2], low[1] + 0.3)
            self.assertAlmostEqual(under, final_level, delta=0.005, msg="the body rests on the final landing")
            self.assertGreater(low[1] - under, -0.06, "not buried in it")
            self.assertLess(low[1] - under, 0.05, "nor floating above it")
        j = joints(payload)
        t_on = event_time(payload, "fall_onset")
        top = 0.18 * params["steps"]
        for i in range(int(5.5 * FPS), int((t_on - 0.3) * FPS), 10):     # the lead-in walk on the top landing
            for heel in (6, 7):
                self.assertAlmostEqual(support(payload, j[i, heel, 0], j[i, heel, 2], top + 0.3), top, delta=1e-3)
        self.assertFalse(any("stairwell" in n for n in payload["meta"]["notes"]), "this seed's walk stays off the flight")

    def test_walk_stays_on_the_deck_when_the_fall_needs_an_opening(self):
        payload = PAYLOADS["opening"]
        self.assertTrue(any("opening" in n for n in payload["meta"]["notes"]))
        j = joints(payload)
        t_on = event_time(payload, "fall_onset")
        pre = np.arange(int(5.5 * FPS), int((t_on - 0.2) * FPS))
        mid = (j[pre][:, 6:8] + j[pre][:, 8:10]) / 2            # mid-soles (heel + toe) / 2, left and right
        sole = np.minimum(j[pre][:, 6:8, 1], j[pre][:, 8:10, 1])
        air = [min(sole[i, s] - support(payload, mid[i, s, 0], mid[i, s, 2], sole[i, s] + 0.3) for s in (0, 1))
               for i in range(0, len(pre), 5)]
        air_s = float(np.sum(np.asarray(air) > 0.1)) * 5 / FPS
        self.assertLess(air_s, 3.0, f"the lead-in walk is on the deck except over the opening ({air_s:.1f} s on air)")

    def test_roof_stance_is_on_the_flat_roof(self):
        payload = PAYLOADS["roof"]
        roof = max(sol["y"] for sol in payload["scene"]["solids"])
        j = joints(payload)
        t_on = event_time(payload, "fall_onset")
        for i in range(int(5.5 * FPS), int((t_on - 0.1) * FPS), 10):     # standing, then kneeling
            for point in (6, 7, 8, 9, 2, 3):                                # heels, toes, knees
                p = j[i, point]
                self.assertAlmostEqual(support(payload, p[0], p[2], roof + 0.3), roof, delta=1e-3,
                                       msg=f"joint {point} over the slope at frame {i}")


class Determinism(unittest.TestCase):
    def test_dyn_is_reproducible(self):
        again = run_payload("STRUCK-KD", 0, "none", "target", OBS)
        self.assertEqual(again["dyn"], PAYLOADS["crane"]["dyn"])
        self.assertEqual(again["scene"]["props"], PAYLOADS["crane"]["scene"]["props"])



# --- physics props: lanyard, safety net, hard hat (scene.py, physics.World) ------------------
def keyed(payload, keys, values, width):
    """Linear interpolation of keyframes (scene.keyframes) to every output frame: (n, width)."""
    n = payload["meta"]["frames"]
    X = np.asarray(values, float).reshape(len(keys), width)
    return np.stack([np.interp(np.arange(n), keys, X[:, j]) for j in range(width)], axis=1)


def lanyard_points(payload):
    rope = payload["dyn"]["lanyard"]
    return keyed(payload, rope["keys"], rope["x"], rope["n"] * 3).reshape(-1, rope["n"], 3)


def cloth(payload):
    """Safety-net cloth vertices per frame (n, nz, nx, 3), decoded like world.js."""
    c = payload["dyn"]["net_cloth"]
    width = c["nx"] * c["nz"] * 3
    offsets, at = np.zeros((len(c["keys"]), width)), 0
    for k in range(len(c["keys"])):
        if k:
            offsets[k] = offsets[k - 1]
        i = -1
        for _ in range(c["nnz"][k]):
            i += c["d"][at] + 1
            offsets[k, i] += c["d"][at + 1]
            at += 2
    frames = np.asarray(c["rest"], float) + keyed(payload, c["keys"], offsets.ravel(), width) / 1000.0
    return frames.reshape(-1, c["nz"], c["nx"], 3)


def surface_under(V, points):
    """Height of the drawn (triangulated) grid V (nz, nx, 3) under points (m, 3); nan outside."""
    a, b, c, d = V[:-1, :-1].reshape(-1, 3), V[:-1, 1:].reshape(-1, 3), V[1:, :-1].reshape(-1, 3), V[1:, 1:].reshape(-1, 3)
    out = np.full(len(points), np.nan)
    for p0, p1, p2 in ((a, c, b), (b, c, d)):          # world.js clothMesh triangles
        e1, e2 = (p1 - p0)[:, [0, 2]], (p2 - p0)[:, [0, 2]]
        det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        w = points[:, None, [0, 2]] - p0[None, :, [0, 2]]
        u = (w[..., 0] * e2[None, :, 1] - w[..., 1] * e2[None, :, 0]) / det
        v = (e1[None, :, 0] * w[..., 1] - e1[None, :, 1] * w[..., 0]) / det
        inside = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
        y = p0[None, :, 1] + u * (p1 - p0)[None, :, 1] + v * (p2 - p0)[None, :, 1]
        hit = inside.any(axis=1)
        y = y[np.arange(len(points)), np.argmax(inside, axis=1)]
        out = np.where(np.isnan(out) & hit, y, out)
    return out


class Lanyard(unittest.TestCase):
    """ARR seed 2: the lead-in walk takes the D-ring 9.6 m from where the old straight lanyard
    was drawn to."""

    def setUp(self):
        self.p = PAYLOADS["lanyard"]
        self.rope = self.p["dyn"]["lanyard"]
        self.pts = lanyard_points(self.p)
        self.length = np.asarray(self.p["dyn"]["lanyard_len"], float)
        self.poly = np.linalg.norm(np.diff(self.pts, axis=1), axis=2).sum(axis=1)
        self.chord = np.linalg.norm(self.pts[:, -1] - self.pts[:, 0], axis=1)
        self.t = np.arange(len(self.pts)) / FPS

    def test_rope_is_never_longer_than_the_lanyard(self):
        over = self.poly / self.length
        self.assertLess(over.max(), 1.02, f"rope {over.max():.3f} x its length at {self.t[np.argmax(over)]:.2f} s")
        self.assertLess(self.length.max(), self.rope["L0"] + 1.07 + 1e-3, "pays out no more than an absorber can")
        np.testing.assert_allclose(self.pts[:, 0], channel(self.p, "dring"), atol=6e-3, err_msg="starts at the D-ring")  # keyframes: 4 mm

    def test_stowed_while_out_of_reach_then_clipped_to_the_anchor(self):
        t_on = event_time(self.p, "fall_onset")
        self.assertIsNotNone(self.rope["t_clip"], "this walk goes beyond the lanyard's reach")
        clipped = self.t >= self.rope["t_clip"] + 0.6
        hook = self.pts[:, -1]
        np.testing.assert_allclose(hook[clipped], np.tile(self.rope["anchor"], (int(clipped.sum()), 1)), atol=2e-3)
        reach = np.linalg.norm(channel(self.p, "dring") - np.array(self.rope["anchor"]), axis=1)
        self.assertTrue(np.all(reach[clipped & (self.t < t_on)] <= self.rope["L0"]), "slack until the fall")
        self.assertLess(self.poly[self.t < t_on].max(), 1.02 * self.rope["L0"], "no long straight line before the fall")

    def test_slack_rope_sags_and_goes_taut_at_the_arrest(self):
        t_ff = event_time(self.p, "free_fall_end")
        t_on = event_time(self.p, "fall_onset")
        line = self.pts[:, :1] + (self.pts[:, -1:] - self.pts[:, :1]) * np.linspace(0, 1, self.rope["n"])[None, :, None]
        sag = (line[..., 1] - self.pts[..., 1]).max(axis=1)
        slack = (self.t > t_on) & (self.chord < 0.85 * self.length)
        self.assertTrue(slack.any())
        self.assertGreater(sag[slack].min(), 0.05, "a slack lanyard hangs below the straight line")
        taut = np.flatnonzero((self.t > t_on) & (self.chord >= 0.995 * self.rope["L0"]))
        self.assertAlmostEqual(self.t[taut[0]], t_ff, delta=0.14, msg="goes taut when the simulated lanyard does")

    def test_loaded_while_hanging_straight_below_the_anchor(self):
        hang = self.t >= event_time(self.p, "suspended") + 0.6
        self.assertGreater(np.mean(self.chord[hang] / self.poly[hang] > 0.99), 0.95, "straight while loaded")
        ring = channel(self.p, "dring")[hang]
        up = np.array(self.rope["anchor"]) - ring.mean(axis=0)
        self.assertLess(np.degrees(np.arctan2(np.hypot(up[0], up[2]), up[1])), 5.0, "hangs below the anchor")


class SafetyNet(unittest.TestCase):
    """NET seed 0: the body (simulated trunk, rig limbs) pushes a cloth net; nothing goes through."""

    @classmethod
    def setUpClass(cls):
        cls.p = PAYLOADS["net"]
        cls.V = cloth(cls.p)
        cls.J = joints(cls.p)
        cls.t = np.arange(len(cls.J)) / FPS
        cls.t_imp = event_time(cls.p, "impact")

    def below(self, frames, points_of):
        """Largest depth (m) of the given points below the drawn net over these frames."""
        worst = 0.0
        for k in frames:
            pts = points_of(k)
            y = surface_under(self.V[k], pts)
            if np.isfinite(y).any():
                worst = max(worst, float(np.nanmax(y - pts[:, 1])))
        return worst

    def test_net_hangs_taut_from_its_border_before_the_catch(self):
        V0 = self.V[0]
        net = [p for p in self.p["scene"]["props"] if p["t"] == "net"][0]
        border = np.concatenate([V0[0], V0[-1], V0[:, 0], V0[:, -1]])
        np.testing.assert_allclose(border[:, 1], net["p"][1], atol=1e-3)
        self.assertLess(net["p"][1] - V0[..., 1].min(), 0.06, "a few cm of sag under its own weight")
        np.testing.assert_allclose(self.V[-1][0], V0[0], atol=1e-3, err_msg="the border stays on its frame")

    def test_no_joint_below_the_net_after_settling(self):
        settled = np.flatnonzero(self.t >= self.t_imp + 3.0)[::2]
        self.assertLess(self.below(settled, lambda k: self.J[k]), 0.03)

    def test_nothing_goes_through_during_the_catch(self):
        catch = np.flatnonzero((self.t >= self.t_imp - 0.1) & (self.t < self.t_imp + 3.0))
        self.assertLess(self.below(catch, lambda k: self.J[k]), 0.03, "feet and hands stay on the net")
        torso = dict(zip(catch, drawn_torso(self.p, catch), strict=True))
        self.assertLess(self.below(catch, lambda k: torso[k]), 0.05, "the torso stays on the net")

    def test_body_rests_on_the_net(self):
        settled = np.flatnonzero(self.t >= self.t_imp + 3.0)[::10]
        torso = drawn_torso(self.p, settled)
        gaps = []
        for i, k in enumerate(settled):
            y = surface_under(self.V[k], torso[i])
            if np.isfinite(y).any():
                gaps.append(float(np.nanmin(torso[i][:, 1] - y)))
        self.assertGreater(min(gaps), -0.03, "torso not in the net")
        self.assertLess(np.median(gaps), 0.06, "torso lies on it, not above it")


class HardHat(unittest.TestCase):
    """HAT-OFF seed 0: free rigid body until its cable catches it at the simulated tug."""

    @classmethod
    def setUpClass(cls):
        cls.p = PAYLOADS["hat"]
        cls.hat = channel(cls.p, "hat")
        cls.off = np.asarray(cls.p["dyn"]["hat_on"]) < 0.5
        cls.anchor = np.asarray(cls.p["frames"]["anchors"], float).reshape(-1, 4, 3)[:, 3]
        cls.dist = np.linalg.norm(cls.hat[:, :3] - cls.anchor, axis=1)
        cls.t = np.arange(len(cls.hat)) / FPS
        t0 = [a[0] for a in cls.p["activities"] if a[2] == "Hard hat knocked off"][0]
        cls.t_tug = t0 + 0.1                       # motions.hat_off: the tug is 0.1 s into the segment

    def test_cable_catches_the_hat_at_the_simulated_tug(self):
        i = np.flatnonzero(self.off & (self.dist >= 0.545))[0]
        t_taut = self.t[i - 1] + (0.545 - self.dist[i - 1]) / (self.dist[i] - self.dist[i - 1]) / FPS   # between frames
        self.assertAlmostEqual(t_taut, self.t_tug, delta=0.025)
        self.assertLess(self.dist.max(), 0.55 + 0.005, "the cable does not stretch")

    def test_falls_under_gravity_until_caught(self):
        s = self.p["meta"]["body"]["s"]
        com = self.hat[:, :3] + Rotation.from_quat(self.hat[:, 3:]).apply([0.0, 0.095 * s / 6, -0.005 * s / 6])
        flight = np.flatnonzero(self.off & (self.dist < 0.54))
        flight = flight[flight < np.flatnonzero(self.off & (self.dist >= 0.545))[0]][1:]
        a = 2 * np.polyfit(self.t[flight], com[flight, 1], 2)[0]
        self.assertAlmostEqual(a, -GRAVITY, delta=0.4)

    def test_comes_to_rest_hanging_on_its_cable(self):
        last = self.t >= self.t[-1] - 2.0
        rel = self.hat[last, :3] - self.anchor[last]
        self.assertLess(np.linalg.norm(np.diff(rel, axis=0), axis=1).max() * FPS, 0.1, "at rest on its cable")
        self.assertGreater(self.dist[last].min(), 0.53, "hanging from the cable")
        self.assertTrue(np.all(self.hat[self.off, 1] > support(self.p, *self.hat[0, [0, 2]], 1.0) + 0.05), "above the floor")


class PhysicsPropsDeterminism(unittest.TestCase):
    def test_net_cloth_is_reproducible(self):
        again = run_payload("NET", 0, "none", "target", OBS)
        self.assertEqual(again["dyn"]["net_cloth"], PAYLOADS["net"]["dyn"]["net_cloth"])


if __name__ == "__main__":
    unittest.main()
