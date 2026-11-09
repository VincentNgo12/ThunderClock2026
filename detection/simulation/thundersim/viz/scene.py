# File: scene.py
# Purpose: The physical environment each scenario implies (ladder, scaffold edge, roof,
#          floor opening, trench, stairs, net, harness anchor, crane load, equipment...),
#          placed from where the simulated worker actually was when each event happened.
# Contributors: Unassigned
"""Scene construction for the 3D viewer.

The simulator models no environment beyond a support height (Builder.ground_y), so every
prop here is inferred from the run: the scenario id, its parameters, the logged primitive
calls and the ground-truth events. Geometry follows the simulator's own numbers where it has
them (ladder rungs 0.3 m apart at 1:4, stairs 0.18 m rise, platform heights, slide slope,
seat height, table height). Everything else is a plausible design value.

A Scene holds static props (sent to the browser as-is), walkable surfaces (used by the rig
to put feet on the right surface) and ladders (rung lines for the climbing pose). Dynamic
props (vehicle, crane load, hat, bags, carried stepladder, creeper, lanyard, safety net) are
computed per frame by dynamics() after the rig, because some of them follow the hands or touch
the body. The harness lanyard (a rope), the safety net (a cloth) and the knocked-off hard hat
(a rigid body on its cable) are simulated with the viewer's PBD engine (physics.World) against
the drawn body as kinematic capsules (body_capsules); they never move the body.

Where the simulated trajectory fixes the geometry (stairs, roof slope, step-up platform,
bucket, deck edge, crane strike point), the props are fitted to the drawn body (body.js torso tube around the simulated
spine) rather than drawn at catalogue sizes, so the body neither floats above nor sinks into
them. When the simulator's walk and fall cannot share one static geometry, scene.notes says so
(scenes_index then avoids that seed as the default).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal
from scipy.special import ellipk

from ..model_api import GRAVITY
from ..spine import root_rotation
from .geom import UP, flat, fwd, perp, r3, right, smoothstep, unit, yaw_of
from .physics import SCENE_MASK, Colliders, World, scene_colliders

FALL_ENDS = ("impact", "free_fall_end", "arrest_start", "slide_start")

# --- drawn body geometry (static/body.js) ---------------------------------------------------
# Torso cross-sections along the trunk curve: (fraction of the curve, half width / (s w),
# half depth / (s sqrt(w))). The curve runs bottom -> pelvis -> lumbar -> thoracic -> upper -> neck.
TORSO_PROFILE = np.array([[0.0, 0.135, 0.105], [0.07, 0.172, 0.13], [0.2, 0.165, 0.118], [0.32, 0.148, 0.104],
                          [0.45, 0.158, 0.112], [0.62, 0.173, 0.124], [0.8, 0.184, 0.12], [0.88, 0.196, 0.112],
                          [0.94, 0.16, 0.098], [0.975, 0.11, 0.075], [1.0, 0.065, 0.06]])
TORSO_KNOTS = np.array([0.0, 0.2, 0.4, 0.62, 0.9, 1.0])
KNOT_SEGMENT = (0, 0, 1, 2, 3, 3)      # segment frame used around each knot


def body_scale(worker) -> tuple[float, float]:
    """Size and width scales of the drawn body, as rig.Dims: s = H / 1.75, w from the BMI."""
    height = worker.height_m
    width = float(np.clip(np.sqrt(worker.mass_kg / height ** 2 / 24.0), 0.88, 1.25))
    return height / 1.75, width


def spine_pose(ctx, idx) -> tuple[np.ndarray, np.ndarray]:
    """Spine-line points (n, 4, 3) and segment rotations (n, 4, 3, 3) at 1 kHz sample indices."""
    idx = np.atleast_1d(np.asarray(idx, int))
    M = np.stack([ctx.kin.seg_rot[i][idx].as_matrix() for i in range(4)], axis=1)
    P = ctx.kin.anchors[idx] - np.einsum("nsij,j->nsi", M, np.array([0.0, 0.0, ctx.worker.sensor_offset_m]))
    return P, M


def torso_rings(P: np.ndarray, M: np.ndarray, height: float, s: float, w: float) -> list:
    """Cross-sections of the drawn torso tube around spine points P (n, 4, 3) with segment
    rotations M (n, 4, 3, 3), one per TORSO_PROFILE row: (ellipse centre (n, 3), lateral axis,
    posterior axis, half width m, half depth m). Straight segments between the knots stand in
    for body.js's Hermite curve (+Z of each segment frame is posterior)."""
    bottom = P[:, 0] + M[:, 0, :, 1] * (-0.085 * height)
    neck = P[:, 3] + M[:, 3, :, 1] * (0.03 * height)
    knots = np.stack([bottom, P[:, 0], P[:, 1], P[:, 2], P[:, 3], neck], axis=1)
    rings = []
    for frac, a0, b0 in TORSO_PROFILE:
        k = int(np.clip(np.searchsorted(TORSO_KNOTS, frac, side="right") - 1, 0, 4))
        u = (frac - TORSO_KNOTS[k]) / (TORSO_KNOTS[k + 1] - TORSO_KNOTS[k])
        centre = knots[:, k] + (knots[:, k + 1] - knots[:, k]) * u
        seg = KNOT_SEGMENT[k] if u < 0.5 else KNOT_SEGMENT[k + 1]
        xr, zr = M[:, seg, :, 0], M[:, seg, :, 2]
        half_w, half_d = a0 * s * w, b0 * s * np.sqrt(w)
        rings.append((centre - zr * (half_d - 0.04), xr, zr, half_w, half_d))
    return rings


def torso_surface(P: np.ndarray, M: np.ndarray, height: float, s: float, w: float, n_ang: int = 16) -> np.ndarray:
    """Points (n, rings, n_ang, 3) on the drawn torso tube (see torso_rings)."""
    ang = np.arange(n_ang) / n_ang * 2 * np.pi
    ca, sa = np.cos(ang), np.sin(ang)
    back = np.where(sa > 0, 0.92, 1.0)                   # the back is flattened
    rings = [c[:, None] + xr[:, None] * (half_w * ca)[None, :, None] + zr[:, None] * (half_d * sa * back)[None, :, None]
             for c, xr, zr, half_w, half_d in torso_rings(P, M, height, s, w)]
    return np.stack(rings, axis=1)


# --- the drawn body as kinematic colliders (lanyard, net and hat physics) --------------------
STADIUM_RINGS = (0, 1, 2, 4, 5, 6, 7, 8, 9)    # TORSO_PROFILE rows joined by stadium sections
COARSE_RINGS = (0, 2, 5, 7, 9)                 # fewer sections, for props that only brush the body
# (joint a, joint b, radius / s): body.js limbs at their narrow end, the neck
LIMB_CAPSULES = {"legs": ((0, 2, 0.065), (1, 3, 0.065), (2, 4, 0.048), (3, 5, 0.048)),
                 "arms": ((10, 12, 0.046), (11, 13, 0.046), (12, 14, 0.04), (13, 15, 0.04)),
                 "head": ((16, 17, 0.054),)}
HEAD_R = 0.095        # skull radius / s (body.js: a 0.1 s sphere scaled 0.8-1.0)
BOOT_R = 0.045        # boot half height / s: the heel-toe (sole) line raised by this much


def body_capsules(ctx, rig, frames: np.ndarray, parts=("torso", "legs", "feet", "arms", "head"),
                  rings_used=STADIUM_RINGS) -> tuple:
    """The drawn body as capsules: endpoints a, b (N, C, 3) per output frame and radii (C,) in m.
    The torso is a row of stadium-shaped sections (two capsules side by side between body.js
    rings, as wide and deep as the ellipse), limbs are capsules along the rig joints, the boots
    capsules along the soles and the head a sphere."""
    s, w = body_scale(ctx.worker)
    J = rig.joints
    A, B, R = [], [], []
    if "torso" in parts:
        _, M = spine_pose(ctx, frames)
        rings = torso_rings(rig.spine, M, ctx.H, s, w)
        for i, j in zip(rings_used[:-1], rings_used[1:], strict=True):
            (ci, xi, _, ai, bi), (cj, xj, _, aj, bj) = rings[i], rings[j]
            r = (bi + bj) / 2
            for side in (-1.0, 1.0):
                A.append(ci + xi * side * max(ai - r, 0.0))
                B.append(cj + xj * side * max(aj - r, 0.0))
                R.append(r)
    for part in ("legs", "arms", "head"):
        if part in parts:
            for a, b, r in LIMB_CAPSULES[part]:
                A.append(J[:, a])
                B.append(J[:, b])
                R.append(r * s)
    if "head" in parts:
        A.append(J[:, 17])
        B.append(J[:, 17])
        R.append(HEAD_R * s)
    if "feet" in parts:
        for side in (0, 1):
            heel, toe, ankle = J[:, 6 + side], J[:, 8 + side], J[:, 4 + side]
            sole = unit(toe - heel)
            up = unit((ankle - heel) - sole * np.sum((ankle - heel) * sole, axis=1, keepdims=True))
            A.append(heel + up * BOOT_R * s)
            B.append(toe + up * BOOT_R * s)
            R.append(BOOT_R * s)
    return np.stack(A, axis=1), np.stack(B, axis=1), np.array(R)


def add_body(col, caps: tuple, k: int, mu: float = 0.5, inflate: float = 0.0) -> list:
    """Add the body capsules of frame k to a physics.Colliders; returns their ids."""
    a, b, r = caps
    return [col.add_capsule(a[k, c], b[k, c], float(r[c]) + inflate, mu=mu, mask=SCENE_MASK) for c in range(len(r))]


def move_body(col, ids: list, caps: tuple, k: int) -> None:
    """Move the body capsules to frame k over the coming step (kinematic shapes)."""
    a, b, _ = caps
    for c, cid in enumerate(ids):
        col.set_capsule(cid, a[k, c], b[k, c])


def lying_leg_points(ctx, idx, n: int = 6) -> np.ndarray:
    """Points (n_frames, 2 n, 3) along both legs of a body lying at 1 kHz sample indices idx: from
    the hips down the pelvis frame's long axis, a leg's length (the rig's 'lie' leg pose)."""
    P, M = spine_pose(ctx, idx)
    s, w = body_scale(ctx.worker)
    leg = 0.475 * ctx.H * 0.95
    down = -M[:, 0, :, 1]
    pts = []
    for side in (-1.0, 1.0):
        hip = P[:, 0] + M[:, 0] @ np.array([side * 0.088 * w * s, -0.02 * ctx.H, -0.075 * s])
        for u in np.linspace(0.3, 1.0, n):
            pts.append(hip + down * leg * u)
    return np.stack(pts, axis=1)


def torso_points(ctx, idx, shift: np.ndarray | None = None, n_ang: int = 12) -> np.ndarray:
    """Drawn torso points (n, k, 3) at 1 kHz sample indices (simulated pose, optional offset)."""
    P, M = spine_pose(ctx, idx)
    if shift is not None:
        P = P + shift[:, None, :]
    s, w = body_scale(ctx.worker)
    pts = torso_surface(P, M, ctx.H, s, w, n_ang)
    return pts.reshape(len(P), -1, 3)


def quat_slerp(qa: np.ndarray, qb: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Row-wise slerp between unit quaternions (xyzw)."""
    qa, qb = np.atleast_2d(qa).astype(float), np.atleast_2d(qb).astype(float).copy()
    u = np.broadcast_to(np.asarray(u, float), (max(len(qa), len(qb)),))
    dot = np.sum(qa * qb, axis=1)
    qb[dot < 0] *= -1
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    theta = np.arccos(dot)
    small = theta < 1e-4
    sin_t = np.where(small, 1.0, np.sin(theta))
    wa = np.where(small, 1 - u, np.sin((1 - u) * theta) / sin_t)
    wb = np.where(small, u, np.sin(u * theta) / sin_t)
    q = wa[:, None] * qa + wb[:, None] * qb
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def continuous_quats(q: np.ndarray) -> np.ndarray:
    """Flip signs so consecutive quaternions stay in one hemisphere (the browser lerps them)."""
    q = np.array(q, float, copy=True)
    for k in range(1, len(q)):
        if np.dot(q[k], q[k - 1]) < 0:
            q[k] = -q[k]
    return q


@dataclass
class Ladder:
    """Straight rung line: rung k sits at base + dir * (0.3 k / 4) + up * 0.3 k."""

    base: np.ndarray        # rung line at the base height (ground under the ladder)
    yaw: float              # climbing direction
    rungs: int              # rungs climbed
    t_up: tuple             # (t0, t1) of the climb up
    t_down: tuple | None    # (t0, t1) of the climb down, if any
    kind: str = "ladder"    # ladder | stepladder | steps | vehicle

    def rung(self, k: float) -> np.ndarray:
        d = fwd(self.yaw)
        return self.base + d * (0.3 * k / 4.0) + UP * (0.3 * k)


@dataclass
class Walkable:
    centre: np.ndarray   # x, z centre
    y: float             # top surface height
    half: tuple          # half sizes along (right, fwd) of yaw
    yaw: float

    def contains(self, x: float, z: float, margin: float = 0.0) -> bool:
        dx, dz = x - self.centre[0], z - self.centre[2]
        r, f = right(self.yaw), fwd(self.yaw)
        a = dx * r[0] + dz * r[2]
        b = dx * f[0] + dz * f[2]
        return abs(a) <= self.half[0] + margin and abs(b) <= self.half[1] + margin


@dataclass
class Scene:
    ground_y: float = 0.0
    surface: str = "concrete"
    condition: str = "dry"
    props: list = field(default_factory=list)
    walkables: list = field(default_factory=list)
    ladders: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    focus: np.ndarray = field(default_factory=lambda: np.zeros(3))
    keepout: list = field(default_factory=list)   # (x, z, radius) that decor must avoid
    dyn: dict = field(default_factory=dict)       # name -> static info for dynamic props
    rail: dict | None = None                      # NF-RAIL grab rail, used by the arm pose

    # --- prop helpers (all coordinates in world metres) ----------------------------------
    def box(self, centre, size, yaw=0.0, mat="concrete", walk=False, **extra) -> None:
        self.props.append({"t": "box", "p": r3(centre), "s": r3(size), "yaw": round(float(yaw), 4), "m": mat, **extra})
        if walk:
            top = float(centre[1]) + float(size[1]) / 2
            self.walkables.append(Walkable(np.asarray(centre, float), top, (size[0] / 2, size[2] / 2), float(yaw)))

    def slab(self, centre_xz, top_y, size_rf, yaw, thickness, mat, walk=True, **extra) -> None:
        """Box given by its top surface: size along (right, fwd) of yaw."""
        c = np.array([centre_xz[0], top_y - thickness / 2, centre_xz[2]])
        self.box(c, (size_rf[0], thickness, size_rf[1]), yaw, mat, walk, **extra)

    def cyl(self, base, radius, height, mat, top_radius=None, **extra) -> None:
        self.props.append({"t": "cyl", "p": r3(base), "r": round(radius, 3), "rt": round(top_radius if top_radius is not None else radius, 3),
                           "h": round(height, 3), "m": mat, **extra})

    def tube(self, a, b, radius, mat="steel") -> None:
        self.props.append({"t": "tube", "a": r3(a), "b": r3(b), "r": round(radius, 3), "m": mat})

    def decor(self, kind, pos, yaw=0.0, **extra) -> None:
        self.props.append({"t": "decor", "k": kind, "p": r3(pos), "yaw": round(float(yaw), 3), **extra})

    def keep(self, point, radius) -> None:
        self.keepout.append((float(point[0]), float(point[2]), float(radius)))

    def support(self, x: float, z: float, below: float, margin: float = 0.0) -> float:
        """Top of the highest walkable surface under (x, z) that is not above `below`."""
        best = self.ground_y
        for w in self.walkables:
            if w.y <= below + 1e-6 and w.y > best and w.contains(x, z, margin):
                best = w.y
        return best


# --- run context -------------------------------------------------------------------------
class Ctx:
    """Read-only view of one run for the scene builders."""

    def __init__(self, run, rec, kin):
        self.run, self.rec, self.kin = run, rec, kin
        self.track = run.track
        self.fs = run.track.fs
        self.n = len(run.track)
        self.events = run.events
        self.params = run.params
        self.worker = run.worker
        self.H = run.worker.height_m
        self.rng = np.random.default_rng(run.seed + 4242)

    def idx(self, t: float) -> int:
        return int(np.clip(round(t * self.fs), 0, self.n - 1))

    def pelvis(self, t: float) -> np.ndarray:
        return self.track.root_pos[self.idx(t)].copy()

    def heading(self, t: float) -> float:
        return float(self.track.heading[self.idx(t)])

    def ev(self, name: str, nth: int = 0, after: float = -1.0):
        hits = [(t, info) for t, n, info in self.events if n == name and t >= after]
        return hits[nth] if len(hits) > nth else (None, None)

    def t_ev(self, name: str, nth: int = 0, after: float = -1.0):
        return self.ev(name, nth, after)[0]

    def first_after(self, names, after: float):
        times = [t for t, n, _ in self.events if n in names and t > after]
        return min(times) if times else None

    def path(self, t0: float, t1: float, step: int = 50) -> np.ndarray:
        a, b = self.idx(t0), self.idx(t1)
        return self.track.root_pos[a:max(b, a + 1):step]

    def spine_point(self, t: float, slot: int) -> np.ndarray:
        """Spine-line point of a slot (anchor minus the skin offset)."""
        i = self.idx(t)
        offset = self.kin.seg_rot[slot][i].apply([0.0, 0.0, self.worker.sensor_offset_m])
        return self.kin.anchors[i, slot] - offset

    def dring_path(self, idx) -> np.ndarray:
        """Dorsal D-ring of the harness at 1 kHz sample indices: between the shoulder blades, on
        the vest (rig.dring without the render-only body shift)."""
        P, M = spine_pose(self, idx)
        return P[:, 3] + M[:, 3] @ np.array([0.0, -0.02 * self.H, 0.09])

    def com_path(self, idx) -> np.ndarray:
        """Simulated centre of mass at 1 kHz sample indices (falls.Faller: pelvis - R_root r_p,
        r_p = (0, -0.03 H, 0.10))."""
        tr = self.track
        R = root_rotation(tr.heading[idx], tr.pitch[idx], tr.roll[idx])
        return tr.root_pos[idx] - R.apply(np.array([0.0, -0.03 * self.H, 0.10]))

    def begin_height(self) -> float:
        heights = [g for t, g in self.rec.ground_log if t == 0.0]
        return heights[-1] if heights else 0.0

    def fall_direction(self, t_onset: float, t_end: float | None) -> np.ndarray:
        """Horizontal fall direction, snapped to forward/back/left/right of the heading (the
        simulator's DIRECTIONS)."""
        h = self.heading(t_onset)
        t_end = t_end if t_end is not None else t_onset + 1.0
        disp = flat(self.pelvis(t_end) - self.pelvis(t_onset))
        options = [fwd(h), -fwd(h), right(h), -right(h)]
        return options[int(np.argmax([float(disp @ o) for o in options]))]


# --- shared builders ---------------------------------------------------------------------
def scaffold_frame(sc: Scene, centre, yaw, size_rf, height, open_side: str | None, base_y=0.0) -> None:
    """Tube-and-plank scaffold under a deck: posts, ledgers, braces, and guardrails on every
    side except `open_side` ('front' = the +fwd edge)."""
    r, f = right(yaw), fwd(yaw)
    hr, hf = size_rf[0] / 2, size_rf[1] / 2
    nr = max(int(np.ceil(size_rf[0] / 2.1)), 1)
    nf = max(int(np.ceil(size_rf[1] / 2.1)), 1)
    pts = []
    for i in range(nr + 1):
        for j in range(nf + 1):
            if 0 < i < nr and 0 < j < nf:
                continue
            a = -hr + 2 * hr * i / nr
            b = -hf + 2 * hf * j / nf
            pts.append((a, b))
    levels = np.arange(base_y + 2.0, height - 0.3, 2.0)
    for a, b in pts:
        foot = centre + r * a + f * b
        top = height + (1.1 if not _on_side(a, b, hr, hf, open_side) else 0.0)
        sc.tube(np.array([foot[0], base_y, foot[2]]), np.array([foot[0], top, foot[2]]), 0.024, "scaffold")
        sc.cyl(np.array([foot[0], base_y, foot[2]]), 0.09, 0.02, "steel")
    # ledgers around the perimeter at each level (none below the open edge: the fall passes there)
    corners = [(-hr, -hf), (hr, -hf), (hr, hf), (-hr, hf)]
    sides = ("back", "right", "front", "left")
    for y in list(levels) + [height - 0.06]:
        for k in range(4):
            if sides[k] == open_side and y < height - 0.1:
                continue
            a0, b0 = corners[k]
            a1, b1 = corners[(k + 1) % 4]
            p0 = centre + r * a0 + f * b0
            p1 = centre + r * a1 + f * b1
            sc.tube(np.array([p0[0], y, p0[2]]), np.array([p1[0], y, p1[2]]), 0.022, "scaffold")
    # diagonal braces on the two long sides
    for a in (-hr, hr):
        p0 = centre + r * a - f * hf
        p1 = centre + r * a + f * hf
        sc.tube(np.array([p0[0], base_y + 0.2, p0[2]]), np.array([p1[0], height - 0.1, p1[2]]), 0.018, "scaffold")
    # guardrails and toe boards
    for side in ("back", "right", "front", "left"):
        if side == open_side:
            continue
        a0, b0 = {"back": (-hr, -hf), "right": (hr, -hf), "front": (hr, hf), "left": (-hr, hf)}[side]
        a1, b1 = {"back": (hr, -hf), "right": (hr, hf), "front": (-hr, hf), "left": (-hr, -hf)}[side]
        p0 = centre + r * a0 + f * b0
        p1 = centre + r * a1 + f * b1
        for y, rad in ((height + 1.07, 0.024), (height + 0.53, 0.02)):
            sc.tube(np.array([p0[0], y, p0[2]]), np.array([p1[0], y, p1[2]]), rad, "rail")
        mid = (p0 + p1) / 2
        length = float(np.linalg.norm(p1 - p0))
        sc.box(np.array([mid[0], height + 0.075, mid[2]]), (0.03, 0.15, length), yaw_of(unit(p1 - p0)), "plank")


def _on_side(a, b, hr, hf, side) -> bool:
    if side is None:
        return False
    return {"front": abs(b - hf) < 1e-6, "back": abs(b + hf) < 1e-6, "right": abs(a - hr) < 1e-6,
            "left": abs(a + hr) < 1e-6}[side]


# Standing feet relative to the pelvis in the rig: toes 0.29 m ahead, heels 0.02 m ahead, outer
# edges 0.15 m to the side; kneeling, the knees reach 0.28 m ahead and the toes 0.2 m behind.
STANCE_FEET = (0.30, 0.02, 0.15)       # (front, back, side) along fwd / right of the heading
KNEEL_LEGS = (0.30, -0.22, 0.15)


def foot_reach(d: np.ndarray, heading: float, legs: tuple = STANCE_FEET) -> float:
    """How far the feet (or kneeling legs) reach beyond the pelvis along horizontal direction d (m)."""
    f, r = fwd(heading), right(heading)
    front, back, side = legs
    return max(float((f * a + r * b) @ d) for a in (back, front) for b in (-side, side))


def fall_footprint(ctx: Ctx, t0: float, t1: float, d: np.ndarray, e: np.ndarray, y_lo: float, y_hi: float):
    """(s_min, s_max, q_min, q_max) along d and e of the drawn torso while it is between y_lo and
    y_hi (m), or None if it never is."""
    idx = np.arange(ctx.idx(t0), ctx.idx(t1) + 1, 10)
    pts = torso_points(ctx, idx).reshape(-1, 3)
    pts = pts[(pts[:, 1] >= y_lo) & (pts[:, 1] <= y_hi)]
    if not len(pts):
        return None
    s, q = pts @ d, pts @ e
    return float(s.min()), float(s.max()), float(q.min()), float(q.max())


def slabs_around(sc: Scene, d, e, outer: tuple, hole: tuple, top_y: float, thickness: float, mat: str) -> None:
    """Walkable slabs covering the rectangle outer = (s0, s1, q0, q1) except hole = (s0, s1, q0, q1)."""
    s0, s1, q0, q1 = outer
    h0, h1, g0, g1 = hole
    yaw = yaw_of(d)
    for a0, a1, b0, b1 in ((s0, h0, q0, q1), (h1, s1, q0, q1), (h0, h1, q0, g0), (h0, h1, g1, q1)):
        if a1 - a0 > 0.02 and b1 - b0 > 0.02:
            c = d * (a0 + a1) / 2 + e * (b0 + b1) / 2
            sc.slab(np.array([c[0], 0.0, c[2]]), top_y, (b1 - b0, a1 - a0), yaw, thickness, mat)


def edge_platform(sc: Scene, ctx: Ctx, height: float, protection: str = "none") -> dict:
    """Elevated deck whose open edge is just beyond the stance feet where the fall starts.

    When the lead-in walk passes beyond that edge (the simulator draws the walk and the fall
    direction independently, e.g. walking forward and then falling backward), the deck covers the
    walk and the fall goes through an opening sized from the falling body; the walk then crosses
    the opening (noted). Returns geometry info; 'hole' is (s0, s1, q0, q1) or None."""
    t_on = ctx.t_ev("fall_onset")
    t_end = ctx.first_after(("impact", "arrest_start"), t_on)
    d = ctx.fall_direction(t_on, t_end)
    e = perp(d)
    yaw = yaw_of(d)
    p_on = ctx.pelvis(t_on)
    s_edge = float(p_on @ d) + foot_reach(d, ctx.heading(t_on)) + 0.03
    pre = ctx.path(0.0, t_on)
    s = pre @ d
    q = pre @ e
    q_lo, q_hi = float(q.min()) - 1.6, float(q.max()) + 1.6
    if q_hi - q_lo < 3.6:
        mid = (q_hi + q_lo) / 2
        q_lo, q_hi = mid - 1.8, mid + 1.8
    hole = None
    if float(s.max()) <= s_edge:
        s_min = min(float(s.min()) - 1.6, s_edge - 3.2)
        s_max = s_edge
        sc.slab(d * (s_min + s_edge) / 2 + e * (q_lo + q_hi) / 2, height, (q_hi - q_lo, s_edge - s_min), yaw, 0.06, "plank")
        open_side = "front"
    else:
        # Opening: wherever the falling torso passes through the deck's level (down to 2.5 m below,
        # for the net's rebound), plus room for the limbs.
        t_stop = (t_end if t_end is not None else t_on + 2.0) + 3.0
        fp = fall_footprint(ctx, t_on, t_stop, d, e, height - 2.5, height + 0.5) or (s_edge, s_edge + 1.0, -0.6, 0.6)
        s_far = max(fp[1] + 0.25, s_edge + 1.0)
        qa, qb = fp[2] - 0.3, fp[3] + 0.3
        if qb - qa < 1.2:
            mid = (qa + qb) / 2
            qa, qb = mid - 0.6, mid + 0.6
        s_min, s_max = min(float(s.min()), s_edge) - 1.6, max(float(s.max()), s_far) + 1.6
        q_lo, q_hi = min(q_lo, qa - 0.8), max(q_hi, qb + 0.8)
        hole = (s_edge, s_far, qa, qb)
        slabs_around(sc, d, e, (s_min, s_max, q_lo, q_hi), hole, height, 0.06, "plank")
        for b in (qa - 0.05, qb + 0.05):    # paint marks along the opening
            c = d * (s_edge + s_far) / 2 + e * b
            sc.box(np.array([c[0], height + 0.003, c[2]]), (0.06, 0.006, s_far - s_edge), yaw, "hazard")
        sc.notes.append("This seed's lead-in walk passes over the spot where the worker later falls (the simulator "
                        "draws the walk and the fall direction independently). The deck is drawn with an opening at "
                        "the fall; the walk crosses it.")
        open_side = None
    centre = d * (s_min + s_max) / 2 + e * (q_lo + q_hi) / 2
    centre[1] = 0.0
    size_rf = (q_hi - q_lo, s_max - s_min)
    scaffold_frame(sc, centre, yaw, size_rf, height, open_side=open_side, base_y=sc.ground_y)
    sc.focus = p_on
    return {"d": d, "e": e, "s_edge": s_edge, "centre": centre, "yaw": yaw, "size": size_rf, "t_on": t_on, "hole": hole}


def slip_patch(sc: Scene, ctx: Ctx, t: float, condition: str) -> None:
    kind = {"wet": "water", "icy": "ice", "oily": "oil", "muddy": "mud", "dusty": "dust"}.get(condition)
    if kind is None:
        return
    p = ctx.pelvis(t) + fwd(ctx.heading(t)) * 0.15
    p[1] = sc.support(p[0], p[2], p[1])
    sc.props.append({"t": "patch", "p": r3(p), "r": 1.1, "k": kind})


def trip_hazard(sc: Scene, ctx: Ctx, t: float) -> None:
    h = ctx.heading(t)
    p = ctx.pelvis(t) + fwd(h) * 0.32
    p[1] = sc.support(p[0], p[2], p[1]) + 0.025
    sc.box(p, (1.3, 0.05, 0.09), h, "lumber")
    sc.box(p + right(h) * 0.9 + fwd(h) * 0.25, (0.8, 0.04, 0.09), h + 0.6, "lumber")


# Crane load (STRUCK-*): a steel beam lifted off-centre from a timber crib by a mobile crane.
# When it leaves the crib it swings in on its rope and hits the worker at the bottom of the arc.
CRANE_ROPE_M = 10.0          # boom-tip sheave to the beam's centre of mass (the pendulum length)
BEAM_LEN_M = 2.6
BEAM_HALF_DEPTH_M = 0.075    # half flange width, along the swing (world.js beamMesh)
BEAM_HALF_H_M = 0.137        # half section height
BEAM_KG = 114.0              # 2.6 m of a 44 kg/m wide-flange section (as drawn)
HOOK_TOP_M = 1.6             # beam centre to the top of the hook block, where the rope starts
CRIB_DEPTH_M = 0.9           # crib extent along the swing, behind the resting beam
CRIB_OVERHANG_M = 0.035      # beam centre ahead of the crib's front edge at rest (rope keeps it up)
BOOM_REACH_M, BOOM_FOOT_Y = 7.5, 2.8     # boom foot: horizontal distance from the tip, height
AIR_DAMPING_S = 60.0         # swing decay time after the strike


def struck_pulse(ctx: Ctx, t_s: float, dvec: np.ndarray, jolt_g: float) -> tuple[float, float]:
    """Width (s) and worker velocity change (m/s) of the simulator's sin^2 jolt along dvec,
    measured on the 1 kHz pelvis track (the pulse peaks at half its width, 25-75 ms)."""
    i = ctx.idx(t_s)
    x = ctx.track.root_pos[i:i + int(0.25 * ctx.fs)] @ dvec
    accel = np.gradient(np.gradient(x)) * ctx.fs ** 2
    k_peak = int(np.argmax(accel[: int(0.1 * ctx.fs)]))
    width = max(2.0 * k_peak / ctx.fs, 0.02)
    return width, jolt_g * GRAVITY * width / 2.0


def crane_boom_direction(ctx: Ctx, chest: np.ndarray, dvec: np.ndarray, reach: float) -> np.ndarray:
    """Horizontal direction from the boom tip to the crane, chosen clear of the worker's path."""
    e = perp(dvec)
    options = [e, -e, -dvec, unit(-dvec + e), unit(-dvec - e)]
    path = ctx.track.root_pos[::100][:, [0, 2]]
    crib = flat(chest) - dvec * reach
    best, best_gap = options[0], -np.inf
    for b in options:
        base = flat(chest) + b * (BOOM_REACH_M + 2.5)       # carrier centre, about 4.5 m in radius
        gap = min(float(np.min(np.hypot(path[:, 0] - base[0], path[:, 1] - base[2]))),
                  float(np.hypot(*(crib - base)[[0, 2]]))) - 4.5
        if gap > best_gap + 1e-6:
            best, best_gap = b, gap
    return best


def crane_load(sc: Scene, ctx: Ctx) -> None:
    """Beam that delivers the 'struck' jolt; its swing is integrated in crane_motion().

    Momentum: the simulator's jolt is a symmetric sin^2 force pulse, so the contact gives back
    what it stores (elastic exchange). A beam of mass m_b that gives the worker (mass m_w) the
    measured velocity change dv arrives at v0 = (1 + m_w/m_b) dv / 2 and keeps
    (1 - m_w/m_b) dv / 2. Released from rest, it must start theta0 = acos(1 - v0^2 / 2gL) off plumb."""
    t_s, info = ctx.ev("struck")
    h = ctx.heading(t_s)
    alpha = {"left": -np.pi / 2, "right": np.pi / 2, "forward": 0.0}[info.get("direction", "forward")]
    dvec = np.cos(alpha) * fwd(h) + np.sin(alpha) * right(h)
    chest = ctx.spine_point(t_s, 2)
    jolt_g = float(info.get("jolt_g", 2.0))
    width, dv_worker = struck_pulse(ctx, t_s, dvec, jolt_g)
    m_worker = float(ctx.worker.mass_kg + ctx.worker.load_kg)
    v0 = 0.5 * (1.0 + m_worker / BEAM_KG) * dv_worker
    theta0 = float(np.arccos(1.0 - v0 ** 2 / (2 * GRAVITY * CRANE_ROPE_M)))
    reach = CRANE_ROPE_M * np.sin(theta0)                    # crib to strike, horizontally
    boom_dir = crane_boom_direction(ctx, chest, dvec, reach)
    sc.dyn["crane"] = {"t_s": t_s, "dvec": dvec, "chest": chest, "jolt_g": jolt_g, "width": width,
                       "dv_worker": dv_worker, "m_worker": m_worker, "v0": v0, "theta0": theta0,
                       "boom_dir": boom_dir}
    sc.keep(flat(chest) - dvec * (reach + 0.5), 1.4)                   # crib
    sc.keep(flat(chest) + boom_dir * (BOOM_REACH_M + 2.5), 5.0)        # crane carrier
    sc.props.append({"t": "dyn", "k": "crane_load", "yaw": round(yaw_of(dvec), 4)})  # beam long axis across the swing
    sc.focus = chest


def overhead_portal(sc: Scene, centre, yaw, bottom_y: float, span: float = 4.4, mat="steel") -> None:
    """Beam overhead across the walking line, on two posts clear of it."""
    r = right(yaw)
    sc.box(np.array([centre[0], bottom_y + 0.15, centre[2]]), (span, 0.3, 0.2), yaw, mat)
    for side in (-1, 1):
        p = centre + r * side * span / 2
        sc.box(np.array([p[0], sc.ground_y + bottom_y / 2, p[2]]), (0.2, bottom_y, 0.2), yaw, mat)


def ladder_from_call(ctx: Ctx, call, kind: str) -> Ladder:
    h = call.heading0
    base = flat(call.pos0) + fwd(h) * 0.12
    base[1] = call.ground0
    rungs = max(int(round(call.args["height"] / 0.3)), 1)
    return Ladder(base, h, rungs, (call.t0, call.t1), None, kind)


def ladder_down_from_call(call, kind: str, t_up: tuple) -> Ladder:
    """Rung line for a climb that starts at the top (motions.ladder up=False moves the pelvis
    0.075 m back and 0.3 m down per rung). t_up is borrowed from the matching climb up so the rig
    finds this ladder only by its t_down."""
    h = call.heading0
    rungs = max(int(round(call.args["height"] / 0.3)), 1)
    base = flat(call.pos0) + fwd(h) * (0.12 - 0.075 * rungs)
    base[1] = call.ground0 - 0.3 * rungs
    return Ladder(base, h, rungs, t_up, (call.t0, call.t1), kind)


# --- per-scenario scenes -----------------------------------------------------------------
def scene_ladder_fall(sc: Scene, ctx: Ctx) -> None:
    call = [c for c in ctx.rec.find("ladder") if c.args.get("up", True)][0]
    lad = ladder_from_call(ctx, call, "ladder")
    sc.ladders.append(lad)
    top = lad.base[1] + lad.rungs * 0.3
    rail_top = top + 1.1
    d = fwd(lad.yaw)
    lean = lad.base + d * ((rail_top - lad.base[1]) / 4.0)
    sc.props.append({"t": "ladder", "base": r3(lad.base), "yaw": round(lad.yaw, 4), "rungs": int(round(rail_top / 0.3)),
                     "top_y": round(rail_top, 3), "w": 0.46, "m": "aluminium"})
    wall_c = flat(lean) + d * 0.2
    sc.box(np.array([wall_c[0], (rail_top + 0.8) / 2, wall_c[2]]), (6.0, rail_top + 0.8, 0.3), lad.yaw, "brick")
    sc.focus = lad.rung(lad.rungs)


def scene_scaffold(sc: Scene, ctx: Ctx) -> None:
    edge_platform(sc, ctx, ctx.begin_height())


def scene_roof(sc: Scene, ctx: Ctx) -> None:
    """Flat roof under the whole pre-fall stance (standing, then kneeling), then the pitched part:
    a plane at the simulated slope angle laid under the lowest point of the drawn torso while it
    slides (the slide starts on the back). If that plane passes below the flat roof's level at
    the slope start there is a short step down."""
    height = ctx.begin_height()
    t_on = ctx.t_ev("fall_onset")
    t_ff = ctx.t_ev("free_fall_start", after=t_on)
    t_slide = ctx.t_ev("slide_start", after=t_on - 1e-6)
    slope = np.deg2rad(ctx.ev("slide_start")[1].get("slope_deg", 30.0))
    tan_a = float(np.tan(slope))
    d = ctx.fall_direction(t_on, t_ff)
    e = perp(d)
    p_on, p_ff = ctx.pelvis(t_on), ctx.pelvis(t_ff)
    s_flat = float(p_on @ d) + foot_reach(d, ctx.heading(t_on), KNEEL_LEGS) + 0.03
    idx = np.arange(ctx.idx(t_slide), ctx.idx(t_ff) + 1, 5)
    pts = torso_points(ctx, idx).reshape(-1, 3)
    over = pts[pts @ d > s_flat]
    plane = float(np.min(over[:, 1] + (over @ d - s_flat) * tan_a)) if len(over) else height
    if plane >= height:            # the plane meets the roof level further down the slope
        s0, y0 = s_flat + (plane - height) / tan_a, height
    else:                          # a step down to the pitched part
        s0, y0 = s_flat, plane
    s_eave = max(float(p_ff @ d) + 0.35, s0 + 0.3)
    y_eave = max(y0 - (s_eave - s0) * tan_a, 0.8)
    pre = ctx.path(0.0, t_on)
    s_back = min(float((pre @ d).min()) - 2.5, s0 - 3.0)
    q = pre @ e
    q_lo, q_hi = float(q.min()) - 3.5, float(q.max()) + 3.5
    yaw = yaw_of(d)
    mid_q = (q_lo + q_hi) / 2
    # walls up to the eave, then a flat roof deck to the slope start
    c_walls = d * (s_back + s_eave) / 2 + e * mid_q
    sc.box(np.array([c_walls[0], y_eave / 2, c_walls[2]]), (q_hi - q_lo, y_eave, s_eave - s_back - 0.3), yaw, "wall")
    c_flat = d * (s_back + s0) / 2 + e * mid_q
    sc.slab(np.array([c_flat[0], 0, c_flat[2]]), height, (q_hi - q_lo, s0 - s_back), yaw, height - y_eave, "roofdeck")
    top_edge = d * s0 + e * mid_q
    top_edge[1] = y0
    sc.props.append({"t": "wedge", "p": r3(top_edge), "yaw": round(yaw, 4), "slope": round(float(np.rad2deg(slope)), 2),
                     "len": round(s_eave - s0, 3), "w": round(q_hi - q_lo, 3), "base": round(y_eave, 3), "m": "roof"})
    gutter = d * (s_eave + 0.05) + e * mid_q
    sc.tube(np.array([gutter[0] - e[0] * (q_hi - q_lo) / 2, y_eave, gutter[2] - e[2] * (q_hi - q_lo) / 2]),
            np.array([gutter[0] + e[0] * (q_hi - q_lo) / 2, y_eave, gutter[2] + e[2] * (q_hi - q_lo) / 2]), 0.06, "white")
    sc.focus = p_on


def scene_hole(sc: Scene, ctx: Ctx) -> None:
    height = ctx.begin_height()
    t_on = ctx.t_ev("fall_onset")
    h = ctx.heading(t_on)
    f, r = fwd(h), right(h)
    p = ctx.pelvis(t_on)
    s_on, q_on = float(p @ f), float(p @ r)
    pre = ctx.path(0.0, t_on)
    s_lo = min(float((pre @ f).min()) - 3.0, s_on - 4.0)
    s_hi = s_on + 5.0
    q_lo, q_hi = q_on - 4.0, q_on + 4.0
    hole = (s_on - 0.3, s_on + 1.5, q_on - 0.65, q_on + 0.65)
    yaw = h
    pieces = [(s_lo, hole[0], q_lo, q_hi), (hole[1], s_hi, q_lo, q_hi), (hole[0], hole[1], q_lo, hole[2]), (hole[0], hole[1], hole[3], q_hi)]
    for a0, a1, b0, b1 in pieces:
        c = f * (a0 + a1) / 2 + r * (b0 + b1) / 2
        sc.slab(np.array([c[0], 0, c[2]]), height, (b1 - b0, a1 - a0), yaw, 0.22, "concrete")
    for a in (s_lo + 0.3, s_hi - 0.3):
        for b in (q_lo + 0.3, q_hi - 0.3):
            c = f * a + r * b
            sc.box(np.array([c[0], (height - 0.22) / 2, c[2]]), (0.35, height - 0.22, 0.35), yaw, "concrete")
    cover = f * (hole[1] + 0.9) + r * (q_on + 1.2)
    sc.box(np.array([cover[0], height + 0.01, cover[2]]), (1.4, 0.02, 1.9), yaw + 0.35, "plywood")
    for b in (hole[2] - 0.05, hole[3] + 0.05):  # paint marks around the opening
        c = f * (hole[0] + hole[1]) / 2 + r * b
        sc.box(np.array([c[0], height + 0.003, c[2]]), (0.06, 0.006, hole[1] - hole[0]), yaw, "hazard")
    sc.focus = p


def scene_excavation(sc: Scene, ctx: Ctx) -> None:
    height = ctx.begin_height()
    t_on = ctx.t_ev("fall_onset")
    t_ff = ctx.t_ev("free_fall_start", after=t_on)
    t_imp = ctx.t_ev("impact", after=t_on)
    slope = np.deg2rad(ctx.ev("slide_start")[1].get("slope_deg", 30.0))
    d = ctx.fall_direction(t_on, t_imp)
    e = perp(d)
    p_on = ctx.pelvis(t_on)
    s_edge = float(p_on @ d) + 0.1
    slide = max(float((ctx.pelvis(t_ff) - p_on) @ d), 0.05)
    s_far = max(float(ctx.pelvis(t_imp) @ d) + 0.45, s_edge + slide + 1.0)
    pre = ctx.path(0.0, t_on)
    s_lo = min(float((pre @ d).min()) - 3.0, s_edge - 4.0)
    q = pre @ e
    q_mid = float(p_on @ e)
    length = max(12.0, float(q.max() - q.min()) + 6.0)
    q_c = (float(q.max()) + float(q.min())) / 2
    yaw = yaw_of(d)
    for a0, a1 in ((s_lo, s_edge), (s_far, s_far + 5.0)):
        c = d * (a0 + a1) / 2 + e * q_c
        sc.slab(np.array([c[0], 0, c[2]]), height, (length, a1 - a0), yaw, height, "soil")
    bank = d * s_edge + e * q_c
    bank[1] = height
    sc.props.append({"t": "wedge", "p": r3(bank), "yaw": round(yaw, 4), "slope": round(float(np.rad2deg(slope)), 2),
                     "len": round(slide, 3), "w": round(length, 3), "base": 0.0, "m": "soil"})
    spoil = d * (s_far + 2.5) + e * (q_mid + 2.0)
    sc.decor("spoil", np.array([spoil[0], height, spoil[2]]), yaw)
    sc.surface = "packed_soil"
    sc.focus = p_on


# Stairs (STAIR). The simulator lowers the support by top / impacts after every body contact but
# the body travels only 0-3 m per drop, so a catalogue 0.18 / 0.28 flight cannot hold it. The
# flight is fitted to the drawn torso instead: rise from the simulated levels, every riser as far
# down the flight as the falling torso allows, treads as regular as the descent permits.
STAIR_RISE_M = 0.18
STAIR_TOL_M = 0.05                          # torso may press this far into a tread (soft tissue, impact)
STAIR_RUNS_M = (0.26, 0.22, 0.18, 0.14, 0.10)   # preferred minimum tread, relaxed for steep descents


def stair_risers(levels: np.ndarray, s_pts: np.ndarray, y_pts: np.ndarray, s_walk: float) -> np.ndarray:
    """Riser positions along the flight; riser j separates tread top levels[j] (above) from
    levels[j + 1]. Every torso point lower than levels[j] - STAIR_TOL_M must lie beyond riser j.
    Tries the preferred minimum tread first and the top landing reaching the walk's front foot."""
    n = len(levels) - 1
    limit = np.empty(n)
    for j in range(n):
        below = s_pts[y_pts < levels[j] - STAIR_TOL_M]
        limit[j] = (float(below.min()) if len(below) else float(s_pts.max())) - 0.02
    risers = limit.copy()
    for run in STAIR_RUNS_M:
        risers = limit.copy()
        for j in range(n - 2, -1, -1):
            risers[j] = min(limit[j], risers[j + 1] - run)
        if risers[0] >= s_walk:
            break
    return risers


def scene_stairs(sc: Scene, ctx: Ctx) -> None:
    steps = int(ctx.params.get("steps", 6))
    n_contacts = max(int(ctx.params.get("impacts", 2)), 1)
    top_y = STAIR_RISE_M * steps
    drop = top_y / n_contacts                          # falls.stairs_fall drop_each
    per_drop = max(int(round(drop / STAIR_RISE_M)), 1)
    levels = top_y - (drop / per_drop) * np.arange((n_contacts - 1) * per_drop + 1)   # top landing first
    t_on = ctx.t_ev("fall_onset")
    impacts = [t for t, n, _ in ctx.events if n == "impact"]
    t_last = impacts[-1] if impacts else t_on + 2.0
    d = ctx.fall_direction(t_on, t_last)
    e = perp(d)
    yaw = yaw_of(d)
    p_on = ctx.pelvis(t_on)
    pre = ctx.path(0.0, t_on)
    s_pre, q_pre = pre @ d, pre @ e
    s_walk = float(p_on @ d) + foot_reach(d, ctx.heading(t_on)) + 0.02
    # Drawn torso from the onset to the end of the last contact, and the legs once it rests
    # (independent of the observation length).
    idx = np.arange(ctx.idx(t_on), ctx.idx(t_last + 1.5) + 1, 10)
    rest = np.arange(ctx.idx(t_last + 0.3), ctx.idx(t_last + 1.5) + 1, 10)
    pts = np.vstack([torso_points(ctx, idx).reshape(-1, 3), lying_leg_points(ctx, rest).reshape(-1, 3)])
    s_pts, q_pts, y_pts = pts @ d, pts @ e, pts[:, 1]
    risers = stair_risers(levels, s_pts, y_pts, s_walk) if len(levels) > 1 else np.array([s_walk + 2.0])
    crossing = float(s_pre.max()) > risers[0] - 0.1         # the walk itself passes over the flight
    q_on = float(p_on @ e)
    qa, qb = min(float(q_pts.min()), q_on) - 0.35, max(float(q_pts.max()), q_on) + 0.35
    if qb - qa < 1.2:
        mid = (qa + qb) / 2
        qa, qb = mid - 0.6, mid + 0.6
    # Lower landing at the final level, under the rest of the run.
    post = ctx.track.root_pos[ctx.idx(t_last)::50]
    s_land = max(float((post @ d).max()), float(s_pts.max())) + 1.0
    la, lb = min(qa, float((post @ e).min()) - 0.8), max(qb, float((post @ e).max()) + 0.8)
    s_back = min(float(s_pre.min()) - 1.6, float(risers[0]) - 3.0)
    q_lo, q_hi = min(float(q_pre.min()) - 1.6, qa), max(float(q_pre.max()) + 1.6, qb)
    if not crossing:
        c = d * (s_back + risers[0]) / 2 + e * (q_lo + q_hi) / 2
        sc.slab(np.array([c[0], 0.0, c[2]]), top_y, (q_hi - q_lo, float(risers[0]) - s_back), yaw, top_y, "concrete")
    else:
        # Stairwell: the top floor goes round the opening of the flight and the lower landing.
        outer = (s_back, max(float(s_pre.max()), s_land) + 1.6, min(q_lo, la) - 0.6, max(q_hi, lb) + 0.6)
        slabs_around(sc, d, e, outer, (float(risers[0]), s_land, min(qa, la), max(qb, lb)), top_y, top_y, "concrete")
        sc.notes.append("This seed's lead-in walk crosses the stairwell before the fall (the simulator draws the walk "
                        "and the fall direction independently): the walk passes over the opening.")
    ends = list(risers) + [s_land]
    for j in range(1, len(levels)):
        a0, a1 = ends[j - 1], ends[j]
        b0, b1 = (qa, qb) if j < len(levels) - 1 else (la, lb)
        c = d * (a0 + a1) / 2 + e * (b0 + b1) / 2
        sc.slab(np.array([c[0], 0.0, c[2]]), float(levels[j]), (b1 - b0, a1 - a0), yaw, float(levels[j]), "concrete")
    # Handrails following the nosings, with posts at the first and last riser.
    for q_r in (qa - 0.04, qb + 0.04):
        nose = [d * r_j + e * q_r + UP * (levels[j] + 0.9) for j, r_j in enumerate(risers)]
        nose.append(d * (risers[-1] + 0.3) + e * q_r + UP * (levels[-1] + 0.9))
        for a, b in zip(nose[:-1], nose[1:], strict=True):
            sc.tube(a, b, 0.025, "rail")
        for j in (0, len(risers) - 1):
            p = d * risers[j] + e * q_r
            sc.tube(np.array([p[0], levels[j + 1], p[2]]), np.array([p[0], levels[j] + 0.9, p[2]]), 0.02, "rail")
    # The flight continues from the lower landing to the ground (catalogue 0.18 / 0.28 steps).
    y_end = float(levels[-1])
    for k in range(1, int(np.floor(y_end / STAIR_RISE_M + 1e-6)) + 1):
        y = y_end - STAIR_RISE_M * k
        if y < 0.05:
            break
        c = d * (s_land + 0.28 * (k - 0.5)) + e * (la + lb) / 2
        sc.slab(np.array([c[0], 0.0, c[2]]), y, (lb - la, 0.28), yaw, y, "concrete")
    sc.focus = p_on


LANYARD_EST_M = (1.2, 1.8)    # the simulator's lanyard_m range (falls.height_fall harness_arrest)
PRE_SLACK = 0.85              # before the fall the D-ring stays within this share of the lanyard
                              # (room for it to drape over the shoulder to an anchor in front)
HEADROOM_M = 2.1              # anchor arm and any anchor above the walk: this high over the deck
TAUT_LEAD_S = 0.1             # the drawn lanyard may go taut this much before free_fall_end
ANCHOR_GRID_M = 0.1           # anchor search step (m)


def swing_length(t: np.ndarray, com: np.ndarray, t0: float, t1: float) -> float | None:
    """Pendulum length (m) of the simulated suspension: falls.harness_arrest swings the COM about
    its arrest_end position with omega = sqrt(g / (lanyard_m + decel)); measured from the zero
    crossings of the swing. None when the swing is too small to time."""
    win = (t >= t0) & (t <= t1)
    if win.sum() < 50:
        return None
    xz = com[win][:, [0, 2]] - com[win][0, [0, 2]]
    _, _, vt = np.linalg.svd(xz - xz.mean(axis=0), full_matrices=False)
    sig = xz @ vt[0]
    cross = np.flatnonzero(np.diff(np.sign(sig[1:])) != 0) + 1
    if len(cross) < 3 or np.abs(sig).max() < 0.02:
        return None
    omega = np.pi / float(np.median(np.diff(t[win][cross])))
    return GRAVITY / omega ** 2


def lanyard_plan(ctx: Ctx, deck_y: float) -> dict:
    """Anchor and relaxed length L0 of the lanyard, consistent with the simulated D-ring path.

    The lanyard is an energy-absorbing stretch lanyard: it pays out under load and recoils when
    the load drops (lanyard_motion). After the arrest the simulated trunk rotates about the COM,
    which lifts the D-ring by up to 0.35 m, so a lanyard that went taut exactly at free_fall_end
    under an anchor straight above the hang point would hang slack. The anchor is therefore
    searched on a grid above the mean hanging D-ring, with L0 = the smaller of the distance at
    free_fall_end and the hang distance (5th percentile, less 0.5 %) so that it is loaded while
    the worker hangs. Required: the lanyard is slack before the fall (the D-ring within
    PRE_SLACK L0 in the last second) and in free fall until TAUT_LEAD_S before free_fall_end.
    Among those anchors: the hanging lanyard as close to vertical as possible, with L0 near the
    simulator's lanyard_m (read back from the suspended swing period)."""
    idx = np.arange(0, ctx.n, 20)
    t = idx / ctx.fs
    ring, com = ctx.dring_path(idx), ctx.com_path(idx)
    t_on = ctx.t_ev("fall_onset")
    t_taut = ctx.t_ev("free_fall_end", after=t_on)
    _, ae = ctx.ev("arrest_end", after=t_on)
    t_s = ctx.t_ev("suspended", after=t_on)
    decel = float(ae.get("decel_distance_m", 0.3))
    L_sim = swing_length(t, com, t_s, t_s + 30.0)
    lanyard_est = float(np.clip(L_sim - decel, *LANYARD_EST_M)) if L_sim else float(np.mean(LANYARD_EST_M))
    hang = (t >= t_s + 0.6) & (t <= t_s + 30.0)        # the trunk has settled into its hang tilt
    if hang.sum() < 10:
        hang = t >= t_s
    ring_h = ring[hang][::10]
    p_h = ring_h.mean(axis=0)
    pre = ring[(t >= t_on - 1.0) & (t < t_on)]
    fall = ring[(t >= t_on) & (t < t_taut - TAUT_LEAD_S)]
    taut_at = np.array([np.interp(t_taut, t, ring[:, j]) for j in range(3)])
    offs = np.arange(-1.6, 1.6 + 1e-9, ANCHOR_GRID_M)
    dx, dz = (g.ravel() for g in np.meshgrid(offs, offs))
    walk = ctx.path(0.0, t_on, 100)[:, [0, 2]]
    over_walk = np.hypot(p_h[0] + dx[:, None] - walk[None, :, 0], p_h[2] + dz[:, None] - walk[None, :, 1]).min(axis=1) < 0.6
    best = None
    for ya in np.arange(max(p_h[1] + 0.5, deck_y + 0.5), p_h[1] + 3.5, 0.05):
        A = np.column_stack([p_h[0] + dx, np.full(len(dx), ya), p_h[2] + dz])
        dist = lambda pts: np.linalg.norm(A[:, None] - pts[None], axis=2)  # noqa: E731
        L0 = np.minimum(np.linalg.norm(A - taut_at, axis=1), np.percentile(dist(ring_h), 5, axis=1) / 1.005)
        slack = (dist(pre).max(axis=1) <= PRE_SLACK * L0) & (dist(fall).max(axis=1, initial=0.0) <= L0)
        tilt = np.degrees(np.arctan2(np.hypot(dx, dz), ya - p_h[1]))
        headroom = ~over_walk | (ya >= deck_y + HEADROOM_M)       # the walk does not run into the anchor
        score = np.where(slack, 0.0, 1e6) + np.where(headroom, 0.0, 1e4) + tilt + 10.0 * np.abs(L0 - lanyard_est)
        k = int(np.argmin(score))
        if best is None or score[k] < best[0]:
            best = (float(score[k]), A[k], float(L0[k]), bool(slack[k]), float(tilt[k]))
    _, anchor, L0, slack_ok, tilt = best
    d_all = np.linalg.norm(ring - anchor, axis=1)
    loaded = np.flatnonzero((t >= t_on) & (d_all >= L0))
    # Before the fall: clipped on from the last time the D-ring was out of the lanyard's reach.
    far = np.flatnonzero((t < t_on) & (d_all > PRE_SLACK * L0))
    return {"anchor": anchor, "L0": L0, "t_taut": float(t_taut),
            "t_loaded": float(t[loaded[0]]) if len(loaded) else float(t_taut),
            "t_clip": float(t[far[-1]] + 0.02) if len(far) else None, "lanyard_est": lanyard_est,
            "swing_len": L_sim, "decel": decel, "hang_tilt_deg": tilt, "slack_ok": slack_ok}


def scene_harness(sc: Scene, ctx: Ctx) -> None:
    height = ctx.begin_height()
    info = edge_platform(sc, ctx, height, "harness")
    plan = lanyard_plan(ctx, height)
    anchor = plan["anchor"]
    # Anchor post on the deck behind the edge, with an arm out to the anchor eye; moved sideways
    # off the lead-in walk if it would stand on it.
    d, e = info["d"], info["e"]
    s_post = info["s_edge"] - 0.45
    post = flat(anchor) - d * (float(flat(anchor) @ d) - s_post)
    walk = flat(ctx.path(0.0, info["t_on"]))
    for shift in (0.0, 0.6, -0.6, 1.2, -1.2, 1.8, -1.8):
        if np.linalg.norm(walk - (post + e * shift), axis=1).min(initial=9.0) > 0.45:
            post = post + e * shift
            break
    top = max(anchor[1] + 0.25, height + HEADROOM_M)                   # the arm clears the worker's head
    sc.box(np.array([post[0], height + (top - height + 0.05) / 2, post[2]]), (0.14, top - height + 0.05, 0.14), info["yaw"], "yellow")
    sc.tube(np.array([post[0], top, post[2]]), np.array([anchor[0], top, anchor[2]]), 0.05, "yellow")
    sc.tube(np.array([anchor[0], top, anchor[2]]), anchor + UP * 0.05, 0.012, "steel")       # anchor sling
    sc.box(np.array([post[0], height + 0.02, post[2]]), (0.6, 0.04, 0.6), info["yaw"], "steel")
    sc.dyn["lanyard"] = plan
    sc.props.append({"t": "dyn", "k": "lanyard", "anchor": r3(anchor)})
    if plan["t_clip"] is not None:
        sc.notes.append(f"The lead-in walk takes the D-ring farther from the anchor than the {plan['L0']:.2f} m lanyard "
                        f"reaches, so the lanyard is drawn stowed on the harness until {plan['t_clip']:.1f} s and clipped "
                        "to the anchor after that.")
    if not plan["slack_ok"]:
        sc.notes.append("No anchor position fits the simulated D-ring path exactly (slack until free_fall_end and taut while "
                        "hanging); the closest one is drawn.")


def scene_net(sc: Scene, ctx: Ctx) -> None:
    height = ctx.begin_height()
    sc.ground_y = -2.6
    info = edge_platform(sc, ctx, height, "net")
    t_imp = ctx.t_ev("impact", after=info["t_on"])
    land = ctx.pelvis(t_imp)
    d, e = info["d"], info["e"]
    s0 = info["s_edge"] - 0.4
    s1 = max(float(land @ d) + 2.5, s0 + 4.5)
    q_c = float(land @ e)
    centre = d * (s0 + s1) / 2 + e * q_c
    centre[1] = 0.0
    sc.dyn["net"] = {"y": 0.0, "centre": centre, "yaw": float(info["yaw"]), "w": 7.0, "l": float(s1 - s0)}
    sc.props.append({"t": "net", "p": r3(centre), "w": 7.0, "l": round(s1 - s0, 3), "yaw": round(info["yaw"], 4)})
    for a in (s0, s1):
        for b in (q_c - 3.5, q_c + 3.5):
            p = d * a + e * b
            sc.tube(np.array([p[0], sc.ground_y, p[2]]), np.array([p[0], 0.35, p[2]]), 0.05, "steel")


def scene_slip(sc: Scene, ctx: Ctx) -> None:
    t_on = ctx.t_ev("fall_onset")
    slip_patch(sc, ctx, t_on, ctx.params.get("condition", "wet"))
    sc.focus = ctx.pelvis(t_on)


def scene_trip(sc: Scene, ctx: Ctx) -> None:
    t_on = ctx.t_ev("fall_onset")
    trip_hazard(sc, ctx, t_on)
    sc.focus = ctx.pelvis(t_on)


def scene_stumble(sc: Scene, ctx: Ctx) -> None:
    t_on = ctx.t_ev("near_fall_onset")
    trip_hazard(sc, ctx, t_on)
    sc.focus = ctx.pelvis(t_on)


def scene_rail(sc: Scene, ctx: Ctx) -> None:
    t_on = ctx.t_ev("near_fall_onset")
    h = ctx.heading(t_on)
    f, r = fwd(h), right(h)
    side = 1.0 if ctx.run.seed % 2 == 0 else -1.0
    p = ctx.pelvis(t_on)
    line = p + r * side * 0.5
    a, b = line - f * 3.5, line + f * 4.0
    for y in (1.07, 0.53):
        sc.tube(np.array([a[0], y, a[2]]), np.array([b[0], y, b[2]]), 0.025, "rail")
    for k in range(6):
        q = a + (b - a) * k / 5
        sc.tube(np.array([q[0], 0.0, q[2]]), np.array([q[0], 1.1, q[2]]), 0.025, "rail")
    sc.rail = {"a": a, "b": b, "y": 1.07, "side": side}
    slip_patch(sc, ctx, t_on, "wet")
    sc.focus = p


def scene_struck(sc: Scene, ctx: Ctx) -> None:
    crane_load(sc, ctx)


def scene_slip_friction(sc: Scene, ctx: Ctx) -> None:
    t_on = ctx.t_ev("fall_onset")
    if t_on is None:
        t_on = ctx.t_ev("near_fall_onset")
    slip_patch(sc, ctx, t_on, ctx.params.get("condition", "wet"))
    sc.focus = ctx.pelvis(t_on)


def scene_collapse(sc: Scene, ctx: Ctx) -> None:
    sc.focus = ctx.pelvis(ctx.t_ev("fall_onset"))


def scene_hammer(sc: Scene, ctx: Ctx) -> None:
    call = ctx.rec.find("hammer")[0]
    h = call.heading0
    f, r = fwd(h), right(h)
    spot = flat(call.pos0)
    bottom = 0.82 * ctx.H + 0.68 * ctx.H / 1.75
    formwork_ceiling(sc, spot, h, bottom, ctx)
    post = spot + f * 0.45 + r * 0.32
    sc.box(np.array([post[0], bottom / 2, post[2]]), (0.1, bottom, 0.1), h, "lumber")
    sc.dyn["hammer_post"] = post
    sc.focus = spot + UP * 1.2


def formwork_ceiling(sc: Scene, spot, h, bottom, ctx, half=3.6) -> None:
    f, r = fwd(h), right(h)
    sc.box(np.array([spot[0], bottom + 0.26, spot[2]]), (2 * half, 0.04, 2 * half), h, "plywood")
    for k in np.arange(-half + 0.3, half, 0.6):  # joists across, beams along
        c = spot + f * k
        sc.box(np.array([c[0], bottom + 0.17, c[2]]), (2 * half, 0.14, 0.07), h, "lumber")
    for k in (-half + 0.4, 0.0, half - 0.4):
        c = spot + r * k
        sc.box(np.array([c[0], bottom + 0.05, c[2]]), (0.1, 0.1, 2 * half), h, "lumber")
    for a in np.arange(-half + 0.4, half, 1.8):
        for b in (-half + 0.4, half - 0.4):
            p = spot + f * a + r * b
            sc.box(np.array([p[0], bottom / 2, p[2]]), (0.08, bottom, 0.08), h, "steel")


def scene_drill(sc: Scene, ctx: Ctx) -> None:
    call = ctx.rec.find("drill_overhead")[0]
    h = call.heading0
    spot = flat(call.pos0)
    bottom = 0.82 * ctx.H + 0.8 * ctx.H / 1.75
    f, r = fwd(h), right(h)
    sc.box(np.array([spot[0], bottom + 0.15, spot[2]]), (8.0, 0.3, 8.0), h, "concrete")
    for a in (-3.4, 3.4):
        for b in (-3.4, 3.4):
            p = spot + f * a + r * b
            sc.box(np.array([p[0], bottom / 2, p[2]]), (0.4, bottom, 0.4), h, "concrete")
    sc.dyn["ceiling_y"] = bottom
    sc.focus = spot + UP * 1.2


def scene_jackhammer(sc: Scene, ctx: Ctx) -> None:
    call = ctx.rec.find("jackhammer")[0]
    h = call.heading0
    spot = flat(call.pos0) + fwd(h) * 0.45
    sc.props.append({"t": "patch", "p": r3(spot + UP * 0.002), "r": 0.75, "k": "broken"})
    for _ in range(7):  # rubble ahead of the chisel, clear of the feet
        p = spot + right(h) * ctx.rng.choice([-1, 1]) * ctx.rng.uniform(0.35, 1.2) + fwd(h) * ctx.rng.uniform(0.1, 1.2)
        sc.decor("rubble", p, ctx.rng.uniform(0, 6.28), s=round(float(ctx.rng.uniform(0.08, 0.2)), 3))
    sc.dyn["jack_rest"] = {"p": flat(call.pos0) + right(h) * 0.9 + fwd(h) * 0.2, "yaw": h + 1.2}
    sc.focus = spot


def scene_equipment(sc: Scene, ctx: Ctx) -> None:
    ride = ctx.rec.find("cab_ride")[0]
    climbs = ctx.rec.find("ladder", inside=ride)
    sits = ctx.rec.find("sit", inside=ride)
    stands = ctx.rec.find("stand_up", inside=ride)
    t_seat = sits[0].t1
    t_leave = stands[-1].t0 if stands else ride.t1
    seat_h = float(sits[0].args.get("seat_height", 0.45))
    floor = float(climbs[0].ground1)
    sc.dyn["vehicle"] = {"t_seat": t_seat, "t_leave": t_leave, "seat_h": seat_h, "floor": floor}
    sc.props.append({"t": "dyn", "k": "vehicle", "floor": round(floor, 3), "seat": round(floor + seat_h, 3)})
    up = ladder_from_call(ctx, climbs[0], "vehicle")
    sc.ladders.append(up)
    if len(climbs) > 1:   # climbing down where the loader is parked at the end of the ride
        sc.ladders.append(ladder_down_from_call(climbs[-1], "vehicle", up.t_up))
    sc.surface = "gravel" if sc.surface == "concrete" else sc.surface
    sc.focus = ctx.pelvis(t_seat)


def scene_kneel(sc: Scene, ctx: Ctx) -> None:
    call = (ctx.rec.find("kneel_work") or ctx.rec.find("kneel"))[0]
    h = call.heading0
    f, r = fwd(h), right(h)
    spot = flat(call.pos0)
    for a in np.arange(0.25, 2.6, 0.31):
        for b in np.arange(-0.75, 0.8, 0.31):
            if a > 1.4 and b > 0.2:
                continue
            p = spot + f * a + r * b
            sc.box(np.array([p[0], 0.006, p[2]]), (0.3, 0.012, 0.3), h, "tile")
    sc.decor("toolbox", spot + r * 0.9 - f * 0.2, h)
    sc.decor("bucket", spot + r * 0.8 + f * 0.9, h)
    sc.focus = spot


def scene_rebar(sc: Scene, ctx: Ctx) -> None:
    call = ctx.rec.find("rebar_tie")[0]
    h = call.heading0
    path = ctx.path(call.t0, call.t1)
    f = fwd(h)
    s = path @ f
    centre = flat(call.pos0) + f * ((float(s.max()) - float(s.min())) / 2 + 0.8)
    length = float(s.max() - s.min()) + 6.0
    sc.props.append({"t": "rebar", "p": r3(centre), "w": 5.0, "l": round(length, 3), "yaw": round(h, 4), "sp": 0.2})
    sc.decor("rebar_bundle", centre + right(h) * 3.2, h)
    sc.focus = flat(call.pos0)


def scene_ladder_work(sc: Scene, ctx: Ctx) -> None:
    calls = [c for c in ctx.rec.find("ladder") if c.depth == 0]
    ups = [c for c in calls if c.args.get("up", True)]
    downs = [c for c in calls if not c.args.get("up", True)]
    for k, up in enumerate(ups):
        lad = ladder_from_call(ctx, up, "stepladder")
        if k < len(downs):
            lad.t_down = (downs[k].t0, downs[k].t1)
        sc.ladders.append(lad)
        top = lad.base[1] + lad.rungs * 0.3
        beam = flat(up.pos1) + fwd(up.heading0) * 0.1
        overhead_portal(sc, beam, up.heading0, top + 0.82 * ctx.H + 0.66 * ctx.H / 1.75)
    sc.dyn["stepladder"] = True
    sc.props.append({"t": "dyn", "k": "stepladder", "h": round(max(l.rungs for l in sc.ladders) * 0.3 + 0.7, 3)})
    sc.focus = sc.ladders[0].rung(sc.ladders[0].rungs) if sc.ladders else np.zeros(3)


PLANK_M = 0.06            # platform deck thickness


def scene_jump(sc: Scene, ctx: Ctx) -> None:
    """Step-up platform per climb: the simulated step-up is a 1:4 ladder climb (0.075 m forward
    per 0.3 m up), so the deck is a thin plank on legs that starts at the heels of the stance on
    top and ends just past the toes at the take-off; a short ladder for two-rung climbs. The feet
    on the ground before the climb stand under the deck's back edge."""
    for call in [c for c in ctx.rec.find("ladder") if c.depth == 0]:
        lad = ladder_from_call(ctx, call, "steps")
        sc.ladders.append(lad)
        f, r = fwd(lad.yaw), right(lad.yaw)
        top = float(lad.base[1] + 0.3 * lad.rungs)
        jump = next((x for x in ctx.rec.find("jump_down") if x.t0 >= call.t1 - 1e-6), None)
        s_stand = [float(flat(call.pos1) @ f)] + ([float(flat(jump.pos0) @ f)] if jump else [])
        rear = min(s_stand) + STANCE_FEET[1] - 0.04
        front = max(s_stand) + STANCE_FEET[0] + 0.04
        if jump is not None:   # the jump lands only ~0.4 m ahead: the landed heels stay off the deck
            front = max(min(front, float(flat(jump.pos1) @ f) - 0.14), max(s_stand) + 0.12)
        q_c = float(flat(call.pos1) @ r)
        width = 1.6
        c = f * (rear + front) / 2 + r * q_c
        sc.slab(np.array([c[0], 0.0, c[2]]), top, (width, front - rear), lad.yaw, PLANK_M, "plank")
        for a in (rear + 0.05, front - 0.05):
            for b in (-width / 2 + 0.05, width / 2 - 0.05):
                p = f * a + r * (q_c + b)
                sc.box(np.array([p[0], lad.base[1] + (top - PLANK_M) / 2, p[2]]), (0.07, top - PLANK_M, 0.07), lad.yaw, "steel")
        if lad.rungs >= 2:   # rungs below the deck, on the climb's rung line
            sc.props.append({"t": "ladder", "base": r3(lad.base), "yaw": round(lad.yaw, 4), "rungs": lad.rungs - 1,
                             "top_y": round(top - PLANK_M, 3), "w": 0.46, "m": "aluminium"})
        sc.focus = flat(call.pos1)


BUCKET_TOP_R = 0.135      # upturned 20 l bucket used as a seat: top radius 0.135, base 0.15 (m)


def scene_sit(sc: Scene, ctx: Ctx) -> None:
    """Upturned bucket per sit: its top at the drawn buttocks (lowest torso point while seated),
    its front rim just behind the heels of the standing stance, buttocks over the top."""
    stands = ctx.rec.find("stand_up")
    for call in ctx.rec.find("sit"):
        if "cab_ride" in ctx.rec.ancestors(call):
            continue
        f, r = fwd(call.heading0), right(call.heading0)
        t_up = next((x.t0 for x in stands if x.t0 >= call.t1 - 1e-6), call.t1 + 2.0)
        idx = np.arange(ctx.idx(call.t1 + 0.3), ctx.idx(max(t_up, call.t1 + 0.5)) + 1, 20)
        pts = torso_points(ctx, idx)
        low = pts[np.arange(len(pts)), np.argmin(pts[..., 1], axis=1)]
        seat_top = float(low[:, 1].min())
        buttock = low.mean(axis=0)
        pelvis = flat(call.pos1)
        centre_s = max(float(pelvis @ f) + STANCE_FEET[1] - 0.02 - BUCKET_TOP_R,
                       float(buttock @ f) - (BUCKET_TOP_R - 0.015))
        p = f * centre_s + r * float(buttock @ r)
        p[1] = call.ground0
        sc.cyl(p, 0.15, max(seat_top - call.ground0, 0.15), "bucket", top_radius=BUCKET_TOP_R)
        sc.focus = p


def scene_lie(sc: Scene, ctx: Ctx, mat="cardboard") -> None:
    call = ctx.rec.find("lie")[0]
    t = call.t1
    p0, p3 = ctx.spine_point(t, 0), ctx.spine_point(t, 3)
    axis = flat(p3 - p0)
    if np.linalg.norm(axis) < 0.1:
        axis = fwd(call.heading0)
    axis = unit(axis)
    centre = flat(p0) + axis * 0.05
    sc.box(np.array([centre[0], 0.004, centre[2]]), (1.0, 0.008, 2.3), yaw_of(axis), mat)
    sc.focus = p0


# Low-profile mechanic's creeper: board 0.42 x 1.0 m, top 6.5 cm above the floor (world.js).
CREEPER_TOP_M, CREEPER_HALF = 0.065, (0.21, 0.5)


def creeper_span(ctx: Ctx) -> tuple[float, float]:
    """Time on the creeper: end of the lie-down to the start of the last get-up (s)."""
    lie = ctx.rec.find("lie")[0]
    gets = ctx.rec.find("get_up")
    return lie.t1, (gets[-1].t0 if gets else ctx.n / ctx.fs)


def scene_liework(sc: Scene, ctx: Ctx) -> None:
    """The board is also a walkable at its resting place, covering the creeper slides, so the
    rendering lifts the lying body onto it (physics.rest_lift) and limbs over it rest on it."""
    call = ctx.rec.find("lie")[0]
    t0, t1 = creeper_span(ctx)
    idx = np.arange(ctx.idx(t0), ctx.idx(t1) + 1, 20)
    mid = (ctx.kin.anchors[idx, 0] + ctx.kin.anchors[idx, 3]) / 2
    p0, p3 = ctx.spine_point(t0, 0), ctx.spine_point(t0, 3)
    axis = flat(p3 - p0)
    axis = unit(axis) if np.linalg.norm(axis) > 0.1 else fwd(call.heading0)
    centre = flat(p0 + p3) / 2
    slide = float(np.max(np.abs((mid - centre) @ axis))) if len(mid) else 0.0
    yaw = yaw_of(axis)
    sc.walkables.append(Walkable(centre, CREEPER_TOP_M, (CREEPER_HALF[0], CREEPER_HALF[1] + slide), yaw))
    sc.dyn["creeper"] = True
    sc.props.append({"t": "dyn", "k": "creeper", "top": CREEPER_TOP_M})
    p0 = ctx.spine_point(call.t1, 0)
    side = right(call.heading0)
    sc.decor("toolbox", flat(p0) + side * 1.25 + fwd(call.heading0) * 0.3, call.heading0)
    sc.decor("drum", flat(p0) - side * 1.6 - fwd(call.heading0) * 0.8, call.heading0)
    sc.focus = p0


def scene_nap(sc: Scene, ctx: Ctx) -> None:
    scene_lie(sc, ctx, "tarp")
    call = ctx.rec.find("lie")[0]
    sc.decor("cooler", ctx.spine_point(call.t1, 3) + right(call.heading0) * 0.6, call.heading0)


def scene_carry(sc: Scene, ctx: Ctx) -> None:
    sc.dyn["bags"] = True
    sc.props.append({"t": "dyn", "k": "bags", "size": list(BAG_SIZE_M)})
    first = ctx.rec.find("carry")[0]
    sc.decor("pallet", flat(first.pos0) + fwd(first.heading0) * 0.6 + right(first.heading0) * 1.0, first.heading0)
    sc.focus = flat(first.pos0)


def scene_hat(sc: Scene, ctx: Ctx) -> None:
    call = ctx.rec.find("hat_off")[0]
    sc.dyn["hat"] = {"t": call.t0 + 0.1, "heading": call.heading0}  # the tug is 0.1 s into hat_off()
    sc.focus = flat(call.pos0) + UP * 1.4


def scene_table(sc: Scene, ctx: Ctx) -> None:
    call = ctx.rec.find("remove_strip")[0]
    table_h = float(call.args.get("table_height", 0.75))
    t_end = call.t1
    i = ctx.idx(t_end)
    centre = ctx.kin.anchors[i].mean(axis=0)
    centre = flat(centre)
    h = call.heading0
    sc.box(np.array([centre[0], table_h - 0.02, centre[2]]), (1.2, 0.04, 0.75), h, "plywood", walk=False)
    for a in (-0.55, 0.55):
        for b in (-0.32, 0.32):
            p = centre + right(h) * a + fwd(h) * b
            sc.box(np.array([p[0], (table_h - 0.04) / 2, p[2]]), (0.05, table_h - 0.04, 0.05), h, "steel")
    sc.focus = centre + UP * table_h


def scene_plain(sc: Scene, ctx: Ctx) -> None:
    sc.focus = ctx.pelvis(min(15.0, ctx.n / ctx.fs - 1))


SCENES = {
    "H-LAD": scene_ladder_fall, "H-SCAF": scene_scaffold, "H-ROOF": scene_roof, "H-HOLE": scene_hole,
    "H-EXC": scene_excavation, "L-SLIP": scene_slip, "L-TRIP": scene_trip, "STAIR": scene_stairs,
    "STRUCK-KD": scene_struck, "ARR": scene_harness, "NET": scene_net, "SYNC": scene_collapse,
    "SYNC-KNEEL": scene_kneel, "NF-STUMBLE": scene_stumble, "NF-RAIL": scene_rail, "STRUCK-NF": scene_struck,
    "SLIP-FRICTION": scene_slip_friction, "A-HAMMER": scene_hammer, "A-DRILL": scene_drill,
    "A-JACK": scene_jackhammer, "A-EQUIP": scene_equipment, "A-KNEEL": scene_kneel, "A-REBAR": scene_rebar,
    "A-LADDER": scene_ladder_work, "A-JUMP": scene_jump, "A-SIT": scene_sit, "A-LIE": scene_lie,
    "A-LIEWORK": scene_liework, "A-NAP": scene_nap, "A-CARRY": scene_carry, "A-RUN": scene_plain,
    "HAT-OFF": scene_hat, "STRAP-SLIP": scene_plain, "DEVICE-OFF": scene_table, "FALL-GETUP": scene_trip,
}


def build_scene(ctx: Ctx, scenario_id: str) -> Scene:
    site = ctx.run.site
    sc = Scene(surface=site.surface, condition=str(ctx.params.get("condition", site.condition)))
    SCENES.get(scenario_id, scene_plain)(sc, ctx)
    add_site_decor(sc, ctx)
    return sc


def add_site_decor(sc: Scene, ctx: Ctx) -> None:
    """Context dressing away from everywhere the worker goes: fence, cones, materials,
    a site trailer and a distant tower crane."""
    pts = ctx.track.root_pos[::200, [0, 2]]
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centre = (lo + hi) / 2
    span = float(np.max(hi - lo))
    rng = np.random.default_rng(ctx.run.seed + 77)
    path = ctx.track.root_pos[::100]

    def clear(x, z, radius) -> bool:
        if np.min(np.hypot(path[:, 0] - x, path[:, 2] - z)) < radius + 1.6:
            return False
        for px, pz, pr in sc.keepout:
            if np.hypot(px - x, pz - z) < pr + radius:
                return False
        for prop in sc.props:
            p = prop.get("p")
            if p is not None and prop["t"] != "patch" and np.hypot(p[0] - x, p[2] - z) < radius + 1.5:
                return False
        return True

    kinds = [("cone", 0.4), ("cone", 0.4), ("cone", 0.4), ("barrier", 1.2), ("pallet", 0.9), ("lumber_stack", 1.6),
             ("drum", 0.4), ("pipes", 1.6), ("block_stack", 0.8), ("cone", 0.4), ("wheelbarrow", 0.8)]
    radius_lo = 4.0
    placed = 0
    for kind, size in kinds * 2:
        for _ in range(30):
            ang = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(radius_lo, radius_lo + 9.0) + span / 2
            x, z = centre[0] + np.cos(ang) * dist, centre[1] + np.sin(ang) * dist
            if clear(x, z, size):
                y = sc.support(x, z, 50.0) if sc.ground_y >= -0.01 else sc.ground_y
                sc.decor(kind, np.array([x, max(y, sc.ground_y), z]), rng.uniform(0, 2 * np.pi))
                sc.keep(np.array([x, 0, z]), size)
                placed += 1
                break
    far = span / 2 + 22.0
    for kind, size in (("trailer", 4.5), ("toilet", 1.0), ("container", 4.0)):
        for _ in range(30):
            ang = rng.uniform(0, 2 * np.pi)
            x, z = centre[0] + np.cos(ang) * (far - 6), centre[1] + np.sin(ang) * (far - 6)
            if clear(x, z, size):
                sc.decor(kind, np.array([x, sc.ground_y, z]), ang + np.pi / 2)
                sc.keep(np.array([x, 0, z]), size)
                break
    ang = rng.uniform(0, 2 * np.pi)
    sc.decor("tower_crane", np.array([centre[0] + np.cos(ang) * (far + 18), sc.ground_y, centre[1] + np.sin(ang) * (far + 18)]),
             rng.uniform(0, 2 * np.pi))
    sc.props.append({"t": "fence", "c": r3([centre[0], sc.ground_y, centre[1]]), "r": round(far, 2)})


# --- dynamic props -------------------------------------------------------------------------
def dynamics(sc: Scene, ctx: Ctx, frames: np.ndarray, rig) -> dict:
    """Per-frame channels for moving props (rows = output frames). frames: 1 kHz sample indices.

    crane (N, 4): beam centre x, y, z and swing angle (rad, + towards the worker); crane_rig:
      static pivot, crib and boom. vehicle (N, 6): chassis x, y (tyre bottoms), z, yaw, seat
      travel (m), odometer (m). stepladder (N, 5): x, y, z, yaw, open. creeper (N, 4): x, y, z,
      yaw. bag<k> (N, 4): sack centre x, y, z, yaw. hat (N, 7): position + xyzw; hat_on (N,).
      dring (N, 3); lanyard: anchor, lengths and the rope points (D-ring to snap hook) as
      keyframes (see lanyard_motion); lanyard_len (N,): its current length (m). net_cloth: the
      safety net's cloth grid as keyframes (see net_motion)."""
    out: dict = {}
    if "crane" in sc.dyn:
        out.update(crane_motion(sc, ctx, frames, rig))
    if "lanyard" in sc.dyn:
        out["dring"] = rig.dring
        out.update(lanyard_motion(sc, ctx, frames, rig))
    if "net" in sc.dyn:
        out.update(net_motion(sc, ctx, frames, rig))
    if "vehicle" in sc.dyn:
        out["vehicle"] = vehicle_motion(sc, ctx, frames, rig)
    if "hat" in sc.dyn:
        out.update(hat_motion(sc, ctx, frames, rig))
    if sc.dyn.get("bags"):
        out.update(bag_motion(sc, ctx, frames, rig))
    if sc.dyn.get("stepladder"):
        out["stepladder"] = stepladder_motion(sc, ctx, frames, rig)
    if sc.dyn.get("creeper"):
        out["creeper"] = creeper_motion(sc, ctx, frames, rig)
    if "jack_rest" in sc.dyn:
        rest = sc.dyn["jack_rest"]
        y = sc.support(rest["p"][0], rest["p"][2], 50.0)
        out["jack_rest"] = np.array([rest["p"][0], y, rest["p"][2], rest["yaw"]])
    held = getattr(rig, "held", None)
    if held:
        # The contact pass (contact.py) moved the held objects with the hands: its poses win.
        if any(key.startswith("bag") for key in held):
            for key in [key for key in out if key.startswith("bag")]:
                del out[key]
        out.update(held)
    return out


# --- crane ---------------------------------------------------------------------------------
def strike_contact(ctx: Ctx, rig, frames: np.ndarray, k: int, dvec: np.ndarray, chest: np.ndarray) -> np.ndarray:
    """Beam centre at the strike: its front face on the first body surface it meets (drawn torso
    tube, upper arms, forearms and thighs) within the beam's height band and length."""
    _, M = spine_pose(ctx, frames[k:k + 1])
    s, w = body_scale(ctx.worker)
    pts = [torso_surface(rig.spine[k:k + 1], M, ctx.H, s, w).reshape(-1, 3)]
    radius = [np.zeros(len(pts[0]))]
    joints = rig.joints[k]
    # (from joint, to joint, radius / s): upper arms, forearms, thighs (body.js limb thickness)
    for a, b, r in ((10, 12, 0.05), (11, 13, 0.05), (12, 14, 0.045), (13, 15, 0.045), (0, 2, 0.07), (1, 3, 0.07)):
        u = np.linspace(0.0, 1.0, 9)[:, None]
        pts.append(joints[a] + (joints[b] - joints[a]) * u)
        radius.append(np.full(9, r * s))
    pts, radius = np.vstack(pts), np.concatenate(radius)
    e = perp(dvec)
    band = (np.abs(pts[:, 1] - chest[1]) <= BEAM_HALF_H_M + radius + 0.01) & (np.abs((pts - chest) @ e) <= BEAM_LEN_M / 2)
    front = float(np.min(pts[band] @ dvec - radius[band])) if band.any() else float(chest @ dvec) - 0.15
    return chest + dvec * (front - BEAM_HALF_DEPTH_M - float(chest @ dvec))


def crane_motion(sc: Scene, ctx: Ctx, frames: np.ndarray, rig) -> dict:
    """Pendulum of length CRANE_ROPE_M under the boom tip, integrated at 1 kHz (RK4): at rest on
    the crib until the release, a free swing that reaches the bottom of the arc exactly at the
    strike (elliptic quarter period), the measured contact force during the jolt, then a slowly
    decaying swing."""
    c = sc.dyn["crane"]
    t = frames / ctx.fs
    dvec, t_s, theta0 = c["dvec"], c["t_s"], c["theta0"]
    k_s = int(np.argmin(np.abs(t - t_s)))
    contact = strike_contact(ctx, rig, frames, k_s, dvec, c["chest"])
    rope = CRANE_ROPE_M
    pivot = contact + UP * rope
    omega2 = GRAVITY / rope
    t_release = t_s - float(ellipk(np.sin(theta0 / 2) ** 2)) / np.sqrt(omega2)
    peak_force = c["m_worker"] * c["jolt_g"] * GRAVITY        # N, on the beam along -dvec
    width = c["width"]

    def angular_accel(ti: float, th: float, om: float) -> float:
        acc = -omega2 * np.sin(th)
        u = (ti - t_s) / width
        if 0.0 <= u <= 1.0:
            acc -= peak_force * np.sin(np.pi * u) ** 2 * np.cos(th) / (BEAM_KG * rope)
        elif u > 1.0:
            acc -= om / AIR_DAMPING_S
        return acc

    dt = 1e-3
    steps = max(int(np.ceil((t[-1] - t_release) / dt)) + 2, 2)
    th_sim = np.empty(steps)
    th, om = -theta0, 0.0
    for i in range(steps):
        th_sim[i] = th
        ti = t_release + i * dt
        k1t, k1o = om, angular_accel(ti, th, om)
        k2t, k2o = om + 0.5 * dt * k1o, angular_accel(ti + 0.5 * dt, th + 0.5 * dt * k1t, om + 0.5 * dt * k1o)
        k3t, k3o = om + 0.5 * dt * k2o, angular_accel(ti + 0.5 * dt, th + 0.5 * dt * k2t, om + 0.5 * dt * k2o)
        k4t, k4o = om + dt * k3o, angular_accel(ti + dt, th + dt * k3t, om + dt * k3o)
        th += dt * (k1t + 2 * k2t + 2 * k3t + k4t) / 6
        om += dt * (k1o + 2 * k2o + 2 * k3o + k4o) / 6
    theta = np.interp(t, t_release + dt * np.arange(steps), th_sim, left=-theta0)
    pos = pivot[None] + rope * (np.sin(theta)[:, None] * dvec[None] - np.cos(theta)[:, None] * UP[None])
    # Crib under the resting beam; its front edge just behind the beam centre.
    rest = pivot + rope * (-np.sin(theta0) * dvec - np.cos(theta0) * UP)
    crib = rest - dvec * (CRIB_OVERHANG_M + CRIB_DEPTH_M / 2)
    crib[1] = rest[1] - BEAM_HALF_H_M
    foot = flat(pivot) + c["boom_dir"] * BOOM_REACH_M
    foot[1] = sc.ground_y + BOOM_FOOT_Y
    info = {"pivot": r3(pivot), "dvec": r3(dvec), "yaw": round(yaw_of(dvec), 4), "rope_m": rope, "mass_kg": BEAM_KG,
            "len": BEAM_LEN_M, "half_depth": BEAM_HALF_DEPTH_M, "half_h": BEAM_HALF_H_M, "hook_top": HOOK_TOP_M,
            "t_release": round(t_release, 3), "v0": round(c["v0"], 3),
            "crib": {"top": r3(crib), "size": [round(BEAM_LEN_M - 0.6, 3), CRIB_DEPTH_M], "ground": round(sc.ground_y, 3)},
            "boom": {"foot": r3(foot), "tip": r3(pivot)}}
    return {"crane": np.column_stack([pos, theta]), "crane_rig": info}


# --- loader (A-EQUIP) ------------------------------------------------------------------------
SEAT_TRAVEL_M = (-0.30, 0.15)    # suspension seat travel limits around its rest height (m)


def vehicle_motion(sc: Scene, ctx: Ctx, frames: np.ndarray, rig) -> np.ndarray:
    """The chassis drives on the ground along the seated pelvis path (zero-phase 0.5 Hz low pass);
    the seat follows the drawn pelvis up and down within its travel, so the simulated jolts stay
    in the operator, not in the loader. Parked before the ride and after it."""
    v = sc.dyn["vehicle"]
    i_seat, i_leave = ctx.idx(v["t_seat"]), ctx.idx(v["t_leave"])
    path = ctx.track.root_pos[i_seat:i_leave + 1][:, [0, 2]].copy()
    if len(path) > 200:
        path = signal.sosfiltfilt(signal.butter(2, 0.5, fs=ctx.fs, output="sos"), path, axis=0)
    ride_idx = np.clip(frames, i_seat, i_leave)
    x, z = path[ride_idx - i_seat, 0], path[ride_idx - i_seat, 1]
    yaw = np.unwrap(ctx.track.heading[ride_idx])
    y = np.array([sc.support(xi, zi, 50.0) for xi, zi in zip(x, z, strict=True)])
    k_seat = int(np.argmin(np.abs(frames - i_seat)))
    riding = (frames >= i_seat) & (frames <= i_leave)
    travel = (rig.pelvis[:, 1] - rig.pelvis[k_seat, 1]) - (y - y[k_seat])
    seat = np.where(riding, np.clip(travel, *SEAT_TRAVEL_M), 0.0)
    odometer = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(z)))])
    return np.column_stack([x, y, z, yaw, seat, odometer])


# --- prop physics (lanyard, safety net, hard hat) --------------------------------------------
# These props are simulated with the viewer's PBD engine (physics.World) against the drawn body
# (body_capsules, kinematic) and the drawn scene (physics.scene_colliders). The body is never
# moved by them: the spine is simulated ground truth and the limbs come from the rig/ragdoll.
def region_colliders(sc: Scene, pts: np.ndarray, pad: float) -> Colliders:
    """Scene colliders near a set of points (m), without the safety net sheet."""
    pts = np.asarray(pts, float).reshape(-1, 3)
    col = scene_colliders(sc, pts.min(axis=0) - pad, pts.max(axis=0) + pad)
    col.sheets.clear()
    return col


def frame_step(t: np.ndarray) -> float:
    return float(np.median(np.diff(t))) if len(t) > 1 else 0.02


# --- harness lanyard (ARR) --------------------------------------------------------------------
LANYARD_PTS = 13              # rope particles from the D-ring to the snap hook
LANYARD_KG = 1.2              # webbing, absorber pack and snap hook
LANYARD_R_M = 0.012           # webbing collision radius (drawn as a 12 mm tube)
ABSORBER_MAX_M = 1.07         # pay-out limit: OSHA 1926.502(d)(16)(iv) deceleration distance
CLIP_S = 0.6                  # snap hook moved from the harness keeper to the anchor
TAUT = 0.999                  # chord / length above which the lanyard is a straight line
LANYARD_TOL_M = 0.004         # keyframes: largest linear-interpolation error between them (m)


def taut_clamp(points: np.ndarray, length: float) -> np.ndarray:
    """Rope points (n, 3) blended toward their straight chord until the polyline is no longer
    than `length` (m); the points themselves when they already fit."""
    def polyline(x):
        return float(np.linalg.norm(np.diff(x, axis=0), axis=1).sum())
    if polyline(points) <= length * 1.005:
        return points
    line = points[0] + (points[-1] - points[0]) * np.linspace(0.0, 1.0, len(points))[:, None]
    lo, hi = 0.0, 1.0
    for _ in range(12):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if polyline(points + (line - points) * mid) > length else (lo, mid)
    return points + (line - points) * hi


def rope_step(world: World, ids: np.ndarray, length: float, prev: np.ndarray, dt: float) -> np.ndarray:
    """Step a rope world by dt and return its points, kept no longer than `length` (m). Only
    when the rope snaps taut faster than the solver follows (taut_clamp moved the points) are
    the particles reset onto the clamped points, with the velocity from the previous frame's
    points `prev`; a slack rope keeps its own state (and falls freely under gravity)."""
    world.step(dt)
    clamped = taut_clamp(world.x, length)
    if clamped is not world.x:
        world.set_state(ids, clamped, (clamped - prev) / dt)
    return clamped.copy()


def lanyard_motion(sc: Scene, ctx: Ctx, frames: np.ndarray, rig) -> dict:
    """PBD rope from the D-ring to the snap hook (both pinned), colliding with the drawn torso
    and scene: slack and sagging in free fall until it goes taut (at most TAUT_LEAD_S before
    free_fall_end), then loaded. Length: the relaxed L0 until then; after that the stretch
    lanyard pays out with the D-ring (up to ABSORBER_MAX_M) and recoils when the load drops, so
    it is straight while the worker hangs and never longer than its length (taut_clamp when it
    snaps taut faster than the solver follows). Before plan['t_clip'] the hook is stowed on a
    keeper at the left hip and is moved over the D-ring to the anchor in CLIP_S.

    Output lanyard: anchor, n (points), L0, t_taut (simulated), t_loaded (drawn taut), t_clip,
    keys (output frame indices) and x (len(keys) * n * 3, m), drawn by linear interpolation
    between keys."""
    plan = sc.dyn["lanyard"]
    anchor, L0 = plan["anchor"], plan["L0"]
    t = frames / ctx.fs
    n, dt = len(t), frame_step(t)
    ring = rig.dring
    chord_anchor = np.linalg.norm(ring - anchor, axis=1)
    length = np.where(t < plan["t_loaded"] - 0.05, L0, np.clip(chord_anchor, L0, L0 + ABSORBER_MAX_M))
    s, w = body_scale(ctx.worker)
    _, M = spine_pose(ctx, frames)
    keeper = rig.spine[:, 0] + M[:, 0] @ np.array([-0.17 * s * w, 0.05 * ctx.H, 0.0])
    if plan["t_clip"] is None:
        hook = np.tile(anchor, (n, 1))
    else:
        # keeper -> over the D-ring -> anchor, so the rope drapes over the back, not through it
        clip = (t - plan["t_clip"]) / CLIP_S
        over = ring + UP * 0.4
        first = keeper + (over - keeper) * smoothstep(2 * clip)[:, None]
        hook = first + (anchor - first) * smoothstep(2 * clip - 1)[:, None]
    caps = body_capsules(ctx, rig, frames, ("torso",), COARSE_RINGS)
    col = region_colliders(sc, np.vstack([ring, hook]), L0 + 0.3)
    body = add_body(col, caps, 0, mu=0.4)
    world = World(col, iterations=2, air_damping=0.3)
    u = np.linspace(0.0, 1.0, LANYARD_PTS)[:, None]
    sag = np.sqrt(max(L0 ** 2 - float(np.sum((hook[0] - ring[0]) ** 2)), 0.0)) / 2
    start = ring[0] + (hook[0] - ring[0]) * u - UP * sag * np.sin(np.pi * u)      # a V-ish loop to start
    ids = world.add_particles(start, LANYARD_KG / LANYARD_PTS, LANYARD_R_M)
    links = world.add_distance(ids[:-1], ids[1:], lo=0.0, hi=L0 / (LANYARD_PTS - 1))
    pins = world.add_pin(ids[[0, -1]])
    for _ in range(50):                                     # settle for 1 s before the first frame
        world.step(dt)
    out = np.empty((n, LANYARD_PTS, 3))
    out[0] = world.x
    for k in range(1, n):
        move_body(col, body, caps, k)
        ends = np.stack([ring[k], hook[k]])
        if np.linalg.norm(hook[k] - ring[k]) >= TAUT * length[k]:     # loaded: straight
            line = ring[k] + (hook[k] - ring[k]) * u
            world.set_state(ids, line, (line - world.x) / dt)
            world.set_pin_targets(pins, ends, start=ends)
            col.end_step()
            out[k] = line
            continue
        world.set_pin_targets(pins, ends, start=np.stack([ring[k - 1], hook[k - 1]]))
        world.set_distance(links, lo=0.0, hi=length[k] / (LANYARD_PTS - 1))
        out[k] = rope_step(world, ids, length[k], out[k - 1], dt)
    keys = keyframes(out, LANYARD_TOL_M)
    rope = {"anchor": r3(anchor), "n": LANYARD_PTS, "L0": round(L0, 3), "t_taut": round(plan["t_taut"], 3),
            "t_loaded": round(plan["t_loaded"], 3),
            "t_clip": None if plan["t_clip"] is None else round(plan["t_clip"], 3),
            "lanyard_est": round(plan["lanyard_est"], 3), "hang_tilt_deg": round(plan["hang_tilt_deg"], 1),
            "keys": [int(k) for k in keys], "x": np.round(out[keys], 3).ravel().tolist()}
    return {"lanyard": rope, "lanyard_len": np.round(length, 4)}


def keyframes(states: np.ndarray, tol: float, max_span: int = 50) -> list:
    """Frames to keep so that linear interpolation between them reproduces every frame of states
    (n, points, 3) within tol (m) (greedy; runs of identical frames need only their ends)."""
    n = len(states)
    same = np.zeros(n, bool)
    same[1:] = np.all(states[1:] == states[:-1], axis=(1, 2))
    keys, a = [0], 0
    while a < n - 1:
        b = a + 1
        while b + 1 < n and same[b + 1] and same[a + 1]:
            b += 1
        if b > a + 1:
            keys.append(b)
            a = b
            continue
        best = a + 1
        for b in range(a + 2, min(a + max_span, n - 1) + 1):
            mid = (np.arange(a + 1, b) - a) / (b - a)
            err = np.abs(states[a + 1:b] - (states[a] + (states[b] - states[a]) * mid[:, None, None])).max()
            if err > tol:
                break
            best = b
        keys.append(best)
        a = best
    return keys


# --- safety net (NET) --------------------------------------------------------------------------
NET_SPACING_M = 0.29          # cloth grid spacing, simulated and drawn
NET_KG_M2 = 0.45              # knotted polypropylene personnel net
NET_PRETENSION = 0.08         # links 8 % shorter than the grid: the net sags ~5 cm under its own
                              # weight with one solver iteration (2 %: ~16 cm, a PBD residual)
NET_DAMPING = 2.0             # 1/s (air and knot friction)
NET_NODE_R_M = 0.03           # half thickness of the net (mesh and knots) under the body
NET_REACH_M = NET_SPACING_M      # nodes up to a grid spacing beside a body part are pushed down
                                 # too, so the drawn triangles between nodes stay under thin limbs
NET_MU = 0.3                  # friction between the net and the body
NET_TOL_M = 0.006             # keyframes: largest linear-interpolation error between them (m)
NET_QUIET_M = 3e-4            # a node moving less than this per frame is at rest (m)
NET_WAKE_M = 0.002            # body motion (m) that wakes a resting net
FAR = 1e6                     # "no body here" height (m), finite so heights can be interpolated


class NetWorld(World):
    """physics.World for a net under the drawn body. The body pushes the nodes straight down (a
    one-sided contact): with the engine's radial capsule push a node that slips past the side of
    a sinking limb would be pushed up and the net would close over the body. The body is given
    per step as the lowest and highest heights a node centre may not be between (set_body)."""

    def __init__(self, *args, mu: float = NET_MU, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = float(mu)
        self.body = None

    def set_body(self, low0, low1, top0, top1) -> None:
        """Body over each particle at the start and the end of the coming step: a node centre
        between low and top (m) is pushed down to low (FAR / -FAR where there is no body)."""
        self.body = (low0, low1 - low0, top0, top1 - top0)

    def _collide(self, p: np.ndarray, xp: np.ndarray, shapes, alpha: float) -> None:
        super()._collide(p, xp, shapes, alpha)
        if self.body is None:
            return
        low0, dlow, top0, dtop = self.body
        low = low0 + dlow * alpha
        hit = (p[:, 1] > low) & (p[:, 1] < top0 + dtop * alpha) & (self.w > 0)
        if not hit.any():
            return
        depth = np.where(hit, p[:, 1] - low, 0.0)
        p[:, 1] -= depth
        tx, tz = p[:, 0] - xp[:, 0], p[:, 2] - xp[:, 2]
        slide = np.sqrt(tx * tx + tz * tz)
        lim = self.mu * np.minimum(depth, 0.002)
        keep = np.where(hit, np.where(slide <= lim, 1.0, np.minimum(0.8 * lim / np.maximum(slide, 1e-12), 1.0)), 0.0)
        p[:, 0] -= tx * keep
        p[:, 2] -= tz * keep
        xp[:, 1] = np.where(hit, np.minimum(xp[:, 1], p[:, 1]), xp[:, 1])     # no velocity away from the body


def body_spheres(caps: tuple) -> tuple:
    """The body capsules as overlapping spheres at most half a radius apart along each axis (the
    union is within ~R/32 of the capsule surface): (centres (N, S, 3) per frame, radii (S,))."""
    a, b, r = caps
    count = np.ceil(np.linalg.norm(b - a, axis=2).max(axis=0) / (0.5 * r)).astype(int) + 1
    cap = np.repeat(np.arange(len(r)), count)
    u = np.concatenate([np.linspace(0.0, 1.0, m) for m in count])[None, :, None]
    return a[:, cap] + (b[:, cap] - a[:, cap]) * u, r[cap]


def body_over(xz: np.ndarray, spheres: tuple, pad: float, reach_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Heights (n,) between which a net node at horizontal positions xz (n, 2) must not be: each
    body sphere (radius r) replaced by a sphere of radius r + reach_m with the same bottom, less
    pad. Nodes up to reach_m beside a limb are pushed down nearly to its bottom, so the drawn
    triangles between nodes reach_m apart stay below it; under a wide torso the net forms a
    shallow bowl rather than hugging its sides. FAR / -FAR where no sphere is over a node."""
    c, r = spheres
    low, top = np.full(len(xz), FAR), np.full(len(xz), -FAR)
    reach = float(r.max()) + reach_m
    near = np.flatnonzero((xz[:, 0] >= c[:, 0].min() - reach) & (xz[:, 0] <= c[:, 0].max() + reach)
                          & (xz[:, 1] >= c[:, 2].min() - reach) & (xz[:, 1] <= c[:, 2].max() + reach))
    if len(near):
        q = xz[near]
        d2 = (q[:, None, 0] - c[None, :, 0]) ** 2 + (q[:, None, 1] - c[None, :, 2]) ** 2
        big = r[None] + reach_m
        half = np.sqrt(np.maximum(big ** 2 - d2, 0.0))
        inside = d2 < big ** 2
        centre_y = c[None, :, 1] - r[None] - pad + big          # same bottom as the sphere, less pad
        low[near] = np.where(inside, centre_y - half, FAR).min(axis=1)
        top[near] = np.where(inside, centre_y + half, -FAR).max(axis=1)
    return low, top


def net_motion(sc: Scene, ctx: Ctx, frames: np.ndarray, rig) -> dict:
    """Safety net as a cloth: a grid of PBD nodes joined by pretensioned links, pinned along its
    border and sagging a few cm under its own weight. The drawn body (body_capsules: torso
    sections, limbs, boots, head) is kinematic and pushes the nodes down (NetWorld), so the net
    wraps under it; the simulated trunk path is ground truth, so the net adapts to the body and
    never the reverse. Simulated from the first frame a body part comes near it, asleep while
    the body and the net are still.

    Output net_cloth: nx, nz (grid size, row-major along the net's length), rest (nz * nx * 3, m:
    the settled sag), keys (output frame indices), and each key's node offsets from rest (mm,
    nz * nx * 3 ints) as changes from the previous key in sparse form (sparse_steps: d = (zeros
    skipped, change) pairs, nnz = pairs per key). Drawn by linear interpolation between keys."""
    info = sc.dyn["net"]
    centre = np.array([info["centre"][0], info["y"], info["centre"][2]])
    r, f = right(info["yaw"]), fwd(info["yaw"])
    nx = int(round(info["w"] / NET_SPACING_M)) + 1
    nz = int(round(info["l"] / NET_SPACING_M)) + 1
    u = np.linspace(-info["w"] / 2, info["w"] / 2, nx)
    v = np.linspace(-info["l"] / 2, info["l"] / 2, nz)
    grid = (centre + v[:, None, None] * f + u[None, :, None] * r).reshape(-1, 3)
    t = frames / ctx.fs
    n, dt = len(t), frame_step(t)
    world = NetWorld(Colliders(None), air_damping=NET_DAMPING)
    ids = world.add_particles(grid, NET_KG_M2 * (u[1] - u[0]) * (v[1] - v[0]), NET_NODE_R_M)
    index = ids.reshape(nz, nx)
    world.add_distance(index[:, :-1].ravel(), index[:, 1:].ravel(), (u[1] - u[0]) * (1 - NET_PRETENSION))
    world.add_distance(index[:-1].ravel(), index[1:].ravel(), (v[1] - v[0]) * (1 - NET_PRETENSION))
    edge = np.ones((nz, nx), bool)
    edge[1:-1, 1:-1] = False
    world.add_pin(index[edge])
    for _ in range(100):                                  # settle under its own weight (2 s)
        world.step(dt)
    rest = np.round(world.x, 3)
    world.set_state(ids, rest)
    states = np.repeat(rest[None].astype(np.float32), n, axis=0)
    caps = body_capsules(ctx, rig, frames)
    a, b, rad = caps
    low = np.minimum(a[..., 1], b[..., 1]) - rad
    rel = (a + b) / 2 - centre
    inside = (np.abs(rel @ r) <= info["w"] / 2 + 0.5) & (np.abs(rel @ f) <= info["l"] / 2 + 0.5)
    near = np.flatnonzero(((low < centre[1] + 0.3) & inside).any(axis=1))
    stats = {"frames": 0, "asleep": 0}
    if len(near):
        k0 = int(near[0])
        asleep, quiet, ref = False, 0, k0
        centres, radii = body_spheres(caps)
        for k in range(k0 + 1, n):
            if asleep:
                drift = max(np.abs(a[k] - a[ref]).max(), np.abs(b[k] - b[ref]).max())
                if drift < NET_WAKE_M:
                    states[k] = states[k - 1]
                    stats["asleep"] += 1
                    continue
                asleep, quiet = False, 0
            low0, top0 = body_over(world.x[:, [0, 2]], (centres[k - 1], radii), NET_NODE_R_M, NET_REACH_M)
            low1, top1 = body_over(world.x[:, [0, 2]], (centres[k], radii), NET_NODE_R_M, NET_REACH_M)
            world.set_body(low0, low1, top0, top1)
            world.step(dt)
            states[k] = world.x
            stats["frames"] += 1
            moved = max(np.abs(a[k] - a[k - 1]).max(), np.abs(b[k] - b[k - 1]).max())
            still = np.abs(states[k] - states[k - 1]).max() < NET_QUIET_M and moved < NET_WAKE_M / 2
            quiet = quiet + 1 if still else 0
            if quiet >= 10:
                asleep, ref = True, k
                world.set_state(ids, world.x)
    keys = keyframes(states, NET_TOL_M)
    mm = np.round((states[keys] - rest[None]) * 1000.0).astype(int).reshape(len(keys), -1)
    d, nnz = sparse_steps(mm)
    return {"net_cloth": {"nx": nx, "nz": nz, "rest": rest.ravel().tolist(), "keys": [int(k) for k in keys], "d": d,
                          "nnz": nnz, "sim_frames": stats["frames"], "asleep_frames": stats["asleep"]}}


def sparse_steps(values: np.ndarray) -> tuple[list, list]:
    """Rows of ints (keys, m) as the changes from the previous row (the first from zeros), each
    row's non-zero changes as (zeros skipped, change) pairs: (flat pair list, pairs per row).
    Far from the body most nodes do not move between keys, so this is a short payload."""
    steps = np.diff(values, axis=0, prepend=np.zeros((1, values.shape[1]), int))
    pairs, counts = [], []
    for row in steps:
        nz = np.flatnonzero(row)
        skips = np.diff(nz, prepend=-1) - 1
        pairs.extend(np.column_stack([skips, row[nz]]).ravel().tolist())
        counts.append(len(nz))
    return pairs, counts


# --- hard hat (HAT-OFF) ---------------------------------------------------------------------
HAT_CABLE_M = 0.55
HAT_KG = 0.40
# The hat as a rigid cluster in its own frame (body.js hat group: +Y crown, -Z peak), / s: the
# cable point (the group origin, where body.js draws the cable from), crown, peak, back and
# both sides of the brim.
HAT_POINTS = np.array([[0.0, 0.0, 0.0], [0.0, 0.095, 0.0], [0.0, 0.0, -0.13], [0.0, 0.0, 0.125],
                       [-0.11, 0.0, 0.0], [0.11, 0.0, 0.0]])
HAT_R_M = 0.025               # collision radius of each point (shell and brim)
HAT_KNOCK = (-2.4, 1.0, 0.4)  # knock-off velocity (m/s) along fwd, up and right of the heading
HAT_SPIN = 6.0                # rad/s about the head's right axis: the hat tips backwards
HAT_REST_S = 0.5              # at rest this long: carried rigidly by the upper back until it moves
HAT_WAKE_ACC = 0.6            # m/s^2 of cable-anchor acceleration that sets a resting hat swinging


def hat_frame(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cable point (n, 3) and xyzw orientation (n, 4) of the hat from its points (n, 6, 3)."""
    from scipy.spatial.transform import Rotation

    y = unit(x[:, 1] - x[:, 0])
    z = x[:, 3] - x[:, 2]
    z = unit(z - y * np.sum(z * y, axis=1, keepdims=True))
    return x[:, 0], Rotation.from_matrix(np.stack([np.cross(y, z), y, z], axis=2)).as_quat()


def hat_flight(sc: Scene, t: np.ndarray, start: dict, anchor: np.ndarray, caps: tuple, frame_R: np.ndarray,
               until_taut: bool) -> dict:
    """Simulate the hat from its release (start: t, points, velocities) to the end of the run, or
    until the cable first goes taut. A rigid cluster of HAT_POINTS (exact distances) on a cable
    of HAT_CABLE_M (a range constraint) to the moving cable anchor; it collides with the scene and
    the drawn body. Once at rest for HAT_REST_S it is carried rigidly by the upper-back frame
    frame_R until the anchor accelerates (cheap long holds)."""
    n, dt = len(t), frame_step(t)
    k0 = int(np.searchsorted(t, start["t"]))
    anchor0 = np.array([np.interp(start["t"], t, anchor[:, j]) for j in range(3)])
    col = region_colliders(sc, np.vstack([start["x"], anchor[k0 - 1:]]), 1.0)
    body = add_body(col, caps, k0 - 1, mu=0.5)
    world = World(col)
    ids = world.add_particles(start["x"], HAT_KG / len(start["x"]), HAT_R_M)
    ii, jj = np.triu_indices(len(ids), 1)
    world.add_distance(ids[ii], ids[jj])
    tie = world.add_particles(anchor0, np.inf, 0.0, group=0)
    world.add_distance(ids[:1], tie, lo=0.0, hi=HAT_CABLE_M)
    pin = world.add_pin(tie)
    world.set_state(ids, start["x"], start["v"])
    acc = np.zeros(n)
    acc[1:-1] = np.linalg.norm(anchor[2:] - 2 * anchor[1:-1] + anchor[:-2], axis=1) / dt ** 2
    X = np.full((n, len(ids), 3), np.nan)
    t_taut, prev_t, prev_a, prev_d = None, start["t"], anchor0, float(np.linalg.norm(start["x"][0] - anchor0))
    quiet, carried = 0, None
    for k in range(k0, n):
        if carried is not None:
            if acc[k] < HAT_WAKE_ACC:
                X[k] = anchor[k] + carried @ frame_R[k].T
                prev_t, prev_a = t[k], anchor[k]
                continue
            # resume from the carried pose, moving with the anchor
            world.set_state(ids, X[k - 1], (X[k - 1] - X[k - 2]) / dt)
            world.set_pin_targets(pin, anchor[k - 1][None], start=anchor[k - 1][None])
            world.x[tie] = anchor[k - 1]
            move_body(col, body, caps, k - 1)
            col.end_step()
            carried, quiet = None, 0
        world.set_pin_targets(pin, anchor[k][None], start=prev_a[None])
        move_body(col, body, caps, k)
        before = world.x[ids].copy()
        world.step(t[k] - prev_t)
        X[k] = world.x[ids]
        dist = float(np.linalg.norm(X[k, 0] - anchor[k]))
        if t_taut is None and dist >= HAT_CABLE_M - 2e-3:
            t_taut = prev_t + (t[k] - prev_t) * float(np.clip((HAT_CABLE_M - prev_d) / max(dist - prev_d, 1e-9), 0, 1))
            if until_taut:
                break
        prev_t, prev_a, prev_d = t[k], anchor[k], dist
        moved = np.abs((X[k] - before) - (anchor[k] - anchor[k - 1])).max()
        quiet = quiet + 1 if moved < 0.002 * dt / 0.02 and acc[k] < HAT_WAKE_ACC else 0
        if quiet >= int(round(HAT_REST_S / dt)):
            carried = (X[k] - anchor[k]) @ frame_R[k]
    return {"x": X, "k0": k0, "t_taut": t_taut}


def hat_motion(sc: Scene, ctx: Ctx, frames: np.ndarray, rig) -> dict:
    """Hard hat knocked off, then caught by its cable on the upper sensor (rig.anchor3). A free
    rigid body under gravity (physics.World) leaving the head from its on-head pose (body.js
    hatOffset) with the head's velocity plus a knock and a backward spin. The release time is
    chosen so the cable first goes taut exactly at the simulated tug (hat_off + 0.1 s), then it
    swings, hits the worker's back and comes to rest hanging on the cable. On-head frames carry
    the on-head pose too, so the browser's interpolation never jumps."""
    from scipy.spatial.transform import Rotation

    t_tug, h = sc.dyn["hat"]["t"], sc.dyn["hat"]["heading"]
    t = frames / ctx.fs
    n, dt = len(t), frame_step(t)
    s, _ = body_scale(ctx.worker)
    head_q = continuous_quats(rig.head_q)
    on_pos = rig.head + Rotation.from_quat(head_q).apply([0.0, 0.045 * s, 0.005])
    pos, quat = on_pos.copy(), head_q.copy()
    caps = body_capsules(ctx, rig, frames, ("torso", "legs", "arms"))
    _, M = spine_pose(ctx, frames)
    knock = fwd(h) * HAT_KNOCK[0] + UP * HAT_KNOCK[1] + right(h) * HAT_KNOCK[2]

    def release(t_rel: float) -> dict | None:
        k0 = int(np.searchsorted(t, t_rel))
        if k0 <= 1 or k0 >= n:
            return None
        u = (t_rel - t[k0 - 1]) / (t[k0] - t[k0 - 1])
        p0 = on_pos[k0 - 1] + (on_pos[k0] - on_pos[k0 - 1]) * u
        R0 = Rotation.from_quat(quat_slerp(head_q[k0 - 1], head_q[k0], u)[0])
        x0 = p0 + R0.apply(HAT_POINTS * s)
        v0 = (on_pos[k0] - on_pos[k0 - 1]) / (t[k0] - t[k0 - 1]) + knock
        return {"t": t_rel, "x": x0, "v": v0 + np.cross(right(h) * HAT_SPIN, x0 - p0)}

    def late(t_rel: float) -> float | None:
        """How late (s) the cable goes taut after the tug for a release at t_rel."""
        start = release(t_rel)
        sim = hat_flight(sc, t, start, rig.anchor3, caps, M[:, 3], until_taut=True) if start else None
        return None if sim is None or sim["t_taut"] is None else sim["t_taut"] - t_tug

    # Release time such that the cable goes taut at the tug (secant search; the flight time
    # depends on how the body moves meanwhile).
    x0, x1 = t_tug - 0.45, t_tug - 0.35
    f0, f1 = late(x0), late(x1)
    for _ in range(6):
        if f0 is None or f1 is None or abs(f1) < 0.003 or f1 == f0:
            break
        x0, f0, x1 = x1, f1, float(np.clip(x1 - f1 * (x1 - x0) / (f1 - f0), t_tug - 1.2, t_tug - 0.1))
        f1 = late(x1)
    t_rel = x1 if f1 is not None else t_tug - 0.4
    start = release(t_rel)
    on = t < t_rel
    if start is None:
        return {"hat": np.column_stack([pos, quat]), "hat_on": on.astype(float)}
    sim = hat_flight(sc, t, start, rig.anchor3, caps, M[:, 3], until_taut=False)
    k0 = sim["k0"]
    pos[k0:], quat[k0:] = hat_frame(sim["x"][k0:])
    sc.dyn["hat"]["t_release"] = start["t"]
    sc.dyn["hat"]["t_taut"] = sim["t_taut"]
    return {"hat": np.column_stack([pos, continuous_quats(quat)]), "hat_on": on.astype(float)}


# --- sacks (A-CARRY) ------------------------------------------------------------------------
BAG_SIZE_M = (0.32, 0.55, 0.16)   # standing sack: across the body, height, depth


def carry_times(ctx: Ctx) -> list:
    """(carry call, lift-off time, lift end time, touchdown time) per carry, from the ground-truth
    events of its lift and put-down (motions.Builder.lift / put_down)."""
    items = []
    for call in ctx.rec.find("carry"):
        inside = [(t, n) for t, n, _ in ctx.events if call.t0 - 1e-6 <= t <= call.t1 + 1e-6]
        times = {n: t for t, n in reversed(inside)}
        if all(n in times for n in ("lift_off", "lift_end", "put_down_touch")):
            items.append((call, times["lift_off"], times["lift_end"], times["put_down_touch"]))
    return items


def bag_motion(sc: Scene, ctx: Ctx, frames: np.ndarray, rig) -> dict:
    """A standing sack per carry: on the floor where the hands grip it at lift-off, held rigidly
    from then on, and left standing on the floor at touchdown. From mid-rise until the worker is
    upright the sack is hugged up to where it will be held at the put-down, which keeps it clear
    of the knees on the walk."""
    t = frames / ctx.fs
    n = len(t)
    hands = (rig.hand_l + rig.hand_r) / 2
    heading = ctx.track.heading[frames]
    half_h = BAG_SIZE_M[1] / 2
    out: dict = {}
    for k, (call, pick, upright, drop) in enumerate(carry_times(ctx)):
        kp = min(int(np.searchsorted(t, pick)), n - 1)
        kd = min(int(np.searchsorted(t, drop)), n - 1)

        def rest_at(j: int) -> np.ndarray:
            p = hands[j].copy()
            p[1] = sc.support(p[0], p[2], p[1] + 0.3) + half_h
            return p

        rest_pick, rest_drop = rest_at(kp), rest_at(kd)
        grip_pick, grip_drop = rest_pick - hands[kp], rest_drop - hands[kd]
        mid = (pick + upright) / 2
        shift = smoothstep((t - mid) / max(upright + 0.1 - mid, 0.1))[:, None]
        offset = grip_pick + (grip_drop - grip_pick) * shift
        pos = hands + offset
        for j in range(kp, kd):                 # held low, the sack stands on the floor
            pos[j, 1] = max(pos[j, 1], sc.support(pos[j, 0], pos[j, 2], pos[j, 1] + 0.3) + half_h)
        pos[:kp], pos[kd:] = rest_pick, rest_drop
        yaw = heading.copy()
        yaw[:kp], yaw[kd:] = heading[kp], heading[kd]
        out[f"bag{k}"] = np.column_stack([pos, np.unwrap(yaw)])
        out.setdefault("bag_mass", []).append(round(float(call.args.get("load_kg", 20.0)), 1))
    return out


# --- stepladder (A-LADDER) -----------------------------------------------------------------
LADDER_SWAP_S = 0.9   # pick-up / set-down time (the rig's carryladder arm blend)


def stepladder_motion(sc: Scene, ctx: Ctx, frames: np.ndarray, rig) -> np.ndarray:
    """Stepladder standing at each climb and carried on edge in the right hand in between (upper
    rail through the hand). It already stands at the first spot when the run starts. Columns:
    x, y, z, yaw, open (1 standing, 0 carried)."""
    t = frames / ctx.fs
    n = len(t)
    out = np.zeros((n, 5))
    ups = sorted(sc.ladders, key=lambda lad: lad.t_up[0])
    if not ups:
        return out
    heading = ctx.track.heading[frames]
    length = (max(lad.rungs for lad in ups) * 0.3 + 0.7) / np.cos(np.arctan2(1, 4))   # rail length (world.js)
    carry = rig.hand_r + UP * (0.03 - 0.24) + right(heading) * (0.03 * length)
    sets = np.array([lad.t_up[0] for lad in ups])
    picks = np.array([lad.t_down[1] if lad.t_down else np.inf for lad in ups])
    weight = np.zeros(n)
    stand = np.zeros(n, int)
    for i, ti in enumerate(t):
        k = int(np.searchsorted(sets, ti, side="right")) - 1
        if k < 0 or ti <= picks[k]:
            stand[i] = max(k, 0)
            continue
        lift = float(smoothstep((ti - picks[k]) / LADDER_SWAP_S))
        if k + 1 < len(ups):
            put = float(smoothstep((sets[k + 1] - ti) / LADDER_SWAP_S))
            stand[i] = k + 1 if sets[k + 1] - ti < ti - picks[k] else k
            weight[i] = min(lift, put)
        else:
            stand[i], weight[i] = k, lift
    base = np.array([ups[j].base for j in stand])
    yaw0 = np.array([ups[j].yaw for j in stand])
    turn = (heading - yaw0 + np.pi) % (2 * np.pi) - np.pi
    out[:, :3] = base * (1 - weight)[:, None] + carry * weight[:, None]
    out[:, 3] = np.unwrap(yaw0 + turn * weight)
    out[:, 4] = 1 - weight
    return out


# --- creeper (A-LIEWORK) --------------------------------------------------------------------
def creeper_motion(sc: Scene, ctx: Ctx, frames: np.ndarray, rig) -> np.ndarray:
    """Board under the lying trunk (follows the creeper slides). Normally on its wheels; where the
    drawn torso is lower than the board top (no render lift at that moment) the board is drawn
    sunk into the floor instead of through the back."""
    from scipy.ndimage import gaussian_filter1d

    t0, t1 = creeper_span(ctx)
    t = frames / ctx.fs
    n = len(t)
    k = np.clip(np.searchsorted(t, np.clip(t, t0, t1)), 0, n - 1)
    mid = (rig.spine[k, 0] + rig.spine[k, 3]) / 2
    axis = flat(rig.spine[k, 3] - rig.spine[k, 0])
    yaw = np.unwrap(np.arctan2(-axis[:, 0], -axis[:, 2]))
    y = np.zeros(n)
    idx = np.flatnonzero((t >= t0) & (t <= t1))
    if len(idx):
        _, M = spine_pose(ctx, frames[idx])
        s, w = body_scale(ctx.worker)
        pts = torso_surface(rig.spine[idx], M, ctx.H, s, w, 12).reshape(len(idx), -1, 3)
        rel = pts - mid[idx][:, None]
        cy, sy = np.cos(yaw[idx])[:, None], np.sin(yaw[idx])[:, None]
        across = rel[..., 0] * cy - rel[..., 2] * sy
        along = -rel[..., 0] * sy - rel[..., 2] * cy
        over = (np.abs(across) <= CREEPER_HALF[0]) & (np.abs(along) <= CREEPER_HALF[1])
        low = np.where(over, pts[..., 1], np.inf).min(axis=1)
        need = np.minimum(np.where(np.isfinite(low), low, CREEPER_TOP_M) - CREEPER_TOP_M, 0.0)
        y[idx] = np.minimum(gaussian_filter1d(need, 10.0, mode="nearest"), need)
    return np.column_stack([mid[:, 0], y, mid[:, 2], yaw])
