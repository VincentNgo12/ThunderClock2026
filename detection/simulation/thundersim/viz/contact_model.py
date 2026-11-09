# File: contact_model.py
# Purpose: Shared geometry for the viewer's contact and balance pass (contact.py) and its gates
#          (contact_gates.py): held objects and their handholds, the drawn body as spheres, the
#          support under the feet, the support polygon, the centre of mass and joint angles.
# Contributors: Unassigned
"""Contact model of the drawn worker. Render only; nothing here touches the simulated spine.

Held objects and their handholds (all [design] unless a source is named):
  - sack (A-CARRY): a 0.32 x 0.55 x 0.16 m box (scene.BAG_SIZE_M). Each hand holds the sack's side
    at LOAD_HALF_WIDTH_M across (the hand centre 2 cm outside the 0.16 m half width). The grip height
    is the simulated one at each end: the lift's 'object_height' (event lift_grasp) from the grasp
    through the rise, and the put-down's (put_down_touch) from the re-grip to the release, so the
    simulated postures reach the sack on near-straight arms at both ends. The re-grip comes right
    after the rise (Sack.regrip): one hand, then the other, slides along the sack's side to the
    put-down grip, as the rig's carry pose already hangs the sack at the put-down grip.
  - ladder rails (climbing): each hand grips its rail on the rig's grip line (RAIL_GRIP_LAT across,
    RAIL_GRIP_BACK behind the rail centre line) at a fixed height; a hand moves only to re-grip.
  - carried stepladder (A-LADDER): the right hand holds the upper rail at mid length (the pose
    scene.stepladder_motion draws).
  - hammer, drill, jackhammer: drawn on the hand joints by body.js, so they are rigid with the
    hands by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..spine import PHASES
from .geom import UP, fwd, right, smoothstep, unit
from .physics import JOINT_MASS, TRUNK_M, support_points, torso_centroid
from .rig import LOAD_HALF_WIDTH_M
from .scene import BAG_SIZE_M, body_scale, carry_times, spine_pose, torso_rings

# --- frames that the pass and the gates look at ----------------------------------------------
# Ground-truth phases in which the worker is on his feet or kneeling by choice (not falling,
# lying, hanging or in a near fall). Frames in the ragdoll's windows (physics.py) are excluded too.
ACTIVE_PHASES = tuple(code for code, name in PHASES.items()
                      if name in ("activity", "calibration_stand", "recovery", "device_off"))

# --- held objects [design] --------------------------------------------------------------------
SACK_HALF = np.array([BAG_SIZE_M[0] / 2, BAG_SIZE_M[1] / 2, BAG_SIZE_M[2] / 2])   # across, up, depth
SACK_GRIP_MAX_M = BAG_SIZE_M[1] - 0.03       # a grip at most this high above the sack's bottom
REGRIP_HAND_S = 0.5                          # each hand's slide to the put-down grip after the rise
RAIL_GRIP_LAT = 0.22                         # hand centre across from the ladder centre line (rig.py)
RAIL_GRIP_BACK = 0.03                        # ... and this far behind the rail centre line (rig.py)
RAIL_SLOPE = 0.25                            # ladder rails rise 4:1 (scene.Ladder, world.js)
STEPLADDER_KG = 6.0                          # aluminium stepladder of 3-4 steps [design]

# --- drawn body (body.js) -------------------------------------------------------------------
# (joint a, joint b, radius at a / s, radius at b / s, scaled by w at a): tapered limbs as drawn.
LIMB_TAPER = {
    "thigh_l": (0, 2, 0.084, 0.058, True), "thigh_r": (1, 3, 0.084, 0.058, True),
    "shank_l": (2, 4, 0.057, 0.043, False), "shank_r": (3, 5, 0.057, 0.043, False),
}
BOOT_HALF_W = 0.0525          # boot half width / s (body.js 0.105 s)
BOOT_PAD = 0.015              # sole extends this far past the heel and toe points (body.js +0.03)
CONTACT_H_M = 0.015           # a sole corner this close above its support carries weight

# --- joint limits (.claude/agents/mocap-curator.md; as payload_gates.py) ------------------------
KNEE_RANGE = (0.0, 160.0)
ELBOW_RANGE = (0.0, 155.0)
HIP_RANGE = (-30.0, 135.0)
HIP_IN_PLANE = 0.5            # hip flexion is judged where the thigh lies mostly in the sagittal plane


@dataclass
class Hold:
    """One hand holding one object over [k0, k1) (output frames)."""

    obj: str                  # "sack0", "rails", "stepladder"
    side: int                 # 0 left, 1 right
    k0: int
    k1: int
    points: np.ndarray        # (k1 - k0, 3) handhold position per frame
    resting: np.ndarray       # (k1 - k0,) bool: the object rests on a support (cannot move)
    label: str = ""


@dataclass
class Sack:
    """One carried sack: rest on the floor, held, set down. Times in s, frames are output frames."""

    index: int
    t_grasp: float
    t_off: float
    t_touch: float
    t_release: float
    grip_pick: float           # handhold height above the sack's bottom at the lift (m)
    grip_drop: float           # ... at the put-down (m)
    t_regrip: float            # the left hand slides to grip_drop from here, then the right hand
    mass: float
    pick: np.ndarray           # sack centre resting at the pick (3,)
    drop: np.ndarray           # sack centre resting at the put-down (3,)
    yaw_pick: float
    yaw_drop: float
    extra: dict = field(default_factory=dict)

    def grips(self, t: np.ndarray) -> np.ndarray:
        """Handhold height above the sack's bottom (n, 2), left then right, at times t (s):
        grip_pick, then each hand in turn moves smoothly to grip_drop (REGRIP_HAND_S each)."""
        t = np.atleast_1d(np.asarray(t, float))
        out = np.zeros((len(t), 2))
        for side in (0, 1):
            start = self.t_regrip + side * REGRIP_HAND_S
            w = smoothstep((t - start) / REGRIP_HAND_S)
            out[:, side] = self.grip_pick + (self.grip_drop - self.grip_pick) * w
        return out


def sack_plan(ctx, scene) -> list:
    """The carried sacks of a run, from the lift and put-down events (motions.Builder.lift /
    put_down). The sack stands on the support at the simulated grip point (gx, gz)."""
    sacks = []
    for k, (call, _, _, _) in enumerate(carry_times(ctx)):
        found = {}
        for t, name, info in ctx.events:
            if call.t0 - 1e-6 <= t <= call.t1 + 1e-6 and name not in found:
                found[name] = (float(t), info)
        names = ("lift_grasp", "lift_off", "lift_end", "put_down_touch", "put_down_release")
        if not all(n in found for n in names):
            continue
        g_pick, g_drop = found["lift_grasp"][1], found["put_down_touch"][1]
        grip_pick = float(min(g_pick["object_height"], SACK_GRIP_MAX_M))
        grip_drop = float(min(g_drop["object_height"], SACK_GRIP_MAX_M))
        # Re-grip once upright (lift_end); a short carry re-grips earlier so that it is done 2 s
        # before the touchdown, i.e. before the put-down's bend.
        t_end, t_touch = found["lift_end"][0], found["put_down_touch"][0]
        t_regrip = t_end
        if t_regrip + 2 * REGRIP_HAND_S > t_touch - 2.0:
            t_regrip = max(found["lift_off"][0], t_touch - 2.0 - 2 * REGRIP_HAND_S)

        def rest(info):
            p = np.array([info["gx"], 0.0, info["gz"]], float)
            p[1] = scene.support(p[0], p[2], float(info["gy"]) + 0.3) + SACK_HALF[1]
            return p

        sacks.append(Sack(k, found["lift_grasp"][0], found["lift_off"][0], found["put_down_touch"][0],
                          found["put_down_release"][0], grip_pick, grip_drop, t_regrip,
                          float(call.args.get("load_kg", 20.0)),
                          rest(g_pick), rest(g_drop), ctx.heading(found["lift_grasp"][0]),
                          ctx.heading(found["put_down_touch"][0])))
    return sacks


def sack_axes(yaw: np.ndarray) -> np.ndarray:
    """Columns: across (right of the heading), up, depth (backwards) of a sack facing `yaw` (N, 3, 3)."""
    yaw = np.atleast_1d(np.asarray(yaw, float))
    R = np.zeros((len(yaw), 3, 3))
    R[:, :, 0] = right(yaw)
    R[:, :, 1] = UP
    R[:, :, 2] = -fwd(yaw)
    return R


def sack_handholds(centre: np.ndarray, yaw: np.ndarray, grip_h) -> np.ndarray:
    """World handholds (N, 2, 3), left then right, of a sack at centre (N, 3) facing yaw (N,);
    grip_h is the handhold height above the sack's bottom: one value, or (N, 2) per frame and hand."""
    R = sack_axes(yaw)
    grip = np.broadcast_to(np.asarray(grip_h, float), (len(centre), 2))
    out = np.zeros((len(centre), 2, 3))
    for s, sign in enumerate((-1.0, 1.0)):
        local = np.zeros((len(centre), 3))
        local[:, 0] = sign * LOAD_HALF_WIDTH_M
        local[:, 1] = grip[:, s] - SACK_HALF[1]
        out[:, s] = centre + np.einsum("nij,nj->ni", R, local)
    return out


def box_signed_distance(points: np.ndarray, centre: np.ndarray, R: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Signed distance (negative inside) of points (N, S, 3) to boxes centre (N, 3), axes R (N, 3, 3)."""
    local = np.einsum("nji,nsj->nsi", R, points - centre[:, None, :])
    q = np.abs(local) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(q.max(axis=-1), 0.0)
    return outside + inside


# --- ladders --------------------------------------------------------------------------------
def rail_axis(lad) -> np.ndarray:
    return unit(fwd(lad.yaw) * RAIL_SLOPE + UP)


def rail_point(lad, side: int, y) -> np.ndarray:
    """Point on a hand's grip line on the rail at height y (scalar or (n,))."""
    y = np.asarray(y, float)
    lf, lr = fwd(lad.yaw), right(lad.yaw)
    sign = -1.0 if side == 0 else 1.0
    base = lad.base + lr * (sign * RAIL_GRIP_LAT) - lf * RAIL_GRIP_BACK
    along = (y - lad.base[1]) * RAIL_SLOPE
    p = base[None, :] + lf[None, :] * np.atleast_1d(along)[:, None]
    p[:, 1] = np.atleast_1d(y)
    return p if np.ndim(y) else p[0]


def rail_distance(lad, side: int, points: np.ndarray) -> np.ndarray:
    """Distance (n,) of points (n, 3) from a hand's grip line on the rail."""
    p0 = rail_point(lad, side, lad.base[1])
    d = rail_axis(lad)
    rel = points - p0
    return np.linalg.norm(rel - np.outer(rel @ d, d), axis=1)


def stepladder_handhold(pos: np.ndarray, yaw: np.ndarray, opened: np.ndarray, height: float,
                        along: np.ndarray | None = None) -> np.ndarray:
    """Right-hand handhold (N, 3) of the drawn stepladder at pose (pos, yaw, open): the point of its
    upper rail at `along` (N,) of its length (default mid length, where scene.stepladder_motion
    carries it), wherever the ladder is."""
    along = np.full(len(pos), 0.5) if along is None else along
    return pos + np.array([handhold_offset(o, y, stepladder_grip_local(height, a), height)
                           for o, y, a in zip(opened, yaw, along, strict=True)])


# --- stepladder geometry (world.js stepladderMesh and setOpen) -----------------------------------
def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _front_chain(opened: float, height: float):
    """(rotation, offset) of the stepladder's front frame relative to its yawed origin."""
    a = np.arctan2(1, 4)
    length = height / np.cos(a)
    roll = _rz((1 - opened) * np.pi / 2)
    rx = _rx(-a * opened + (1 - opened) * (-np.pi / 2 + 0.06))
    pos = np.array([0.0, 0.0, (1 - opened) * length / 2])
    return roll @ rx, roll @ pos


def stepladder_grip_local(height: float, along: float = 0.5) -> np.ndarray:
    """The right-hand handhold in the stepladder's front frame: on its upper rail where
    scene.stepladder_motion puts the hand on the carried ladder (0.21 m above the drawn origin,
    0.03 L to its left: mid length), moved to `along` (fraction of the rail length)."""
    length = height / np.cos(np.arctan2(1, 4))
    R, off = _front_chain(0.0, height)
    local = np.array([-0.03 * length, 0.21, 0.0])
    return R.T @ (local - off) + np.array([0.0, (along - 0.5) * length, 0.0])


def handhold_offset(opened: float, yaw: float, v: np.ndarray, height: float) -> np.ndarray:
    """World offset from the stepladder's drawn origin to its handhold, for a pose (open, yaw)."""
    R, off = _front_chain(opened, height)
    return _ry(yaw) @ (off + R @ v)


def base_handhold(base: np.ndarray, yaw: float, v: np.ndarray, height: float) -> np.ndarray:
    return base + handhold_offset(1.0, yaw, v, height)


# --- drawn body as spheres ------------------------------------------------------------------
def leg_spheres(joints: np.ndarray, s: float, w: float, parts=("thigh_l", "thigh_r", "shank_l", "shank_r"),
                step_m: float = 0.02) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spheres along the drawn tapered thighs and shanks: centres (N, S, 3), radii (S,) and the
    part index (S,) into `parts`."""
    centres, radii, part = [], [], []
    for p, name in enumerate(parts):
        a, b, r0, r1, wide = LIMB_TAPER[name]
        length = float(np.median(np.linalg.norm(joints[:, b] - joints[:, a], axis=1)))
        u = np.linspace(0.0, 1.0, max(int(np.ceil(length / step_m)) + 1, 2))
        ra, rb = r0 * s * (w if wide else 1.0), r1 * s
        centres.append(joints[:, a, None, :] + (joints[:, b] - joints[:, a])[:, None, :] * u[None, :, None])
        radii.append(ra + (rb - ra) * u)
        part.append(np.full(len(u), p))
    return np.concatenate(centres, axis=1), np.concatenate(radii), np.concatenate(part)


def torso_spheres(ctx, spine: np.ndarray, frames: np.ndarray, step_m: float = 0.03) -> tuple[np.ndarray, np.ndarray]:
    """Spheres filling the drawn torso: for each body.js ring, a row of spheres across its width
    (radius = half depth), joined along the spine. Centres (N, S, 3), radii (S,)."""
    s, w = body_scale(ctx.worker)
    _, M = spine_pose(ctx, frames)
    rings = torso_rings(spine, M, ctx.H, s, w)
    centres, radii = [], []
    for i in range(len(rings) - 1):
        (ci, xi, _, ai, bi), (cj, xj, _, aj, bj) = rings[i], rings[i + 1]
        seg = float(np.median(np.linalg.norm(cj - ci, axis=1)))
        for u in np.linspace(0.0, 1.0, max(int(np.ceil(seg / step_m)) + 1, 2))[:-1]:
            c = ci + (cj - ci) * u
            x = xi + (xj - xi) * u
            a, b = ai + (aj - ai) * u, bi + (bj - bi) * u
            half = max(a - b, 0.0)
            for v in np.linspace(-1.0, 1.0, max(int(np.ceil(2 * half / step_m)) + 1, 2)):
                centres.append(c + x * (v * half))
                radii.append(b)
    return np.stack(centres, axis=1), np.array(radii)


# --- support ----------------------------------------------------------------------------------
def rung_tops(scene, t: np.ndarray) -> list:
    """Rung tread rectangles while each ladder stands: (k0 mask (N,), centre (3,), lateral, fwd, half sizes)."""
    out = []
    for lad in scene.ladders:
        t_end = lad.t_down[1] if lad.t_down else np.inf
        standing = (t >= lad.t_up[0] - 1.0) & (t <= t_end + 0.3)
        for k in range(1, lad.rungs + 1):
            top = lad.rung(k).copy()
            if lad.kind != "steps":
                top[1] += 0.03
            out.append((standing, top, right(lad.yaw), fwd(lad.yaw), (0.25, 0.13)))
    return out


def support_under(scene, points: np.ndarray, t: np.ndarray, rungs: list | None = None) -> np.ndarray:
    """Support height under points (N, S, 3): walkables and the ground, plus ladder rungs while
    the ladder stands, counting only surfaces at most 5 cm above the point."""
    below = points[..., 1] + 0.05
    best = support_points(scene, points[..., 0], points[..., 2], below)
    for standing, top, lr, lf, (hr, hf) in rungs if rungs is not None else rung_tops(scene, t):
        rel = points - top
        a = rel @ lr
        b = rel @ lf
        on = (np.abs(a) <= hr) & (np.abs(b) <= hf) & (top[1] <= below) & (top[1] > best) & standing[:, None]
        best = np.where(on, top[1], best)
    return best


def sole_corners(joints: np.ndarray, s: float) -> np.ndarray:
    """Corners of both drawn soles (N, 2, 4, 3): heel-left, heel-right, toe-left, toe-right."""
    out = np.zeros((len(joints), 2, 4, 3))
    for side in (0, 1):
        heel, toe, ankle = joints[:, 6 + side], joints[:, 8 + side], joints[:, 4 + side]
        along = unit(toe - heel)
        up = unit(np.cross(np.cross(along, ankle - heel), along))
        across = unit(np.cross(along, up))
        h0, t0 = heel - along * BOOT_PAD, toe + along * BOOT_PAD
        half = across * (BOOT_HALF_W * s)
        out[:, side] = np.stack([h0 - half, h0 + half, t0 - half, t0 + half], axis=1)
    return out


def support_polygon_margin(points_xz: list, com_xz: np.ndarray) -> float:
    """Signed distance (m, + inside) of a point from the convex hull of contact points (xz)."""
    pts = np.unique(np.round(np.asarray(points_xz, float), 6), axis=0)
    if len(pts) == 0:
        return -np.inf
    if len(pts) < 3:
        if len(pts) == 1:
            return -float(np.linalg.norm(com_xz - pts[0]))
        a, b = pts
        u = float(np.clip((com_xz - a) @ (b - a) / max(float((b - a) @ (b - a)), 1e-12), 0, 1))
        return -float(np.linalg.norm(com_xz - (a + u * (b - a))))
    hull = convex_hull(pts)
    if len(hull) < 3:
        return support_polygon_margin(hull[[0, -1]], com_xz)
    inside = True
    best = np.inf
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        e = b - a
        cross = e[0] * (com_xz[1] - a[1]) - e[1] * (com_xz[0] - a[0])
        if cross < 0:
            inside = False
        u = float(np.clip((com_xz - a) @ e / max(float(e @ e), 1e-12), 0, 1))
        best = min(best, float(np.linalg.norm(com_xz - (a + u * e))))
    return best if inside else -best


def convex_hull(pts: np.ndarray) -> np.ndarray:
    """Counter-clockwise convex hull (Andrew's monotone chain) of 2D points."""
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(points):
        chain = []
        for p in points:
            while len(chain) >= 2:
                o, a = chain[-2], chain[-1]
                if (a[0] - o[0]) * (p[1] - o[1]) - (a[1] - o[1]) * (p[0] - o[0]) > 1e-12:
                    break
                chain.pop()
            chain.append(p)
        return chain
    lower, upper = half(pts), half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def contact_points(scene, joints: np.ndarray, t: np.ndarray, s: float, rungs: list | None = None) -> tuple:
    """Candidate weight-bearing points (N, P, 3) and which of them bear weight (N, P) per frame:
    both faces of each drawn boot (the sole, and the boot's upper face when the foot lies sole up,
    kneeling) within CONTACT_H_M of their support; the knee's footprint (4 points 3 cm around
    the knee) where the drawn knee touches the floor; the shin where it lies on it; and hands
    within 3 cm of a support (hands on the floor)."""
    from .scene import BOOT_R

    rungs = rung_tops(scene, t) if rungs is None else rungs
    n = len(joints)
    corners = sole_corners(joints, s)                           # (N, 2, 4, 3) sole face
    pts, lift = [], []
    for side in (0, 1):
        heel, toe, ankle = joints[:, 6 + side], joints[:, 8 + side], joints[:, 4 + side]
        along = unit(toe - heel)
        up = unit(np.cross(np.cross(along, ankle - heel), along))  # sole -> ankle side
        for c in range(4):
            pts += [corners[:, side, c], corners[:, side, c] + up * (2 * BOOT_R * s)]
            lift += [0.0, 0.0]
    knee_r = LIMB_TAPER["shank_l"][2] * s
    for side in (0, 1):
        knee, ankle = joints[:, 2 + side], joints[:, 4 + side]
        shin = unit(ankle - knee)
        across = unit(np.cross(shin, UP))
        ahead = unit(np.cross(UP, across))
        for off in (ahead * 0.03, -ahead * 0.03, across * 0.03, -across * 0.03):
            pts.append(knee + off)
            lift.append(knee_r)
        for u in (0.33, 0.66):
            pts.append(knee + (ankle - knee) * u)
            lift.append(knee_r + (LIMB_TAPER["shank_l"][3] * s - knee_r) * u)
    pts += [joints[:, 14], joints[:, 15]]
    lift += [0.0, 0.0]
    P = np.stack(pts, axis=1)
    lift = np.array(lift)
    tol = np.r_[np.full(16, CONTACT_H_M), np.full(len(lift) - 18, 0.02), np.full(2, 0.03)]
    height = P[..., 1] - lift[None, :] - support_under(scene, P, t, rungs)
    knees = slice(16, len(lift) - 2)                   # knees and shins rest on floors, not on rungs
    height[:, knees] = P[:, knees, 1] - lift[None, knees] - support_under(scene, P[:, knees], t, [])
    return P, height <= tol[None, :]


# --- centre of mass -----------------------------------------------------------------------------
def body_com(ctx, joints: np.ndarray, spine: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Whole-body centre of mass of the drawn figure (N, 3), de Leva segment masses (physics.py)."""
    s, w = body_scale(ctx.worker)
    _, M = spine_pose(ctx, frames)
    return TRUNK_M * torso_centroid(spine, M, ctx.H, s, w) + np.einsum("j,njk->nk", JOINT_MASS, joints)


# --- joint angles -------------------------------------------------------------------------------
def inner_angle_deg(a, mid, b) -> np.ndarray:
    u, v = a - mid, b - mid
    cos = np.sum(u * v, axis=-1) / np.maximum(np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1), 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def joint_angles(joints: np.ndarray, pelvis_R: np.ndarray) -> dict:
    """Knee and elbow flexion (N, 2) in deg (0 = straight), knee hyperextension (N, 2) bool and hip
    flexion (N, 2) relative to the pelvis segment (NaN where the thigh is mostly out of the
    sagittal plane), as payload_gates.py measures them."""
    hips, knee, ankle = joints[:, 0:2], joints[:, 2:4], joints[:, 4:6]
    sh, el, hand = joints[:, 10:12], joints[:, 12:14], joints[:, 14:16]
    knee_flex = 180.0 - inner_angle_deg(hips, knee, ankle)
    elbow_flex = 180.0 - inner_angle_deg(sh, el, hand)
    axis = unit(np.cross(knee - hips, ankle - knee))
    hyper = (np.sum(axis * pelvis_R[:, None, :, 0], axis=-1) > 0.5) & (knee_flex > 5.0)
    thigh = np.einsum("nji,nsj->nsi", pelvis_R, knee - hips)
    hip = np.degrees(np.arctan2(-thigh[..., 2], -thigh[..., 1]))
    hip = np.where(hip < -90.0, hip + 360.0, hip)
    in_plane = np.hypot(thigh[..., 1], thigh[..., 2]) >= HIP_IN_PLANE * np.linalg.norm(thigh, axis=-1)
    return {"knee": knee_flex, "elbow": elbow_flex, "hyper": hyper, "hip": np.where(in_plane, hip, np.nan)}
