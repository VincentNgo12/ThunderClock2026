# File: rig.py
# Purpose: Full-body pose for the 3D viewer, built around the simulated 4-link spine:
#          hips, legs and feet (footsteps timed to the simulated heel strikes, foot roll,
#          ladder rungs), shoulders, arms and head. Limbs are illustrative; only the spine
#          comes from the simulator.
# Contributors: Unassigned
"""Procedural rig.

Input: the spine kinematics at the output frame rate, the recorded primitive calls, the
ground-truth events and the scene (walkable surfaces, ladders). Output: joint positions for
both legs and arms, the neck and head, plus the head orientation.

Each limb follows a timeline of modes (stand, ladder, kneel, quad, sit, slip, lie, air, hang
for legs; idle, hammer, carry, ladder, brace... for arms). A mode gives a target for every
frame; consecutive modes are blended in time order, so a blend that starts before the previous
one finished still starts from wherever the limb actually was. Targets go through analytic
two-bone IK with a pole vector; the bend plane (where a knee or elbow points) is kept
continuous from frame to frame so joints never flip.

Feet never teleport. In 'stand' mode a foot is either planted (fixed on the support, rolling
over the heel or ball when the hip moves away) or in a swing that runs from its pose at
lift-off to its planted pose at touchdown (at least MIN_SWING_S long). Footsteps come from:
  - the simulated gait: motions.walk() rotates the pelvis by 4 deg * sin(stride phase) and puts
    the heel-strike pulses at the extremes of that rotation, so its extremes give each strike's
    time and side; motions.run() rolls the trunk the same way once per step;
  - jumps: both feet leave at free-fall start and land at free-fall end under the pelvis;
  - otherwise reactive steps with look-ahead on the known pelvis path (idle shuffles, turns,
    stumbles, getting going), and an emergency step before a planted foot goes out of reach.
On ladders each foot moves rung to rung; it leaves a rung before the hip rises out of its
reach and lands only once the next rung is reachable.

With mocap on (mocap.py, off by default) a Session replaces some targets with real clips: walking
footsteps get the clip's toe-off time, swing path, foot roll and knee direction (the feet are still
planned and planted here), and walking, hammering and kneeling-work arms get 'clip' items. Everything
else, and every activity without a listed clip, stays procedural.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation

from .geom import UP, flat, fwd, right, smoothstep, unit, yaw_of

DOWN = -UP
JOINTS = ("hip_l", "hip_r", "knee_l", "knee_r", "ankle_l", "ankle_r", "heel_l", "heel_r", "toe_l", "toe_r",
          "shoulder_l", "shoulder_r", "elbow_l", "elbow_r", "hand_l", "hand_r", "neck", "head")
SIDES = (-1.0, 1.0)  # left, right

# --- tuning (design values; lengths in metres, times in seconds) ------------------------------
REACH = 0.995                     # a planted leg stays within this fraction of its full length
MIN_SWING_S = 0.15                # shortest foot swing to a new support
EMERGENCY_LOOKAHEAD_S = 0.1       # step before a planted foot goes out of reach within this time
STANCE_TRAVEL_M = 0.62            # pelvis travel over one planted foot when walking (at H = 1.75 m)
PLACE_FRACTION = 0.32             # a foot lands under where the hip will be this far into its stance
STEP_SWAY_RAD = np.deg2rad(4.0)   # motions.walk pelvis yaw / motions.run trunk roll per stride
MAX_BEND_STEP_M = 0.05            # a knee or elbow swings about its limb axis by at most this per frame
MIN_JOINT_ANGLE = np.deg2rad(35.0)  # knees and elbows flex to at most 145 deg (inner angle >= 35 deg)
PLANTED_ROLL_STEP = 0.2           # a planted foot's heel/toe roll changes by at most this per frame (rad)
MAX_FOOT_SPEED = 5.25             # an ankle target moves at most this fast relative to the pelvis (m/s)
LANDING_LOG_LAG_S = 0.6           # a jump's new ground level is logged at most this long after touchdown
HEAD_RADIUS = 0.09                # head centre stays this far (x s) above the support (body.js skull)
CONSCIOUS_BLEND_S = 0.5           # head tone follows a consciousness change over this time
LOAD_HALF_WIDTH_M = 0.18          # hand centres this far either side of a lifted load's grip point
LOAD_CLEAR_OF_HIPS_M = 0.24       # a held load's grip stays this far ahead of the hip joints
LOAD_CLEAR_OF_KNEES_M = 0.15      # ... and this far ahead of the knees: half a 0.16 m sack, the knee and 2 cm
CARRY_KNEE_PUSH_MAX_M = 0.15      # most a carry pose moves the hands forward to clear the knees
CARRY_RAISE_MAX_M = 0.25          # ... and up, hugging the load higher, if forward is not enough


@dataclass
class Dims:
    """Body proportions as fractions of height (Drillis & Contini style; design values)."""

    H: float
    mass: float

    def __post_init__(self):
        s = self.H / 1.75
        bmi = self.mass / self.H**2
        w = float(np.clip(np.sqrt(bmi / 24.0), 0.88, 1.25))
        self.s, self.w = s, w
        self.hip = np.array([0.088 * w * s, -0.02 * self.H, -0.075 * s])     # pelvis frame (+Z posterior)
        self.shoulder = np.array([0.172 * w * s, 0.022 * self.H, -0.055 * s])  # upper-thoracic frame
        self.thigh, self.shank = 0.24 * self.H, 0.235 * self.H
        self.leg = self.thigh + self.shank
        self.ankle_h = 0.039 * self.H
        self.ball, self.heel, self.toe = 0.13 * s, 0.055 * s, 0.2 * s
        self.upper_arm, self.forearm = 0.18 * self.H, 0.19 * self.H   # forearm includes half the hand
        self.arm = self.upper_arm + self.forearm
        self.neck = 0.03 * self.H

    def export(self) -> dict:
        return {"H": round(self.H, 3), "s": round(self.s, 3), "w": round(self.w, 3), "thigh": round(self.thigh, 4),
                "shank": round(self.shank, 4), "upper_arm": round(self.upper_arm, 4), "forearm": round(self.forearm, 4),
                "ankle_h": round(self.ankle_h, 4), "neck": round(self.neck, 4)}


@dataclass
class Item:
    t0: float
    t1: float
    mode: str
    info: dict = field(default_factory=dict)


class Timeline:
    def __init__(self, default: str):
        self.items = [Item(0.0, 0.0, default)]

    def add(self, t0, t1, mode, **info) -> None:
        if t0 is None:
            return
        self.items.append(Item(float(t0), float(max(t1, t0 + 1e-3)), mode, info))

    def finalize(self) -> list:
        self.items.sort(key=lambda it: it.t0)
        kept = [self.items[0]]
        for it in self.items[1:]:
            prev = kept[-1]
            same = it.mode == prev.mode and it.info.get("key") == prev.info.get("key") and not it.info.get("restart")
            if not same:
                kept.append(it)
        return kept


@dataclass
class Rig:
    joints: np.ndarray           # (N, 18, 3)
    head_q: np.ndarray           # (N, 4) xyzw
    pelvis: np.ndarray
    spine: np.ndarray
    dring: np.ndarray
    anchor3: np.ndarray
    head: np.ndarray
    hand_l: np.ndarray
    hand_r: np.ndarray
    shoulder_r: np.ndarray
    tool: np.ndarray             # (N,) int: 0 none, 1 hammer, 2 drill, 3 jackhammer
    leg_modes: list
    arm_modes: list


# --- footstep records ---------------------------------------------------------------------------
@dataclass
class Swing:
    """One foot in the air: from its pose at lift-off to its planted pose at touchdown."""

    t0: float                       # lift-off (s)
    t1: float                       # touchdown (s)
    a0: np.ndarray                  # ankle at lift-off (m)
    yaw0: float
    pitch0: float                   # foot pitch, toes up + (rad)
    contact: np.ndarray | None      # sole point on the support at touchdown; None = foot hangs free
    yaw1: float
    a1: np.ndarray                  # ankle at touchdown
    pitch1: float
    lift: float                     # apex of the arc above the straight path (m)
    back: np.ndarray                # horizontal pull-back at mid-swing (m), e.g. away from a ladder
    pivot: str = "auto"             # how the landed foot rolls ("auto" heel or ball, "ball" on rungs)
    kind: str = "step"              # step | jump | land
    shape: object = None            # mocap.SwingShape: the clip's foot path (None = procedural arc)


@dataclass
class Foot:
    """Stepping state of one foot: planted on `contact` (sole point under the ankle) or swinging."""

    contact: np.ndarray | None      # None while hanging free (no support within reach)
    yaw: float
    pivot: str = "auto"
    swing: Swing | None = None
    last_lift: float = -1e9
    pitch: float | None = None      # pitch drawn last frame (rad), for the planted roll rate limit


@dataclass
class Step:
    """A footstep planned from the simulated gait."""

    t_lift: float
    t_land: float                   # = the simulated heel strike (s)
    side: int                       # 0 left, 1 right
    t_place: float                  # the foot lands under where the hip will be at this time
    lift: float                     # swing apex (m)
    shape: object = None            # mocap.SwingShape for a clip-shaped step


# --- small vector helpers --------------------------------------------------------------------
def two_bone(root, target, pole, l1, l2, continuous: bool = False):
    """Analytic two-bone IK (vectorised). Returns (mid joint, end joint).

    The end joint stops short of a target that is out of reach, or closer to the root than the
    joint's flexion limit (MIN_JOINT_ANGLE) allows. continuous=True treats axis 0 as time
    (frames): the bend plane may not flip between frames and swings about the limb axis so that
    the mid joint moves at most MAX_BEND_STEP_M per frame from that swing; where the pole lies
    almost on the limb line the previous plane is kept."""
    d = target - root
    dist = np.linalg.norm(d, axis=-1, keepdims=True)
    direction = d / np.maximum(dist, 1e-9)
    folded = np.sqrt(l1 * l1 + l2 * l2 - 2 * l1 * l2 * np.cos(MIN_JOINT_ANGLE))
    reach = np.clip(dist, max(abs(l1 - l2) + 1e-4, folded), (l1 + l2) * 0.9995)
    cos_a = np.clip((l1 * l1 + reach * reach - l2 * l2) / (2 * l1 * reach), -1, 1)
    sin_a = np.sqrt(1 - cos_a * cos_a)
    p = pole - root
    p_len = np.linalg.norm(p, axis=-1)
    p = p - direction * np.sum(p * direction, axis=-1, keepdims=True)
    fallback = np.cross(direction, np.array([1.0, 0.0, 0.0]))
    p_perp = np.linalg.norm(p, axis=-1)
    p = np.where(p_perp[..., None] < 1e-6, fallback, p)
    bend = unit(p)
    if continuous and bend.shape[0] > 1:
        conditioning = p_perp / np.maximum(p_len, 1e-9)          # sine of the pole-to-limb angle
        bend = continuous_bend(bend, direction, conditioning, l1 * sin_a[..., 0])
    mid = root + direction * (l1 * cos_a) + bend * (l1 * sin_a)
    end = root + direction * reach
    return mid, end


def continuous_bend(bend, direction, conditioning, lever):
    """Bend directions (frames along axis 0, unit, perpendicular to the limb) made continuous.

    Each frame starts from the previous frame's bend (projected onto the new limb's normal
    plane) and turns toward the pole's bend by at most MAX_BEND_STEP_M / lever radians, where
    lever (m) is the mid joint's distance from the root-end line; an ill-conditioned pole
    (conditioning, the sine of the pole-to-limb angle, below 0.2) turns it more slowly still."""
    out = bend.copy()
    n = bend.shape[0]
    trust = np.clip(conditioning / 0.2, 0.15, 1.0)
    limit = np.clip(MAX_BEND_STEP_M / np.maximum(lever, 1e-3), 0.05, np.pi) * trust
    # Frames where following the pole exactly is already within the limits (vectorised check).
    prev = bend[:-1] - direction[1:] * np.sum(bend[:-1] * direction[1:], axis=-1, keepdims=True)
    prev = unit(prev)
    turn = np.arccos(np.clip(np.sum(prev * bend[1:], axis=-1), -1, 1))
    clean = np.all((turn <= limit[1:]) & (trust[1:] >= 1.0), axis=tuple(range(1, turn.ndim)))
    synced = True
    for i in range(1, n):
        if synced and clean[i - 1]:
            continue                                  # out[i] = bend[i] already
        dvec = direction[i]
        carry = out[i - 1] - dvec * np.sum(out[i - 1] * dvec, axis=-1, keepdims=True)
        carry_len = np.linalg.norm(carry, axis=-1, keepdims=True)
        carry = np.where(carry_len > 1e-6, carry / np.maximum(carry_len, 1e-12), bend[i])
        angle = np.arccos(np.clip(np.sum(carry * bend[i], axis=-1), -1, 1))
        step = np.minimum(angle, limit[i])
        side = np.cross(dvec, carry)
        sign = np.where(np.sum(side * bend[i], axis=-1) < 0, -1.0, 1.0)
        out[i] = carry * np.cos(step)[..., None] + side * (sign * np.sin(step))[..., None]
        synced = bool(np.all(np.sum(out[i] * bend[i], axis=-1) > 1.0 - 1e-9))
    return out


def speed_limited(path, max_step: float):
    """Positions (frames along axis 0, .., 3) that follow `path` but move at most `max_step`
    (m) per frame; a point held back catches up at that speed. Normal motion passes through
    unchanged, so this only guards against a target that jumps (e.g. between two modes)."""
    step = np.linalg.norm(np.diff(path, axis=0), axis=-1)
    if not len(step) or step.max() <= max_step:
        return path
    out = path.copy()
    calm = np.all(step <= max_step, axis=tuple(range(1, step.ndim)))
    synced = True
    for i in range(1, len(path)):
        if synced and calm[i - 1]:
            continue                                  # out[i] = path[i] already
        delta = path[i] - out[i - 1]
        dist = np.linalg.norm(delta, axis=-1, keepdims=True)
        out[i] = out[i - 1] + delta * np.minimum(1.0, max_step / np.maximum(dist, 1e-12))
        synced = bool(np.all(dist <= max_step))
    return out


def lerp(a, b, w):
    w = np.asarray(w, float)
    while w.ndim < np.ndim(a):
        w = w[..., None]
    return a + (b - a) * w


def blend_frames(u0, n0, u1, n1, w):
    """Foot orientation blend (frames along axis 0, feet along axis 1): rotate the sole frame
    (u heel->toe, n sole normal) from (u0, n0) toward (u1, n1) by fraction w. A rotation, unlike
    a vector lerp, stays well defined when a foot turns over (standing <-> kneeling)."""
    def frames(u, n):
        u = unit(u)
        n = unit(n - u * np.sum(n * u, axis=-1, keepdims=True))
        return np.stack([u, n, np.cross(u, n)], axis=-1).reshape(-1, 3, 3)
    shape = u0.shape
    r0 = Rotation.from_matrix(frames(u0, n0))
    r1 = Rotation.from_matrix(frames(u1, n1))
    weight = np.broadcast_to(np.asarray(w, float)[:, None], shape[:-1]).reshape(-1)
    turn = (r0.inv() * r1).as_rotvec().reshape(shape)
    # A foot turning over (standing <-> kneeling) goes heel up and over, toes down first.
    first = turn[0]
    angle0 = np.linalg.norm(first, axis=-1)
    toes = np.cross(r0[: shape[1]].apply(first), u0[0])[..., 1]          # toe rise per unit turn
    over = (angle0 > 2.0) & (toes > 0)
    if over.any():
        first[over] -= (2 * np.pi) * first[over] / angle0[over][:, None]
    # A half turn can come out as +pi or -pi about its axis from frame to frame; keep the sense
    # of the previous frame so the foot turns over one way only.
    for i in range(1, shape[0]):
        angle = np.linalg.norm(turn[i], axis=-1)
        flip = (angle > 2.0) & (np.sum(turn[i] * turn[i - 1], axis=-1) < 0)
        if flip.any():
            turn[i][flip] -= (2 * np.pi) * turn[i][flip] / angle[flip][:, None]
    rot = r0 * Rotation.from_rotvec(turn.reshape(-1, 3) * weight[:, None])
    m = rot.as_matrix()
    return m[..., 0].reshape(shape), m[..., 1].reshape(shape)


def swing_profile(u, ramp: float = 0.3):
    """0..1 progress of a swing: speed eases in over the first `ramp` of the time (cosine),
    stays constant, and eases out symmetrically. Zero velocity and acceleration at both ends,
    but the peak speed is only 1 / (1 - ramp) times the mean (minimum jerk: 1.875), so
    a long stride does not whip the foot."""
    u = float(np.clip(u, 0.0, 1.0))
    top = 1.0 / (1.0 - ramp)

    def start(x):
        return top * (x / 2 - ramp / (2 * np.pi) * np.sin(np.pi * x / ramp))
    if u < ramp:
        return start(u)
    if u > 1.0 - ramp:
        return 1.0 - start(1.0 - u)
    return top * (ramp / 2 + u - ramp)


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# --- the rig builder ---------------------------------------------------------------------------
class RigBuilder:
    def __init__(self, ctx, scene, frames: np.ndarray, body_idx: np.ndarray, shift: np.ndarray | None = None,
                 mocap=None):
        self.ctx, self.scene = ctx, scene
        self.frames = frames
        self.t = frames / ctx.fs
        self.dt = float(self.t[1] - self.t[0]) if len(self.t) > 1 else 0.02
        self.N = len(frames)
        self.dims = Dims(ctx.H, ctx.worker.mass_kg)
        kin = ctx.kin
        offset = ctx.worker.sensor_offset_m
        self.M = np.stack([kin.seg_rot[i][body_idx].as_matrix() for i in range(4)], axis=1)  # (N,4,3,3)
        self.P = kin.anchors[body_idx] - np.einsum("nsij,j->nsi", self.M, np.array([0.0, 0.0, offset]))
        if shift is not None:
            self.P = self.P + shift[:, None, :]
        self.heading = ctx.track.heading[body_idx]
        self.f = fwd(self.heading)
        self.r = right(self.heading)
        pel = self.P[:, 0]
        vel = np.gradient(pel, self.dt, axis=0) if self.N > 2 else np.zeros_like(pel)
        k = max(int(round(0.2 / self.dt)), 1)
        kernel = np.ones(k) / k
        self.vel = np.stack([np.convolve(vel[:, j], kernel, mode="same") for j in range(3)], axis=1)
        self.vel[:, 1] = 0.0
        d = self.dims
        self.hips = np.stack([self.P[:, 0] + self.M[:, 0] @ (d.hip * [s, 1, 1]) for s in SIDES], axis=1)
        self.shoulders = np.stack([self.P[:, 3] + self.M[:, 3] @ (d.shoulder * [s, 1, 1]) for s in SIDES], axis=1)
        self.axes = {name: self.M[:, i] for i, name in enumerate(("pelvis", "lumbar", "thoracic", "upper"))}
        self.ground_under_hips = np.array([[scene.support(h[0], h[2], h[1] - 0.25) for h in hp] for hp in self.hips])
        self.sim_ground = self._sim_ground()
        self._stand_snap: dict = {}
        self._ladder_plans: dict = {}
        self.mocap = mocap                 # mocap.Session or None (procedural limbs only)
        if mocap is not None:
            mocap.attach(self)

    def ix(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.t, t), 0, self.N - 1))

    def _sim_ground(self) -> np.ndarray:
        """Simulator support height under the worker per frame (Builder.ground_y, from the
        recording). Entries logged at the same time: the last one wins. Inside a fall the log is
        stale (recorder.py), but stepping is off there. A feet-first landing (falls.
        feet_first_landing: jumps) logs the new level only after its deceleration, so a change
        logged within LANDING_LOG_LAG_S after a free_fall_end applies from the touchdown on (the
        feet land on the level they reach, not on the one they left)."""
        g = np.full(self.N, float(self.scene.ground_y))
        lands = [float(t) for t, name, _ in self.ctx.events if name == "free_fall_end"]
        log = []
        for t_log, y in self.ctx.rec.ground_log:
            touch = [tl for tl in lands if 0.0 < t_log - tl <= LANDING_LOG_LAG_S]
            log.append((min(touch) if touch else t_log, y))
        for t_log, y in sorted(log, key=lambda e: e[0]):
            g[self.t >= t_log - 1e-9] = y
        return g

    def floor_at(self, point, k: int) -> float:
        """Support height for a foot at `point` while the hips are at frame k: the drawn surface
        under it, or the simulator's own floor when the drawing has nothing there (a scene that
        does not cover the walked path must not leave the feet dangling)."""
        hip_y = float(self.hips[k, :, 1].mean())
        drawn = self.scene.support(point[0], point[2], hip_y - 0.3)
        sim = float(self.sim_ground[k])
        if sim > drawn + 0.02 and sim <= hip_y - 0.45:
            return sim
        return drawn

    def home(self, k: int, side: int) -> np.ndarray:
        """Resting sole point of a foot under the hip at frame k (on y = 0)."""
        d = self.dims
        return flat(self.hips[k, side]) + right(self.heading[k]) * (SIDES[side] * (0.1 * d.s - 0.088 * d.w * d.s))

    # ===================================================================================
    # Legs
    # ===================================================================================
    def leg_timeline(self) -> list:
        rec, ctx = self.ctx.rec, self.ctx
        tl = Timeline("stand")
        ladders = self.scene.ladders
        for c in rec.calls:
            n = c.name
            if n == "stand_up":
                tl.add(c.t0, c.t1, "stand")
            elif n in ("walk", "run"):
                tl.add(c.t0, c.t0 + 0.3, "stand")
            elif n == "ladder":
                lad = next((l for l in ladders if abs(l.t_up[0] - c.t0) < 1e-6 or (l.t_down and abs(l.t_down[0] - c.t0) < 1e-6)), None)
                if lad is None:
                    continue
                tl.add(c.t0, c.t0 + 0.8, "ladder", key=id(lad), ladder=lad)
                going_up = abs(lad.t_up[0] - c.t0) < 1e-6
                if (going_up and lad.kind == "steps") or not going_up:
                    tl.add(c.t1, c.t1 + 0.4, "stand", restart=True)
            elif n in ("lift", "put_down"):
                # Feet stay planted (no shuffles) while the hips drop and move back over them.
                tl.add(c.t0, c.t0 + 0.05, "stand", key=("lift", c.index), nostep=True, restart=True)
                tl.add(c.t1, c.t1 + 0.05, "stand", key=("lifted", c.index), restart=True)
            elif n == "kneel":
                tl.add(c.t0, c.t1, "kneel")
            elif n == "quadruped":
                tl.add(c.t0, c.t1, "quad")
            elif n == "crawl":
                tl.add(c.t0, c.t0 + 0.3, "quad", key="crawl", crawl=True)
            elif n == "sit_ground":
                tl.add(c.t0, c.t1, "sitg")
            elif n == "lie":
                nested = [x for x in rec.calls if x.name in ("kneel", "sit_ground", "quadruped") and rec.is_inside(x, c)]
                start = max([x.t1 for x in nested], default=c.t0)
                tl.add(start, c.t1, "lie")
            elif n == "sit":
                vehicle = "cab_ride" in rec.ancestors(c)
                tl.add(c.t0, c.t1, "sit", key=("sit", c.index), floor=c.ground0, seat=float(c.args.get("seat_height", 0.45)),
                       vehicle=vehicle, t_seated=c.t1)
        phase = ctx.track.phase[self.frames]
        t_flight = None
        for t, name, info in ctx.events:
            if name == "fall_onset":
                kind = info.get("kind")
                standing = self._standing(t)
                if kind == "collapse":
                    # Knees buckle with the feet where they stand; then the topple onto the floor.
                    if standing:
                        tl.add(t, t + 0.1, "stand", key="collapse", nostep=True, restart=True)
                    k = int(np.argmax((self.t > t) & (phase == 3))) if np.any((self.t > t) & (phase == 3)) else None
                    if k is not None:
                        tl.add(self.t[k], self.t[k] + 1.0, "lie")
                elif kind == "slip" and standing:
                    tl.add(t, t + 0.15, "slip", t_on=t)        # the feet slide out from under the body
                elif standing:
                    tl.add(t, t + 0.1, "stand", key="fall", nostep=True, restart=True)
            elif name == "near_fall_onset" or name == "struck":
                tl.add(t, t + 0.1, "stand", key="perturb")
            elif name == "free_fall_start":
                t_flight = t
                if info.get("kind") != "jump":                 # jumps are footsteps (stand_feet)
                    tl.add(t, t + 0.25, "air")
            elif name == "free_fall_end":
                if info.get("kind") == "jump":
                    continue
                if info.get("reason") == "feet":
                    # Feet reach the floor exactly at touchdown: the landing swing starts a bit before.
                    t0 = max(t - 0.12, (t_flight if t_flight is not None else t - 0.12) + 0.02)
                    tl.add(t0, t, "stand", key="land", nostep=True, restart=True, landing=True, t_land=t)
                elif info.get("reason") == "ground":
                    tl.add(t, t + 0.35, "lie")
            elif name == "impact" and info.get("mode") != "feet_first" and not info.get("guided"):
                tl.add(t - 0.05, t + 0.45, "lie")
            elif name == "slide_start":
                tl.add(t, t + 0.5, "lie")
            elif name == "arrest_start":
                tl.add(t, t + 0.5, "hang")
        # After a feet-first landing the worker either topples (fall_onset follows) or stands up.
        return tl.finalize()

    def _standing(self, t: float) -> bool:
        i = self.ix(t)
        return self.P[i, 0, 1] - self.ground_under_hips[i].mean() > 0.7 * self.ctx.worker.pelvis_height

    def build_legs(self):
        d = self.dims
        N = self.N
        items = self.leg_timeline()
        self._gait = self.gait_plan()
        ankle = np.zeros((N, 2, 3))
        sole_u = np.zeros((N, 2, 3))
        sole_n = np.zeros((N, 2, 3))
        pole = np.zeros((N, 2, 3))
        toe_bend = np.zeros((N, 2))      # toes bent up at the ball (rad); stand mode sets it
        planted = np.zeros((N, 2), bool)  # stand mode: foot planted on its support (for the gates)
        # Plant record of stand mode (for contact.py): sole point under the ankle when flat, yaw,
        # pitch drawn and pivot ("auto" / "ball") of each planted foot; NaN where not planted.
        plant_contact = np.full((N, 2, 3), np.nan)
        plant_yaw = np.full((N, 2), np.nan)
        plant_pitch = np.full((N, 2), np.nan)
        plant_ball = np.zeros((N, 2), bool)
        # item 0: standing from the start, feet under the hips
        for k, item in enumerate(items):
            k0 = self.ix(item.t0) if k > 0 else 0
            end_t = items[k + 1].t1 if k + 1 < len(items) else self.t[-1] + 1
            k1 = min(self.ix(end_t) + 1, N) if end_t <= self.t[-1] else N
            if k0 >= N or k1 <= k0:
                continue
            sl = slice(k0, k1)
            prev_mode = items[k - 1].mode if k > 0 else "stand"
            start = (ankle[k0].copy(), sole_u[k0].copy()) if k > 0 else None
            next_start = self.ix(items[k + 1].t0) if k + 1 < len(items) else None
            self._toe_bend = None
            self._planted_now = None
            self._plant_now = None
            a, u, nrm, p = self.leg_target(item, sl, start, prev_mode, next_start)
            a, u, nrm, p = (np.broadcast_to(x, (k1 - k0, 2, 3)).copy() for x in (a, u, nrm, p))
            bend = self._toe_bend if self._toe_bend is not None else np.zeros((k1 - k0, 2))
            w = smoothstep((self.t[sl] - item.t0) / max(item.t1 - item.t0, 1e-3)) if k > 0 else np.ones(k1 - k0)
            toe_bend[sl] = lerp(toe_bend[sl], bend, w) if k > 0 else bend
            item_planted = self._planted_now if self._planted_now is not None else np.zeros((k1 - k0, 2), bool)
            planted[sl] = np.where((w >= 0.5)[:, None], item_planted, planted[sl])
            if self._plant_now is not None:
                rows = np.arange(k0, k1)[w >= 0.5]
                c, yw, pt, ball = (x[w >= 0.5] for x in self._plant_now)
                plant_contact[rows], plant_yaw[rows], plant_pitch[rows], plant_ball[rows] = c, yw, pt, ball
            else:
                rows = np.arange(k0, k1)[w >= 0.5]
                plant_contact[rows], plant_yaw[rows], plant_pitch[rows], plant_ball[rows] = np.nan, np.nan, np.nan, False
            if k > 0:
                ankle[sl] = lerp(ankle[sl], a, w)
                pole[sl] = lerp(pole[sl], p, w)
                mixing = w < 1.0 - 1e-12                 # frames after the blend take the target as is
                rows = np.arange(k0, k1)[mixing]
                new_u, new_n = u.copy(), nrm.copy()
                if len(rows):
                    new_u[mixing], new_n[mixing] = blend_frames(sole_u[rows], sole_n[rows], u[mixing], nrm[mixing],
                                                                w[mixing])
                sole_u[sl], sole_n[sl] = new_u, new_n
            else:
                ankle[sl], sole_u[sl], sole_n[sl], pole[sl] = a, u, nrm, p
        ankle[..., 1] += self._sole_lift(ankle, sole_u, sole_n, toe_bend)
        ankle = self.P[:, 0, None, :] + speed_limited(ankle - self.P[:, 0, None, :], MAX_FOOT_SPEED * self.dt)
        self.ankle_target = ankle
        knee, ankle_ik = two_bone(self.hips, ankle, pole, d.thigh, d.shank, continuous=True)
        # Feet follow the IK ankle (it moves only when the target is out of reach). On a heel
        # lift the toes bend up at the ball by toe_bend and stay flat on the ground.
        sole = ankle_ik - sole_n * d.ankle_h
        heel = sole - sole_u * d.heel
        ball = sole + sole_u * d.ball
        toe_dir = sole_u * np.cos(toe_bend)[..., None] + sole_n * np.sin(toe_bend)[..., None]
        toe = ball + toe_dir * (d.toe - d.ball)
        self.leg_items = items
        self.planted = planted
        self.ankle, self.knee, self.heel, self.toe = ankle_ik, knee, heel, toe
        # Kept for the contact pass (contact.py), which re-poses feet from these.
        self.sole_u, self.sole_n, self.toe_bend, self.leg_pole = sole_u, sole_n, toe_bend, pole
        self.plant = {"contact": plant_contact, "yaw": plant_yaw, "pitch": plant_pitch, "ball": plant_ball}

    def _sole_lift(self, ankle, sole_u, sole_n, toe_bend) -> np.ndarray:
        """Upward correction (N, 2) of the ankle targets so the heel, ball and toe tip stay on or
        above the support just under the foot (a turning-over foot rises onto its toes rather
        than sweeping through the floor). Only surfaces at most 5 cm above the ankle count, so
        a foot never pops up onto a box it is beside."""
        d = self.dims
        sole = ankle - sole_n * d.ankle_h
        ball = sole + sole_u * d.ball
        toe_dir = sole_u * np.cos(toe_bend)[..., None] + sole_n * np.sin(toe_bend)[..., None]
        points = np.stack([sole - sole_u * d.heel, ball, ball + toe_dir * (d.toe - d.ball)], axis=2)
        lowest = points[..., 1].min(axis=2)
        lift = np.zeros(lowest.shape)
        near = lowest < self.ground_under_hips + 0.3         # nothing to check high above the floor
        for k, s in zip(*np.nonzero(near), strict=True):
            support = self.scene.support(ankle[k, s, 0], ankle[k, s, 2], ankle[k, s, 1] + 0.05)
            lift[k, s] = max(support - lowest[k, s], 0.0)
        return lift

    def leg_target(self, item: Item, sl: slice, start, prev_mode: str, next_start):
        """Targets for one leg item over frames `sl`: ankle (n,2,3), sole direction u and normal n,
        and the knee pole. `start` = (ankles, sole directions) where the previous items left the
        feet at the item's first frame; next_start = first frame of the following item."""
        mode = item.mode
        if mode == "stand":
            return self.stand_feet(item, sl, start, prev_mode, next_start)
        if mode == "ladder":
            return self.ladder_feet(item.info["ladder"], sl)
        if mode == "slip":
            return self.slip_feet(item, sl, start)
        hips = self.hips[sl]
        n = hips.shape[0]
        f = self.f[sl][:, None, :]
        r = self.r[sl][:, None, :]
        side = np.array(SIDES)[None, :, None]
        X0, Y0, Z0 = (self.M[sl, 0, :, j][:, None, :] for j in range(3))
        anterior = -Z0
        d = self.dims
        ground = self.ground_under_hips[sl][..., None]
        up = np.broadcast_to(UP, (n, 2, 3))
        if mode in ("kneel", "quad"):
            # Forward along the heading: the pelvis axis is useless here when it points down.
            fb = np.broadcast_to(f, (n, 2, 3))
            crawl = item.info.get("crawl")
            knee = flat(hips) + fb * 0.04 + r * side * 0.02
            if crawl:
                dist = np.cumsum(np.linalg.norm(self.vel[sl], axis=1)) * self.dt
                phase = np.pi * dist / 0.22
                knee = knee + f * (0.07 * np.sin(phase[:, None, None] + (side > 0) * np.pi))
            knee_y = ground + 0.06
            ank = knee - fb * (d.shank * 0.93)
            ank[..., 1] = (ground + 0.07)[..., 0]
            knee_pt = knee.copy()
            knee_pt[..., 1] = knee_y[..., 0]
            p = knee_pt + fb * 0.6
            u = -fb
            nrm = np.broadcast_to(DOWN, (n, 2, 3)).copy()
            return ank, u.copy(), nrm, p
        if mode == "sit":
            floor = np.full(n, float(item.info["floor"]))
            if item.info.get("vehicle"):
                i_seat = self.ix(item.info["t_seated"])
                floor = floor + self.P[sl, 0, 1] - self.P[i_seat, 0, 1]
            ank = flat(hips) + f * (d.thigh * 0.95) + r * side * 0.04
            ank[..., 1] = floor[:, None] + d.ankle_h
            p = hips + f * 1.0 + UP * 0.6
            return ank, np.broadcast_to(f, (n, 2, 3)).copy(), up.copy(), p
        if mode == "sitg":
            ank = flat(hips) + f * (d.leg * 0.9) + r * side * 0.06
            ank[..., 1] = (ground + 0.06)[..., 0]
            p = hips + UP * 1.0 + f * 0.3
            u = unit(f + UP * 1.4)
            nrm = unit(UP - f * 0.7)
            return ank, u, nrm, p
        if mode == "lie":
            down_body = -Y0
            reach = np.array([0.93, 0.97])[None, :, None] * d.leg
            # Legs along the body, flexed at the hip just enough to stay on the floor.
            rise = ground[..., 0] + d.ankle_h * 0.8 - hips[..., 1]
            ank = hips + self._flex_above(down_body + 0.04 * X0 * side, anterior, reach[..., 0], rise,
                                          np.radians(100.0), np.radians(45.0)) * reach
            u = unit(anterior * 0.75 + down_body * 0.65)
            nrm = unit(Y0 - u * np.sum(Y0 * u, axis=-1, keepdims=True))
            p = hips + anterior * 1.0 + down_body * 0.3
            return ank, u, nrm, p
        if mode == "air":
            wob = np.sin(self.t[sl] * 9.0)[:, None, None] * 0.08 * side
            dirn = unit(-Y0 * 0.65 + DOWN * 0.35 + anterior * wob)
            ank = hips + dirn * (d.leg * 0.92)
            u = unit(anterior * 0.5 - Y0 * 0.85)
            nrm = unit(Y0 - u * np.sum(Y0 * u, axis=-1, keepdims=True))
            return ank, u, nrm, hips + anterior * 1.0
        if mode == "hang":
            ank = hips + unit(DOWN * 0.85 - Y0 * 0.15 + anterior * 0.06) * (d.leg * 0.96)
            u = unit(anterior * 0.6 + DOWN * 0.8)
            nrm = unit(UP - u * np.sum(UP * u, axis=-1, keepdims=True))
            return ank, u, nrm, hips + anterior * 1.0
        raise ValueError(mode)

    @staticmethod
    def _flex_above(base, swing, length, rise, max_angle, back_angle: float = 0.0, rate: float = 0.08):
        """Unit limb directions (frames, .., 3) turned from `base` toward `swing` (its plane of
        motion, e.g. hip flexion or shoulder abduction; negative angles turn away from it, e.g.
        hip extension) by the angle of least size in [-back_angle, max_angle] (rad) that keeps
        the limb's end at least `rise` (m, negative = below) above its root; when no angle in
        range does, the angle that lifts it most. The angle changes by at most `rate` rad per
        frame (axis 0), which keeps it continuous where the best choice jumps."""
        base = unit(base)
        swing = unit(swing - base * np.sum(swing * base, axis=-1, keepdims=True))
        need = rise / np.maximum(length, 1e-9)
        a, b = base[..., 1], swing[..., 1]
        radius = np.maximum(np.hypot(a, b), 1e-9)
        phi = np.arctan2(b, a)                               # angle of the highest direction
        half = np.arccos(np.clip(need / radius, -1.0, 1.0))
        feasible = need <= radius                            # the end clears `need` on [phi-half, phi+half]
        lo, hi = wrap_angle(phi - half), wrap_angle(phi + half)
        up = np.where(feasible & (lo > 0) & (lo <= max_angle), lo, np.inf)        # least flexion
        down = np.where(feasible & (hi < 0) & (hi >= -back_angle), hi, -np.inf)   # least extension
        nearest = np.where(up <= -down, up, down)
        # Otherwise the highest direction in range: phi itself, or the range end nearer to it.
        ends = np.where(np.cos(max_angle - phi) >= np.cos(-back_angle - phi), max_angle, -back_angle)
        best = np.where((phi <= max_angle) & (phi >= -back_angle), phi, ends)
        angle = np.where(a >= need, 0.0, np.where(np.isfinite(nearest), nearest, best))
        for i in range(1, len(angle)):
            angle[i] = np.clip(angle[i], angle[i - 1] - rate, angle[i - 1] + rate)
        return base * np.cos(angle)[..., None] + swing * np.sin(angle)[..., None]

    # --- slip: the feet slide out ahead while the body goes down ---------------------------
    def slip_feet(self, item: Item, sl: slice, start):
        """Slip fall (fall_onset kind=slip): each heel stays on the floor and slides out
        opposite the fall direction, the leg nearly straight, as the hips drop; the foot that was
        further ahead leads. Toes come up as the foot shoots out."""
        d = self.dims
        hips = self.hips[sl]
        n = hips.shape[0]
        t_on = float(item.info["t_on"])
        out_dir = -self._fall_dir(t_on)                       # horizontal, where the feet go
        ahead = [0.0, 0.0] if start is None else [float(flat(start[0][s]) @ out_dir) for s in (0, 1)]
        lead = int(np.argmax(ahead))
        reach = np.where(np.arange(2) == lead, 0.97, 0.9) * d.leg
        pitch = 0.5 * smoothstep((self.t[sl] - t_on) / 0.4)    # toes up (rad)
        r = self.r[sl]
        ank = np.zeros((n, 2, 3))
        u = np.zeros((n, 2, 3))
        nrm = np.zeros((n, 2, 3))
        yaw = yaw_of(out_dir)
        for s in (0, 1):
            ground = self.ground_under_hips[sl, s]
            # Ankle height with the heel on the floor and the toes pitched up.
            rise = d.ankle_h * np.cos(pitch) + d.heel * np.sin(pitch)
            drop = hips[:, s, 1] - (ground + rise)
            horiz = np.sqrt(np.maximum(reach[s] ** 2 - drop ** 2, 0.0))
            ank[:, s] = flat(hips[:, s]) + out_dir * horiz[:, None] + r * (SIDES[s] * 0.04)
            ank[:, s, 1] = ground + rise
            for i in range(n):
                u[i, s], nrm[i, s] = self._foot_axes(yaw, float(pitch[i]))
        poles = hips + out_dir * 0.5 + UP * 1.0
        return ank, u, nrm, poles

    # ===================================================================================
    # Gait plan: footsteps from the simulated stride (walk, run) and jumps
    # ===================================================================================
    def gait_plan(self) -> dict:
        """Planned footsteps (sorted by lift-off), the time windows they cover (reactive
        stepping is off inside them) and the voluntary jumps (t_takeoff, t_touchdown)."""
        steps: list[Step] = []
        windows = []
        for c in self.ctx.rec.calls:
            if c.name not in ("walk", "run"):
                continue
            planned = self._stride_steps(c)
            if planned:
                steps += planned
                windows.append((min(s.t_lift for s in planned) - 0.1, max(s.t_land for s in planned) + 0.25))
        steps.sort(key=lambda s: s.t_lift)
        jumps = []
        for t, name, info in self.ctx.events:
            if name == "free_fall_start" and info.get("kind") == "jump":
                t_end = next((te for te, ne, ie in self.ctx.events if ne == "free_fall_end" and te > t), None)
                if t_end is not None:
                    jumps.append((float(t), float(t_end)))
        return {"steps": steps, "windows": windows, "jumps": jumps}

    def _stride_strikes(self, call):
        """Heel-strike times (s) and sides (0 left, 1 right) of one walk or run call.

        motions.walk: pelvis yaw = path + 4 deg * ramp * sin(theta); its heel-strike pulses sit at
        theta = pi/2 + k pi, the extremes of that yaw (+ = right foot, which is forward when the
        pelvis is turned left). motions.run: trunk roll = 4 deg * ramp * sin(pi c t), stance k
        starts at t = k / c, half a step before each roll extreme (+ = right)."""
        tr = self.ctx.track
        fs = self.ctx.fs
        a = int(round(call.t0 * fs))
        b = min(int(round(call.t1 * fs)), len(tr.root_pos) - 1)
        if b - a < int(0.6 * fs):
            return np.array([]), np.array([], int)
        tt = np.arange(a, b) / fs
        sway = np.unwrap(tr.heading[a:b]) if call.name == "walk" else np.asarray(tr.roll[a:b], float)
        sway = sway - np.polyval(np.polyfit(tt, sway, 1), tt)
        speed = np.linalg.norm(np.gradient(tr.root_pos[a:b][:, [0, 2]], axis=0), axis=1) * fs
        box = max(int(0.1 * fs), 1)
        speed = np.convolve(speed, np.ones(box) / box, mode="same")
        ramp = speed / max(float(speed.max()), 1e-6)
        level = sway / (STEP_SWAY_RAD * np.maximum(ramp, 0.3))
        gap = int(0.18 * fs)
        right_i, _ = find_peaks(level, height=0.5, distance=gap)
        left_i, _ = find_peaks(-level, height=0.5, distance=gap)
        # Extremes while the gait is barely moving are distorted by the ramp; reactive and
        # emergency steps cover the first and last few centimetres instead.
        found = sorted([(int(i), 1) for i in right_i if ramp[i] >= 0.25] +
                       [(int(i), 0) for i in left_i if ramp[i] >= 0.25])
        alternating: list = []
        for i, s in found:                      # keep sides alternating (the stronger of a pair)
            if alternating and alternating[-1][1] == s:
                if abs(level[i]) > abs(level[alternating[-1][0]]):
                    alternating[-1] = (i, s)
                continue
            alternating.append((i, s))
        times = np.array([tt[i] for i, _ in alternating])
        sides = np.array([s for _, s in alternating], int)
        if call.name == "run" and len(times) >= 2:
            times = times - 0.5 * float(np.median(np.diff(times)))
        return times, sides

    def _stride_steps(self, call) -> list:
        """Planned footsteps of one walk or run call: each lands on its heel strike, after a
        swing of the remaining stride time once the stance is limited to what the leg can reach
        (STANCE_TRAVEL_M of pelvis travel over the planted foot)."""
        times, sides = self._stride_strikes(call)
        if len(times) < 2:
            return []
        if self.mocap is not None:
            shaped = self.mocap.walk_steps(self, call, times, sides, Step, PLACE_FRACTION)
            if shaped:
                return shaped
        running = call.name == "run"
        intervals = np.diff(times)
        steps = []
        for j, (t_land, side) in enumerate(zip(times, sides, strict=True)):
            near = intervals[max(j - 1, 0):j + 1]
            t_step = float(np.mean(near))                     # time between consecutive strikes
            cycle = 2.0 * t_step
            speed = float(np.linalg.norm(self.vel[self.ix(t_land)]))
            if running:
                t_stance = 0.65 * t_step
            else:
                t_stance = float(np.clip(STANCE_TRAVEL_M * self.dims.s / max(speed, 0.1), 0.5 * cycle, 0.65 * cycle))
            t_swing = float(np.clip(cycle - t_stance, 0.25, 0.55))
            lift = (0.2 if running else 0.11) * self.dims.s
            steps.append(Step(float(t_land) - t_swing, float(t_land), int(side), float(t_land) + PLACE_FRACTION * t_stance, lift))
        return steps

    # --- standing / walking: planted feet + footsteps -------------------------------------
    def stand_feet(self, item: Item, sl: slice, start, prev_mode: str, next_start):
        """Planted feet and footsteps over frames `sl` (one per-frame pass over both feet).
        Each frame: land finished swings, put hanging feet down, then (unless the item says
        nostep: falls, collapse, landing) take due jumps, planned steps, emergency steps and,
        outside the gait's windows, reactive steps; with nostep, planted feet slide instead.
        The stepping state at the next item's first frame is kept for a following stand item."""
        n = sl.stop - sl.start
        t = self.t[sl]
        out_a = np.zeros((n, 2, 3))
        out_u = np.zeros((n, 2, 3))
        out_n = np.zeros((n, 2, 3))
        bend = np.zeros((n, 2))
        planted = np.zeros((n, 2), bool)
        log_c = np.full((n, 2, 3), np.nan)
        log_yaw = np.full((n, 2), np.nan)
        log_pitch = np.full((n, 2), np.nan)
        log_ball = np.zeros((n, 2), bool)
        poles = self.hips[sl] + self.f[sl][:, None, :] * 1.0 + UP * 0.1
        if self.mocap is not None:
            poles = self.mocap.knee_poles(self, sl, poles)
        nostep = bool(item.info.get("nostep"))
        # Reactive steps wait for the blend into this item, except when it continues stepping.
        gate_t = -np.inf if prev_mode == "stand" else item.t1
        feet = self._stand_start(item, sl, start, prev_mode)
        plan = self._gait
        steps, windows, jumps = plan["steps"], plan["windows"], plan["jumps"]
        p = int(np.searchsorted([s.t_lift for s in steps], t[0] - 1e-9)) if steps else 0
        used: set = set()
        jumped: set = set()
        for i in range(n):
            k = sl.start + i
            ti = float(t[i])
            if next_start is not None and k == next_start:
                self._stand_snap[k] = copy.deepcopy(feet)
            for foot in feet:
                if foot.swing is not None and ti >= foot.swing.t1 - 1e-9:
                    self._land(foot)
            for s, foot in enumerate(feet):
                if foot.contact is None and foot.swing is None:
                    self._find_ground(foot, s, k, ti)
            if not nostep:
                # Jumps: both feet leave together and land under the pelvis at touchdown.
                for j, (t_ff, t_ffe) in enumerate(jumps):
                    if j not in jumped and t_ff - 0.03 <= ti < t_ffe - 0.08:
                        jumped.add(j)
                        k_land = self.ix(t_ffe)
                        for s, foot in enumerate(feet):
                            home = self.home(k_land, s)
                            self._start_swing(foot, s, k, ti, t_ffe, self._ground_point(home, k_land), self.heading[k_land],
                                              0.12 * self.dims.s, kind="jump")
                # Footsteps of the simulated gait, at their lift-off times.
                while p < len(steps) and steps[p].t_lift <= ti:
                    if p not in used:
                        self._take_step(feet, steps[p], k, ti)
                        used.add(p)
                    p += 1
                in_jump = any(t_ff - 0.3 <= ti < t_ffe + 0.45 for t_ff, t_ffe in jumps)
                if not in_jump:
                    self._emergency_steps(feet, k, ti, steps, used, p)
                    scheduled = any(a <= ti <= b for a, b in windows)
                    if not scheduled and ti >= gate_t:
                        self._reactive_step(feet, k, ti)
            else:
                self._drag(feet, k)
            for s, foot in enumerate(feet):
                pos, yaw, pitch = self._foot_pose(foot, s, k, ti)
                out_a[i, s] = pos
                out_u[i, s], out_n[i, s] = self._foot_axes(yaw, pitch)
                bend[i, s] = self._toe_bend_of(foot, pitch, ti)
                planted[i, s] = foot.swing is None and foot.contact is not None
                if planted[i, s]:
                    log_c[i, s], log_yaw[i, s], log_pitch[i, s] = foot.contact, yaw, pitch
                    log_ball[i, s] = foot.pivot == "ball"
                foot.pitch = float(pitch)
        self._toe_bend = bend
        self._planted_now = planted
        self._plant_now = (log_c, log_yaw, log_pitch, log_ball)
        return out_a, out_u, out_n, poles

    def _toe_bend_of(self, foot: Foot, pitch: float, t: float) -> float:
        """Toe bend (rad): a planted foot rolled onto its ball keeps the toes flat on the ground;
        a swinging foot straightens them over the first 40 % of the swing."""
        if foot.swing is None:
            return max(-pitch, 0.0) if foot.contact is not None else 0.0
        sw = foot.swing
        u = (t - sw.t0) / max(sw.t1 - sw.t0, 1e-6)
        return max(-sw.pitch0, 0.0) * float(1.0 - smoothstep(u / 0.4))

    def _stand_start(self, item: Item, sl: slice, start, prev_mode: str) -> list:
        """Stepping state at the first frame of a stand item."""
        k0 = sl.start
        if prev_mode == "stand" and k0 in self._stand_snap:
            return copy.deepcopy(self._stand_snap[k0])          # stepping continues seamlessly
        d = self.dims
        t0 = float(self.t[k0])
        feet = []
        if start is None or prev_mode in ("kneel", "quad", "sit", "sitg", "lie", "slip"):
            # Feet already where they will stand; the item's blend carries them there.
            end_k = self.ix(item.t1)
            for s in (0, 1):
                feet.append(Foot(self._ground_point(self.home(end_k, s), end_k), float(self.heading[end_k])))
            return feet
        ankles, soles = start
        for s in (0, 1):
            a = ankles[s]
            pitch = float(np.arcsin(np.clip(soles[s][1], -1, 1)))
            yaw = yaw_of(flat(soles[s])) if np.linalg.norm(flat(soles[s])) > 0.2 else float(self.heading[k0])
            if prev_mode == "ladder":
                contact = a - UP * d.ankle_h
                g = self.floor_at(a, k0)
                if contact[1] - g < 0.03:
                    contact[1] = g
                feet.append(Foot(contact, yaw, "auto" if contact[1] == g else "ball"))
                continue
            foot = Foot(None, yaw)
            if prev_mode in ("air", "hang") or item.info.get("landing"):
                t_land = float(item.info.get("t_land", max(item.t1, t0 + MIN_SWING_S)))
                k_land = self.ix(t_land)
                self._start_swing(foot, s, k0, t0, max(t_land, t0 + 0.1), self._ground_point(self.home(k_land, s), k_land),
                                  self.heading[k_land], 0.0, pose0=(a, yaw, pitch), kind="land")
            else:
                g = self.floor_at(a, k0)
                if a[1] - d.ankle_h - g < 0.03:
                    foot.contact = np.array([a[0], g, a[2]])
                else:                                            # foot in the air: put it down
                    c = np.array([a[0], g, a[2]])
                    self._start_swing(foot, s, k0, t0, t0 + 0.2, c, yaw, 0.0, pose0=(a, yaw, pitch))
            feet.append(foot)
        return feet

    def _ground_point(self, point, k: int) -> np.ndarray | None:
        """Sole point on the floor below `point` at frame k, or None when that floor is well out
        of reach of the hips (more than 0.25 m beyond a straight leg: the foot then hangs; a hip
        bobbing up while running still lands the foot)."""
        c = flat(point)
        c[1] = self.floor_at(point, k)
        if float(self.hips[k, :, 1].mean()) - c[1] - self.dims.ankle_h > self.dims.leg + 0.25:
            return None
        return c

    def _take_step(self, feet: list, step: Step, k: int, t: float) -> None:
        """Lift the foot of a planned step now; it lands on the simulated heel strike."""
        foot = feet[step.side]
        if foot.swing is not None or foot.contact is None or step.t_land - t < 0.12:
            return
        k_place = self.ix(step.t_place)
        target = self._ground_point(self.home(k_place, step.side), self.ix(step.t_land))
        self._start_swing(foot, step.side, k, t, step.t_land, target, self.heading[k_place], step.lift, shape=step.shape)

    def _emergency_steps(self, feet: list, k: int, t: float, steps: list, used: set, p: int) -> None:
        """A planted foot about to go out of reach. With its planned lift-off close, the toe
        slides a few cm (heel already up) until then; otherwise the next planned step is taken
        early, or an extra step is made. `p` indexes the first planned step not yet due."""
        k_ahead = self.ix(t + EMERGENCY_LOOKAHEAD_S)
        for s, foot in enumerate(feet):
            if foot.swing is not None or foot.contact is None:
                continue
            over_reach = self._reach_excess(foot.contact, foot.yaw, self.hips[k_ahead, s], foot.pivot) > 0.005
            over_hip = (self.mocap is not None and not over_reach
                        and self.mocap.overextended(self, foot, s, k_ahead))
            if not over_reach and not over_hip:
                continue
            nxt = None
            for j in range(max(p - 4, 0), len(steps)):
                st = steps[j]
                if st.t_lift > t + 1.0:
                    break
                if j not in used and st.side == s and st.t_land - t >= MIN_SWING_S:
                    nxt = j
                    break
            if over_hip:
                # Mocap only: a clip-shaped step leaves before the thigh extends past the limit.
                if nxt is not None and steps[nxt].shape is not None and steps[nxt].t_land - t <= 0.8:
                    used.add(nxt)
                    self._take_step(feet, steps[nxt], k, t)
                continue
            # A clip-shaped step lifts early instead: its clip stance can be longer than the
            # simulated hip height lets the rig leg reach, and a sliding toe would read as skate.
            if nxt is not None and steps[nxt].t_lift - t <= 0.3 and steps[nxt].shape is None:
                if self._reach_excess(foot.contact, foot.yaw, self.hips[k, s], foot.pivot) < 0.1:
                    self._drag([foot] if s == 0 else [None, foot], k)
                    continue
            if nxt is not None and steps[nxt].t_land - t <= 0.8:
                used.add(nxt)
                self._take_step(feet, steps[nxt], k, t)
                continue
            t_land = t + 0.22
            k_place = self.ix(t_land + 0.15)
            target = self._ground_point(self.home(k_place, s), self.ix(t_land))
            self._start_swing(foot, s, k, t, t_land, target, self.heading[k_place], 0.1 * self.dims.s)

    def _reactive_step(self, feet: list, k: int, t: float) -> None:
        """Steps that keep the feet under a body moving without a simulated gait: catch-up
        steps when moving (with look-ahead on the known pelvis path) and idle shuffles/turns."""
        d = self.dims
        speed = float(np.linalg.norm(self.vel[k]))
        running = speed > 2.0
        if any(f.swing is not None for f in feet) and not running:
            return
        if any(f.contact is None for f in feet):
            return
        f_step = 2.75 if running else float(np.clip(1.8 * np.sqrt(max(speed, 0.05) / 1.3), 1.4, 2.4))
        cycle = 2.0 / f_step
        t_swing = cycle * (0.55 if running else 0.4)
        t_stance = cycle - t_swing
        candidates = []
        if speed > 0.25:
            vdir = self.vel[k] / speed
            for s, foot in enumerate(feet):
                if foot.swing is not None:
                    continue
                behind = -float((flat(foot.contact) - self.home(k, s)) @ vdir)
                if behind > 0.5 * speed * t_stance:
                    candidates.append((behind, s))
            place = t + t_swing + PLACE_FRACTION * t_stance
            lift = (0.2 if running else 0.11) * d.s
        else:
            t_swing = 0.4
            k_soon = self.ix(t + 0.25)
            for s, foot in enumerate(feet):
                err = float(np.linalg.norm(flat(foot.contact) - self.home(k_soon, s)))
                yaw_err = abs(wrap_angle(foot.yaw - self.heading[k_soon]))
                if (err > 0.17 * d.s or yaw_err > 0.6) and t - foot.last_lift > 0.35:
                    candidates.append((err + yaw_err * 0.2, s))
            place = t + t_swing
            lift = 0.08 * d.s
        if candidates:
            _, s = max(candidates)
            k_place = self.ix(place)
            target = self._ground_point(self.home(k_place, s), self.ix(t + t_swing))
            self._start_swing(feet[s], s, k, t, t + t_swing, target, self.heading[k_place], lift)

    def _drag(self, feet: list, k: int) -> None:
        """No stepping (falls, collapse, landing): a planted foot that the hip pulls out of reach
        slides along its support toward the hip instead of leaving it."""
        for s, foot in enumerate(feet):
            if foot is None or foot.swing is not None or foot.contact is None:
                continue
            for _ in range(3):
                excess = self._reach_excess(foot.contact, foot.yaw, self.hips[k, s], foot.pivot)
                if excess <= 1e-4:
                    break
                toward = flat(self.hips[k, s] - foot.contact)
                dist = float(np.linalg.norm(toward))
                if dist < 1e-6:
                    break
                foot.contact = foot.contact + toward / dist * min(excess, dist)

    def _find_ground(self, foot: Foot, side: int, k: int, t: float) -> None:
        """A hanging foot steps down once a support comes within reach."""
        target = self._ground_point(self.home(k, side), k)
        if target is not None:
            k_land = self.ix(t + 0.2)
            target = self._ground_point(self.home(k_land, side), k_land)
            self._start_swing(foot, side, k, t, t + 0.2, target, self.heading[k_land], 0.0)

    def _start_swing(self, foot: Foot, side: int, k: int, t0: float, t1: float, contact, yaw1, lift: float,
                     pose0=None, back=None, pivot: str = "auto", kind: str = "step", shape=None) -> None:
        """Lift `foot` now (frame k, time t0) from its current pose; it touches down at t1 on
        `contact` (None: it keeps hanging) with yaw1, rolled as the hips at t1 require. A mocap
        `shape` sets the swing path and the roll at touchdown."""
        t1 = max(float(t1), t0 + MIN_SWING_S)
        if pose0 is None:
            pose0 = self._foot_pose(foot, side, k, t0)
        a0, yaw0, pitch0 = pose0
        k1 = self.ix(t1)
        if contact is None:
            a1, pitch1 = self._dangle(k1, side), -0.4
            yaw1 = float(self.heading[k1])
        else:
            a1, pitch1 = self._planted(contact, float(yaw1), self.hips[k1, side], pivot)
            if shape is not None:
                a1, pitch1 = shape.touchdown(self, contact, float(yaw1), k1, a1, pitch1)
            if kind == "step":                             # clear the edge when stepping up
                rise = float(contact[1] + self.dims.ankle_h - a0[1])
                lift = lift + 0.6 * max(rise, 0.0) + 0.1 * max(-rise, 0.0)
        foot.swing = Swing(t0, t1, np.array(a0, float), float(yaw0), float(pitch0), contact, float(yaw1), a1,
                           float(pitch1), float(lift), np.zeros(3) if back is None else back, pivot, kind,
                           shape if contact is not None else None)
        foot.last_lift = t0

    def _land(self, foot: Foot) -> None:
        sw = foot.swing
        foot.contact = None if sw.contact is None else sw.contact.copy()
        foot.yaw = sw.yaw1
        foot.pivot = sw.pivot
        foot.swing = None

    def _foot_pose(self, foot: Foot, side: int, k: int, t: float):
        """(ankle, yaw, pitch) of a foot at frame k / time t."""
        if foot.swing is not None:
            sw = foot.swing
            if sw.shape is not None:
                return sw.shape.pose(self, sw, k, t)
            u = float(np.clip((t - sw.t0) / max(sw.t1 - sw.t0, 1e-6), 0.0, 1.0))
            w = swing_profile(u)
            bump = float(np.sin(np.pi * u))
            pos = sw.a0 + (sw.a1 - sw.a0) * w + UP * (sw.lift * bump) + sw.back * bump
            return pos, sw.yaw0 + float(wrap_angle(sw.yaw1 - sw.yaw0)) * w, sw.pitch0 + (sw.pitch1 - sw.pitch0) * w
        if foot.contact is None:
            return self._dangle(k, side), float(self.heading[k]), -0.4
        ankle, pitch = self._planted(foot.contact, foot.yaw, self.hips[k, side], foot.pivot)
        if self.mocap is not None and foot.pivot == "auto":
            rolled = self.mocap.stance_pitch(k, side, pitch)      # heel strike and heel rise of the clip
            if rolled is not None and rolled != pitch:
                pitch = rolled
                ankle = self._ankle_at(foot.contact, foot.yaw, pitch)
        if foot.pitch is not None and abs(pitch - foot.pitch) > PLANTED_ROLL_STEP:
            # Hip far out of reach (e.g. running): roll toward the needed pitch, do not flip.
            pitch = float(np.clip(pitch, foot.pitch - PLANTED_ROLL_STEP, foot.pitch + PLANTED_ROLL_STEP))
            ankle = self._ankle_at(foot.contact, foot.yaw, pitch)
        return ankle, foot.yaw, pitch

    def _dangle(self, k: int, side: int) -> np.ndarray:
        """Ankle of a foot hanging free under the hip (no support within reach)."""
        dirn = unit(-self.M[k, 0, :, 1] * 0.7 + DOWN * 0.3)
        return self.hips[k, side] + dirn * self.dims.leg * 0.95

    def _foot_axes(self, yaw: float, pitch: float):
        """Sole direction u (heel->toe) and sole normal n for a foot yawed and pitched (toes up +)."""
        f = fwd(yaw)
        u = f * np.cos(pitch) + UP * np.sin(pitch)
        nrm = UP * np.cos(pitch) - f * np.sin(pitch)
        return u, nrm

    BALL_ROLL_MAX, HEEL_ROLL_MAX = 1.1, 0.45     # rad: heel lift over the ball / toe lift over the heel

    def _ankle_at(self, contact, yaw: float, pitch: float) -> np.ndarray:
        """Ankle of a foot on `contact` (sole point under the ankle when flat) pitched by `pitch`
        (toes up +): pitch < 0 rolls over the ball (heel up), pitch > 0 over the heel (toes
        up). Both meet at pitch 0, so the pose is continuous in pitch."""
        d = self.dims
        f = fwd(yaw)
        if pitch <= 0.0:
            a = -pitch
            ball = contact + f * d.ball
            return ball + d.ball * (-f * np.cos(a) + UP * np.sin(a)) + d.ankle_h * (UP * np.cos(a) + f * np.sin(a))
        heel = contact - f * d.heel
        return heel + d.heel * (f * np.cos(pitch) + UP * np.sin(pitch)) + d.ankle_h * (UP * np.cos(pitch) - f * np.sin(pitch))

    def _roll_needed(self, contact, yaw: float, hip, sign: float, limit: float) -> float:
        """Smallest roll (rad, >= 0) over the ball (sign -1) or heel (+1) that brings the ankle
        within `limit` of the hip, or that pivot's largest roll if none does."""
        hi = self.BALL_ROLL_MAX if sign < 0 else self.HEEL_ROLL_MAX
        if np.linalg.norm(hip - self._ankle_at(contact, yaw, sign * hi)) > limit:
            return hi
        lo = 0.0
        for _ in range(10):
            mid = (lo + hi) / 2
            if np.linalg.norm(hip - self._ankle_at(contact, yaw, sign * mid)) > limit:
                lo = mid
            else:
                hi = mid
        return hi

    def _heel_weight(self, contact, yaw: float, hip, pivot: str) -> float:
        """1 when the hip is clearly behind the foot (roll over the heel), 0 when it is ahead or
        the foot stands on a rung (roll over the ball), smooth over +-8 cm in between."""
        if pivot == "ball":
            return 0.0
        ahead = float((hip - contact) @ fwd(yaw))
        return float(smoothstep(0.5 - ahead / 0.16))

    def _planted(self, contact, yaw, hip, pivot: str = "auto"):
        """(ankle, pitch) of a planted foot, rolling over the ball (heel lift) or the heel (toe
        lift) when the hip has moved out of reach of a flat foot."""
        d = self.dims
        flat_ankle = contact + UP * d.ankle_h
        limit = d.leg * REACH
        if np.linalg.norm(hip - flat_ankle) <= limit:
            return flat_ankle, 0.0
        w = self._heel_weight(contact, yaw, hip, pivot)
        heel_roll = self._roll_needed(contact, yaw, hip, 1.0, limit) if w > 0.0 else 0.0
        ball_roll = self._roll_needed(contact, yaw, hip, -1.0, limit) if w < 1.0 else 0.0
        pitch = w * heel_roll - (1.0 - w) * ball_roll
        return self._ankle_at(contact, yaw, pitch), pitch

    def _reach_excess(self, contact, yaw, hip, pivot: str = "auto") -> float:
        """How far (m) the hip is beyond the reach of this planted foot at its largest roll."""
        d = self.dims
        limit = d.leg * REACH
        dist_flat = float(np.linalg.norm(hip - (contact + UP * d.ankle_h)))
        if dist_flat <= limit:
            return 0.0
        w = self._heel_weight(contact, yaw, hip, pivot)
        best = dist_flat
        if w > 0.0:
            best = min(best, float(np.linalg.norm(hip - self._ankle_at(contact, yaw, self.HEEL_ROLL_MAX))))
        if w < 1.0:
            best = min(best, float(np.linalg.norm(hip - self._ankle_at(contact, yaw, -self.BALL_ROLL_MAX))))
        return max(0.0, best - limit)

    # --- ladders -------------------------------------------------------------------------------
    def ladder_levels(self, lad, t: np.ndarray) -> np.ndarray:
        """Rung level of each foot (fractional while moving) and the lift (0..1 bump). Nominal
        schedule, used for the hands; the feet follow ladder_plan()."""
        levels = np.zeros((len(t), 2))
        lift = np.zeros((len(t), 2))

        def climb(t0, t1, start, step):
            seg = (t1 - t0 - 0.8) / lad.rungs
            for k in range(lad.rungs):
                a = t0 + 0.8 + k * seg
                lead = k % 2
                for s, (b0, b1) in ((lead, (a, a + 0.5 * seg)), (1 - lead, (a + 0.5 * seg, a + seg))):
                    u = np.clip((t - b0) / (b1 - b0), 0, 1)
                    active = (t >= b0) & (t < b1)
                    lift[active, s] = np.sin(np.pi * u[active])
                    levels[:, s] = np.where(t >= b0, start + step * (k + smoothstep(u)), levels[:, s])
        levels[:] = 0.0
        climb(lad.t_up[0], lad.t_up[1], 0, 1)
        top = lad.rungs
        if lad.t_down:
            after = t >= lad.t_down[0]
            saved = levels.copy()
            levels[after] = top
            climb_levels = np.full_like(levels, top, dtype=float)
            seg = (lad.t_down[1] - lad.t_down[0] - 0.8) / lad.rungs
            for k in range(lad.rungs):
                a = lad.t_down[0] + 0.8 + k * seg
                lead = k % 2
                for s, (b0, b1) in ((lead, (a, a + 0.5 * seg)), (1 - lead, (a + 0.5 * seg, a + seg))):
                    u = np.clip((t - b0) / (b1 - b0), 0, 1)
                    active = (t >= b0) & (t < b1)
                    lift[active, s] = np.sin(np.pi * u[active])
                    climb_levels[:, s] = np.where(t >= b0, top - (k + smoothstep(u)), climb_levels[:, s])
            levels = np.where(after[:, None], climb_levels, saved)
        return levels, lift

    def rung_contact(self, lad, level: int, side: int) -> np.ndarray:
        """Sole point (under the ankle) of a foot standing on rung `level` (0 = the ground at the
        ladder's foot); the rung itself is 0.07 m ahead of it, under the arch. Steps (blocks)
        have their tread top at the rung height; round rungs are 0.03 m above their line."""
        f, r = fwd(lad.yaw), right(lad.yaw)
        p = lad.rung(level) - f * 0.07 + r * (SIDES[side] * 0.11)
        if level > 0 and lad.kind != "steps":
            p[1] += 0.03                                   # top of the rung
        return p

    def ladder_plan(self, lad) -> list:
        """Per foot: rung stances [(t_land, t_lift, level)] in time order (the first from -inf,
        the last to +inf). Nominal climbing order: per rung going up, the lead foot steps in the
        first 30 % of the rung's time and the other foot follows; going down, the lead foot
        steps late, once the body has lowered. Each foot then leaves a rung before the hip rises
        out of its reach and lands only once the next rung is within reach."""
        key = id(lad)
        if key in self._ladder_plans:
            return self._ladder_plans[key]
        moves = []                                         # (side, b0, b1, from, to)
        seg = (lad.t_up[1] - lad.t_up[0] - 0.8) / lad.rungs
        for k in range(lad.rungs):
            a = lad.t_up[0] + 0.8 + k * seg
            lead = k % 2
            moves += [(lead, a, a + 0.3 * seg, k, k + 1), (1 - lead, a + 0.3 * seg, a + 0.75 * seg, k, k + 1)]
        if lad.t_down:
            top = lad.rungs
            seg = (lad.t_down[1] - lad.t_down[0] - 0.8) / lad.rungs
            for k in range(lad.rungs):
                a = lad.t_down[0] + 0.8 + k * seg
                lead = k % 2
                moves += [(lead, a + 0.3 * seg, a + 0.7 * seg, top - k, top - k - 1),
                          (1 - lead, a + 0.7 * seg, a + 1.05 * seg, top - k, top - k - 1)]
        plan = []
        for s in (0, 1):
            mine = sorted((m for m in moves if m[0] == s), key=lambda m: m[1])
            stances = []
            t_land_prev, level = -np.inf, 0
            for _, b0, b1, _, to in mine:
                here = self.rung_contact(lad, level, s)
                pivot = "ball" if level > 0 else "auto"
                k_from = self.ix(t_land_prev) if np.isfinite(t_land_prev) else self.ix(b0 - 2.0)
                t_lift = b0
                for kk in range(k_from, self.ix(b0) + 1 if to > level else k_from):   # climbing up only
                    if self._reach_excess(here, lad.yaw, self.hips[kk, s], pivot) > 0.005:
                        t_lift = float(self.t[kk]) - 0.04
                        break
                t_lift = max(min(t_lift, b0), t_land_prev + 0.1)
                there = self.rung_contact(lad, to, s)
                t_land = max(b1, t_lift + 0.2)
                for kk in range(self.ix(t_land), self.ix(t_land + 0.6) + 1):
                    if self._reach_excess(there, lad.yaw, self.hips[kk, s], "ball" if to > 0 else "auto") <= 0.005:
                        t_land = max(float(self.t[kk]), t_lift + 0.2)
                        break
                stances.append((t_land_prev, t_lift, level))
                t_land_prev, level = t_land, to
            stances.append((t_land_prev, np.inf, level))
            plan.append(stances)
        self._ladder_plans[key] = plan
        return plan

    def ladder_feet(self, lad, sl: slice):
        """Feet on the rungs (ladder_plan), swinging rung to rung away from the ladder."""
        t = self.t[sl]
        n = len(t)
        f = fwd(lad.yaw)
        plan = self.ladder_plan(lad)
        ank = np.zeros((n, 2, 3))
        u = np.zeros((n, 2, 3))
        nrm = np.zeros((n, 2, 3))
        for s in (0, 1):
            stances = plan[s]
            contacts = [self.rung_contact(lad, level, s) for _, _, level in stances]
            pivots = ["ball" if level > 0 else "auto" for _, _, level in stances]
            ends: dict = {}                                # swing j -> poses at lift-off and touchdown
            j = 0
            for i in range(n):
                k = sl.start + i
                ti = float(t[i])
                while j + 1 < len(stances) and ti >= stances[j + 1][0]:
                    j += 1
                t_land, t_lift, level = stances[j]
                if ti < t_lift or j + 1 >= len(stances):
                    a, pitch = self._planted(contacts[j], lad.yaw, self.hips[k, s], pivots[j])
                else:
                    nxt_land = stances[j + 1][0]
                    if j not in ends:
                        ends[j] = (self._planted(contacts[j], lad.yaw, self.hips[self.ix(t_lift), s], pivots[j]),
                                   self._planted(contacts[j + 1], lad.yaw, self.hips[self.ix(nxt_land), s], pivots[j + 1]))
                    (a0, p0), (a1, p1) = ends[j]
                    w_u = float(np.clip((ti - t_lift) / max(nxt_land - t_lift, 1e-6), 0, 1))
                    w = swing_profile(w_u)
                    bump = float(np.sin(np.pi * w_u))
                    a = a0 + (a1 - a0) * w + UP * (0.05 * bump) - f * (0.12 * bump)
                    pitch = p0 + (p1 - p0) * w
                ank[i, s] = a
                u[i, s], nrm[i, s] = self._foot_axes(lad.yaw, pitch)
        poles = self.hips[sl] + f * 1.0 + UP * 0.2
        return ank, u, nrm, poles

    # ===================================================================================
    # Arms
    # ===================================================================================
    def arm_timeline(self) -> list:
        rec, ctx = self.ctx.rec, self.ctx
        tl = Timeline("idle")
        sc = self.scene
        for c in rec.calls:
            n = c.name
            anc = rec.ancestors(c)
            if self.mocap is not None and n in ("walk", "hammer", "kneel_work") and self.mocap.add_arm_items(self, tl, c):
                continue                                   # a clip drives these arms
            if n == "walk" and "carry" not in anc:
                tl.add(c.t0, c.t0 + 0.5, "idle")
            elif n == "run":
                tl.add(c.t0, c.t0 + 0.6, "run")
                tl.add(c.t1 - 0.3, c.t1 + 0.4, "idle")
            elif n == "hammer":
                on_ladder = any(l.t_up[1] <= c.t0 + 1e-6 and (l.t_down is None or c.t0 < l.t_down[0]) for l in sc.ladders
                                if l.kind in ("ladder", "stepladder"))
                strikes = [s for x in rec.find("impulses", inside=c) for s in x.args.get("times_abs", [])]
                tl.add(c.t0, c.t0 + 0.5, "hammer", key=("hammer", c.index), strikes=np.array(sorted(strikes)),
                       overhead=bool(c.args.get("overhead")) or on_ladder)
                tl.add(c.t1, c.t1 + 0.5, "ladder" if on_ladder else "idle",
                       **({"ladder": self._ladder_at(c.t1), "key": "ladder"} if on_ladder else {}))
            elif n == "drill_overhead":
                tl.add(c.t0, c.t0 + 1.0, "overhead")
                tl.add(c.t1 - 1.0, c.t1, "idle")
            elif n == "jackhammer":
                tl.add(c.t0, c.t0 + 0.8, "jack")
                tl.add(c.t1 - 1.0, c.t1, "idle")
            elif n == "lift":
                # Reach to the grip while going down, hold it still through the set, then the load
                # hangs from the shoulders as the body rises (no arm curl), then the carry pose.
                ev = self._lift_events(c, ("lift_grasp", "lift_off", "lift_end"))
                if ev:
                    (t_grasp, info), (t_off, _), (t_end, _) = ev
                    hands = self._grip_hands(info, t_grasp)
                    tl.add(c.t0 + 0.15 * (t_grasp - c.t0), t_grasp, "grip", key=("grip", c.index), hands=hands)
                    tl.add(t_off, t_off + 0.02, "hold", key=("hold", c.index), hands=hands, k_ref=self.ix(t_off))
                    # Carry pose once upright. hang: how far the load reaches below the hands on the
                    # walk (the sack is held as it will be at the put-down, scene.bag_motion).
                    touch = [i for t, n, i in self.ctx.events if n == "put_down_touch" and t > t_end]
                    hang = float(touch[0]["object_height"]) if touch else 0.0
                    tl.add(t_end, t_end + 0.7, "carry", key=("carry", c.index), hang=hang)
            elif n == "put_down":
                # The load hangs from the shoulders on the way down and lands on the grip point at
                # touchdown; the hands stay on it until the release, then hang.
                ev = self._lift_events(c, ("put_down_touch", "put_down_release", "put_down_end"))
                if ev:
                    (t_touch, info), (t_release, _), _ = ev
                    hands = self._grip_hands(info, t_touch)
                    tl.add(c.t0, c.t0 + 0.6 * (t_touch - c.t0), "hold", key=("hold", c.index), hands=hands,
                           k_ref=self.ix(t_touch))
                    tl.add(t_touch, t_touch + 0.02, "grip", key=("grip", c.index), hands=hands)
                    tl.add(t_release, t_release + 0.8, "idle")
            elif n == "rebar_tie":
                tl.add(c.t0, c.t0 + 1.5, "tie", strikes=np.array(sorted(s for x in rec.find("impulses", inside=c)
                                                                        for s in x.args.get("times_abs", []))))
                tl.add(c.t1 - 1.5, c.t1, "idle")
            elif n == "kneel_work":
                tl.add(c.t0 + 0.8, c.t0 + 2.0, "floor")
                tl.add(c.t1 - 1.5, c.t1, "idle")
            elif n == "ladder":
                lad = self._ladder_for_call(c)
                if lad is None:
                    continue
                tl.add(c.t0, c.t0 + 0.8, "ladder", key="ladder", ladder=lad)
                going_up = abs(lad.t_up[0] - c.t0) < 1e-6
                if (going_up and lad.kind == "steps") or not going_up:
                    tl.add(c.t1 - 0.2, c.t1 + 0.5, "idle")
            elif n == "sit":
                if "cab_ride" in anc:
                    tl.add(c.t0, c.t1, "sit")
                    tl.add(c.t1, c.t1 + 1.0, "cab")
                else:
                    tl.add(c.t0, c.t1, "sit")
            elif n == "stand_up":
                tl.add(c.t0, c.t1, "idle")
            elif n in ("quadruped", "crawl"):
                tl.add(c.t0, c.t0 + (c.t1 - c.t0 if n == "quadruped" else 0.4), "quad", key="quad", crawl=(n == "crawl"))
            elif n == "sit_ground":
                tl.add(c.t0, c.t1, "sitg")
            elif n == "lie":
                nested = [x for x in rec.calls if x.name in ("kneel", "sit_ground", "quadruped") and rec.is_inside(x, c)]
                start = max([x.t1 for x in nested], default=c.t0)
                tl.add(start, c.t1, "lie")
            elif n == "lying_activity":
                tl.add(c.t0, c.t0 + 0.8, "liework")
                tl.add(c.t1 - 0.6, c.t1, "lie")
            elif n == "remove_strip":
                tl.add(c.t0, c.t0 + 1.2, "remove")
                tl.add(c.t0 + 2.0, c.t0 + 2.4, "carrystrip")
                tl.add(c.t1, c.t1 + 1.0, "idle")
            elif n == "hat_off":
                tl.add(c.t0 + 0.3, c.t0 + 0.7, "hatgrab")
                tl.add(c.t0 + 1.4, c.t0 + 2.2, "idle")
        if sc.dyn.get("stepladder"):
            # Between climbs the stepladder is carried in the right hand (scene.stepladder_dynamics).
            ups = sorted(sc.ladders, key=lambda l: l.t_up[0])
            for k, lad in enumerate(ups):
                if lad.t_down is None:
                    continue
                t_set = ups[k + 1].t_up[0] if k + 1 < len(ups) else self.t[-1] + 10
                tl.add(lad.t_down[1] + 0.2, lad.t_down[1] + 0.9, "carryladder", key=("cl", k))
                if t_set < self.t[-1] + 5:
                    tl.add(t_set - 0.9, t_set - 0.1, "idle", key=("set", k))
        behaviour = ctx.params.get("behaviour")
        for t, name, info in ctx.events:
            if name == "fall_onset":
                tl.add(t, t + 0.3, "brace", key=("brace", t), t_on=t)
            elif name == "free_fall_start" and info.get("kind") != "jump":
                tl.add(t, t + 0.3, "air")
            elif name == "impact" and info.get("mode") != "feet_first":
                tl.add(t - 0.05, t + 0.7, "lie")
            elif name == "free_fall_end" and info.get("reason") == "feet":
                tl.add(t, t + 0.3, "balance")
            elif name == "arrest_start":
                tl.add(t, t + 0.6, "hang", struggle=behaviour == "struggle")
            elif name == "near_fall_onset":
                if self.scene.rail is not None:
                    tl.add(t, t + 0.25, "rail")
                    tl.add(t + 1.4, t + 2.0, "idle")
                else:
                    tl.add(t, t + 0.2, "balance")
                    tl.add(t + 1.0, t + 1.8, "idle")
            elif name == "struck":
                tl.add(t + 0.05, t + 0.3, "balance")
        return tl.finalize()

    def _lift_events(self, call, names) -> list:
        """(t, info) of each named ground-truth event inside a lift / put-down call, or [] if any
        is missing."""
        found = []
        for name in names:
            hit = [(t, info) for t, n, info in self.ctx.events if n == name and call.t0 - 1e-6 <= t <= call.t1 + 1e-6]
            if not hit:
                return []
            found.append(hit[0])
        return found

    def _grip_hands(self, info: dict, t: float) -> np.ndarray:
        """Left and right hand on the sides of the load around the simulated grip point (2, 3)."""
        grip = np.array([info["gx"], info["gy"], info["gz"]], float)
        across = right(self.heading[self.ix(t)])
        return np.stack([grip + across * (s * LOAD_HALF_WIDTH_M) for s in SIDES])

    def _ladder_for_call(self, c):
        return next((l for l in self.scene.ladders if abs(l.t_up[0] - c.t0) < 1e-6 or (l.t_down and abs(l.t_down[0] - c.t0) < 1e-6)), None)

    def _ladder_at(self, t):
        return next((l for l in self.scene.ladders if l.t_up[0] <= t and (l.t_down is None or t <= l.t_down[1])), None)

    def build_arms(self):
        d = self.dims
        N = self.N
        items = self.arm_timeline()
        hand = np.zeros((N, 2, 3))
        pole = np.zeros((N, 2, 3))
        tool = np.zeros(N, int)
        for k, item in enumerate(items):
            k0 = self.ix(item.t0) if k > 0 else 0
            end_t = items[k + 1].t1 if k + 1 < len(items) else self.t[-1] + 1
            k1 = min(self.ix(end_t) + 1, N) if end_t <= self.t[-1] else N
            if k0 >= N or k1 <= k0:
                continue
            sl = slice(k0, k1)
            h, p = self.arm_target(item, sl)
            w = smoothstep((self.t[sl] - item.t0) / max(item.t1 - item.t0, 1e-3)) if k > 0 else np.ones(k1 - k0)
            hand[sl] = lerp(hand[sl], h, w) if k > 0 else h
            pole[sl] = lerp(pole[sl], p, w) if k > 0 else p
            code = {"hammer": 1, "overhead": 2, "jack": 3}.get(item.info.get("as_mode", item.mode), 0)
            stop = self.ix(items[k + 1].t0) if k + 1 < len(items) else N
            tool[k0:max(stop, k0)] = code
        hand = self._clear_of_shoulder(hand)
        self.hand_target = hand
        self.arm_pole = pole
        elbow, hand_ik = two_bone(self.shoulders, hand, pole, d.upper_arm, d.forearm, continuous=True)
        self.arm_items = items
        self.elbow, self.hand, self.tool = elbow, hand_ik, tool

    def _clear_of_shoulder(self, hand: np.ndarray, radius: float = 0.28) -> np.ndarray:
        """Hand targets passing within `radius` (m) of the shoulder (e.g. the blend from a hanging
        arm to an overhead pose) arc out forward and sideways instead, as a real arm raise does;
        a target through the shoulder would whip the elbow around it."""
        side = np.array(SIDES)[None, :, None]
        out = unit(self.f[:, None, :] * 0.8 + self.r[:, None, :] * side * 0.6)
        dist = np.linalg.norm(hand - self.shoulders, axis=-1, keepdims=True)
        w = np.clip(1.0 - dist / radius, 0.0, 1.0)
        return hand + out * (radius * w * w)

    def arm_target(self, item: Item, sl: slice):
        d = self.dims
        mode = item.mode
        sh = self.shoulders[sl]
        t = self.t[sl]
        f = self.f[sl][:, None, :]
        r = self.r[sl][:, None, :]
        side = np.array(SIDES)[None, :, None]
        X3, Y3, Z3 = (self.M[sl, 3, :, j][:, None, :] for j in range(3))
        A3 = -Z3
        hips = self.hips[sl]
        hip_mid = hips.mean(axis=1, keepdims=True)
        ground = self.ground_under_hips[sl].mean(axis=1)[:, None, None]
        default_pole = sh + DOWN * 0.25 - f * 0.35 + r * side * 0.2

        if mode == "clip":
            return self.mocap.arm_target(self, item, sl)
        if mode in ("idle", "run"):
            ank = self.ankle[sl]
            swing = np.sum((ank[:, ::-1] - ank) * f, axis=-1, keepdims=True) * 0.5
            if mode == "run":
                hand = sh + DOWN * 0.3 * d.s + f * (0.14 + 0.5 * swing) + r * side * 0.02
                return hand, sh - f * 0.4 + DOWN * 0.2 + r * side * 0.1
            hand = sh + DOWN * (d.arm * 0.93) + r * side * 0.07 + f * (0.05 + 0.42 * swing)
            hand[..., 1] = np.maximum(hand[..., 1], ground[..., 0] + 0.08)
            return hand, default_pole
        if mode == "hammer":
            return self._hammer(item, sl, sh, f, r, side, default_pole)
        if mode == "overhead":
            hand = sh + UP * 0.48 * d.s + f * 0.2 - r * side * 0.09
            jitter = np.sin(t * 2 * np.pi * 3.0)[:, None, None] * 0.004
            return hand + jitter, sh + r * side * 0.4 + DOWN * 0.1 - f * 0.2
        if mode == "jack":
            base = flat(hip_mid) + f * 0.45
            grip = base + UP * (0.82 * d.s) + r * side * 0.2
            grip[..., 1] = ground[..., 0] + 0.82 * d.s
            return grip, sh + r * side * 0.4 - f * 0.1
        if mode == "carryladder":
            ank = self.ankle[sl]
            swing = np.sum((ank[:, ::-1] - ank) * f, axis=-1, keepdims=True) * 0.5
            hand = sh + DOWN * (d.arm * 0.93) + r * side * 0.07 + f * (0.05 + 0.42 * swing)
            hand[:, 1] = hips[:, 1] + self.r[sl] * 0.3 + UP * 0.12
            return hand, default_pole
        if mode == "carry":
            hand = hip_mid + f * 0.3 + UP * 0.08 + r * side * 0.17
            # A load held in front clears the swinging knees: one steady forward shift for the
            # whole carry (a per-frame push would pump the arms with each stride), no further than
            # the arms reach with the elbows still bent (a straight-arm clamp would raise the hands).
            front_knee = np.max(np.sum(self.knee[sl] * f, axis=-1), axis=1)[:, None, None]
            need = LOAD_CLEAR_OF_KNEES_M - (np.sum(hand * f, axis=-1, keepdims=True) - front_knee)
            rel = hand - sh
            b = np.sum(rel * f, axis=-1)
            room = -b + np.sqrt(np.maximum(b * b - np.sum(rel * rel, axis=-1) + (0.95 * (d.upper_arm + d.forearm)) ** 2, 0.0))
            push = min(float(need.max()), CARRY_KNEE_PUSH_MAX_M, float(room.min()))
            hand = hand + f * max(push, 0.0)
            # If that is not enough, hug the load higher so its bottom passes over the knees.
            if item.info.get("hang", 0.0) > 0.0 and need.max() > push:
                knee_top = self.knee[sl][..., 1].max() + LOAD_CLEAR_OF_KNEES_M / 2
                raise_m = knee_top - float((hand[..., 1] - item.info["hang"]).min())
                hand = hand + UP * float(np.clip(raise_m, 0.0, CARRY_RAISE_MAX_M))
            return hand, sh + r * side * 0.35 - f * 0.2 + DOWN * 0.2
        if mode == "grip":
            # Hands fixed on the load (world) at the simulated grip point.
            hand = np.broadcast_to(item.info["hands"][None], sh.shape).copy()
            return hand, sh + r * side * 0.4 - f * 0.1
        if mode == "hold":
            # The load hangs from the shoulders: hands keep their offset from the shoulders at k_ref,
            # but stay far enough ahead of the hips and knees for the load to clear the legs and belly.
            k = item.info["k_ref"]
            hand = sh + (item.info["hands"] - self.shoulders[k])[None]
            ahead = np.sum((hand - hip_mid) * f, axis=-1, keepdims=True)
            hand = hand + f * np.maximum(LOAD_CLEAR_OF_HIPS_M - ahead, 0.0)
            front_knee = np.max(np.sum(self.knee[sl] * f, axis=-1), axis=1)[:, None, None]  # the leading knee
            ahead = np.sum(hand * f, axis=-1, keepdims=True) - front_knee
            hand = hand + f * np.maximum(LOAD_CLEAR_OF_KNEES_M - ahead, 0.0)
            return hand, sh + r * side * 0.4 - f * 0.1
        if mode == "tie":
            wiggle = np.sin(t * 2 * np.pi * 1.5)[:, None, None] * 0.03
            hand = flat(hip_mid) + f * (0.55 + wiggle) + r * side * (0.09 + wiggle)
            hand[..., 1] = ground[..., 0] + 0.2
            return hand, sh + r * side * 0.4 - f * 0.1
        if mode == "floor":
            reach_f = 0.5 + 0.12 * np.sin(t * 0.9)[:, None, None] * side
            hand = flat(hip_mid) + f * reach_f + r * side * 0.2
            hand[..., 1] = ground[..., 0] + 0.05
            return hand, sh + r * side * 0.4 - f * 0.1 + UP * 0.1
        if mode == "ladder":
            lad = item.info.get("ladder") or self._ladder_at(t[0])
            if lad is None:
                return sh + DOWN * d.arm * 0.9, default_pole
            lf, lr = fwd(lad.yaw), right(lad.yaw)
            levels, _ = self.ladder_levels(lad, t)
            y = sh[..., 1] + 0.15 + 0.1 * (levels - levels.mean(axis=1, keepdims=True))
            along = (y - lad.base[1]) / 4.0
            hand = lad.base[None, None, :] + lf[None, None, :] * along[..., None] + lr[None, None, :] * side * 0.22
            hand[..., 1] = y
            hand = hand - lf * 0.03
            return hand, sh + r * side * 0.4 + DOWN * 0.3 - f * 0.2
        if mode == "quad":
            hand = flat(sh) + f * 0.04 + r * side * 0.02
            if item.info.get("crawl"):
                dist = np.cumsum(np.linalg.norm(self.vel[sl], axis=1)) * self.dt
                phase = np.pi * dist / 0.22
                hand = hand + f * (0.08 * np.sin(phase[:, None, None] + (side < 0) * np.pi))
            hand[..., 1] = ground[..., 0] + 0.04
            return hand, sh + r * side * 0.5 - f * 0.2
        if mode == "sit":
            hand = hip_mid + f * (d.thigh * 0.55) + UP * 0.12 + r * side * 0.12
            return hand, default_pole
        if mode == "cab":
            hand = hip_mid + f * 0.4 + UP * 0.3 + r * side * 0.24
            return hand, sh + r * side * 0.3 + DOWN * 0.3
        if mode == "sitg":
            hand = flat(hip_mid) - f * 0.2 + r * side * 0.3
            hand[..., 1] = ground[..., 0] + 0.04
            return hand, sh + r * side * 0.4 - f * 0.4
        if mode == "lie":
            # Arms along the body, raised sideways just enough to stay on the floor.
            length = np.linalg.norm(-Y3 * (d.arm * 0.86) + X3 * side * 0.12, axis=-1)
            rise = ground[..., 0] + 0.05 - sh[..., 1]
            hand = sh + self._flex_above(-Y3 * (d.arm * 0.86) + X3 * side * 0.12, X3 * side, length, rise,
                                         np.radians(120.0)) * length[..., None]
            return hand, sh + X3 * side * 0.4 - Y3 * 0.2
        if mode == "liework":
            work = np.sin(t * 1.3)[:, None, None] * 0.06
            hand = sh + A3 * (0.42 * d.s) + Y3 * (0.12 + work) - X3 * side * 0.05
            return hand, sh + X3 * side * 0.4
        if mode == "brace":
            t_on = item.info["t_on"]
            fall = self._fall_dir(t_on)
            hand = sh + unit(fall[None, None, :] * 0.75 + DOWN * 0.55 + X3 * side * 0.25) * (d.arm * 0.95)
            return hand, sh + X3 * side * 0.4 + UP * 0.1
        if mode == "air":
            flail = np.sin(t * 7.0 + 1.0)[:, None, None] * 0.15
            hand = sh + unit(X3 * side * 0.8 + UP * (0.5 + flail) + A3 * 0.2) * (d.arm * 0.92)
            return hand, sh + DOWN * 0.3 - A3 * 0.2
        if mode == "balance":
            hand = sh + unit(X3 * side * 0.85 + UP * 0.3 + A3 * 0.2) * (d.arm * 0.9)
            return hand, sh + DOWN * 0.3 - A3 * 0.2
        if mode == "rail":
            rail = self.scene.rail
            a, b = rail["a"], rail["b"]
            hand = sh + unit(X3 * side * 0.85 + UP * 0.3 + A3 * 0.2) * (d.arm * 0.9)
            s_idx = 1 if rail["side"] > 0 else 0
            ab = b - a
            u = np.clip(np.sum((sh[:, s_idx] - a) * ab, axis=-1) / float(ab @ ab), 0, 1)
            grab = a[None] + u[:, None] * ab[None]
            grab[:, 1] = rail["y"] + 0.03
            hand[:, s_idx] = grab
            return hand, sh + X3 * side * 0.4 + DOWN * 0.3
        if mode == "hang":
            if item.info.get("struggle"):
                wave = np.sin(t * 2.2)[:, None, None] * 0.08
                hand = sh + UP * (0.5 + wave) * d.s - X3 * side * 0.08 + A3 * 0.12
            else:
                hand = sh + DOWN * (d.arm * 0.9) + A3 * 0.06
            return hand, sh + X3 * side * 0.4 - A3 * 0.2
        if mode == "remove":
            target = (self.anchors_strip[sl, 2] + self.anchors_strip[sl, 3]) / 2
            hand = target[:, None, :] + r * side * 0.12 + UP * 0.05
            return hand, sh + X3 * side * 0.5 + UP * 0.2
        if mode == "carrystrip":
            target = self.anchors_strip[sl].mean(axis=1)
            hand = target[:, None, :] + r * side * 0.15 + UP * 0.03
            return hand, default_pole
        if mode == "hatgrab":
            hand = sh + DOWN * (d.arm * 0.93) + r * side * 0.07
            hand[:, 1] = self.head_pos[sl] + UP * 0.1 - self.f[sl] * 0.04
            return hand, default_pole
        raise ValueError(mode)

    def _hammer_points(self, sl, overhead: bool):
        """Right-hand impact point and raised point (n,3) of a hammer stroke: on the scene's hammer
        post when there is one, else ahead of the right shoulder; overhead above it."""
        d = self.dims
        shR = self.shoulders[sl, 1]
        fr, rr = self.f[sl], self.r[sl]
        if overhead:
            impact = shR + UP * 0.55 * d.s + fr * 0.2 - rr * 0.12
            raised = shR + UP * 0.32 * d.s - fr * 0.02 + rr * 0.06
            return impact, raised
        post = self.scene.dyn.get("hammer_post")
        impact = shR + fr * 0.45 + DOWN * 0.14 - rr * 0.04
        if post is not None:
            impact = np.column_stack([np.full(len(shR), post[0]), shR[:, 1] - 0.14, np.full(len(shR), post[2])]) - rr * 0.07
        raised = shR + UP * 0.14 + fr * 0.2 + rr * 0.07
        return impact, raised

    def _hammer(self, item, sl, sh, f, r, side, default_pole):
        t = self.t[sl]
        strikes = item.info.get("strikes", np.array([]))
        overhead = item.info.get("overhead")
        fr, rr = f[:, 0], r[:, 0]
        impact, raised = self._hammer_points(sl, bool(overhead))
        u = np.zeros(len(t))
        if len(strikes) >= 2:
            idx = np.clip(np.searchsorted(strikes, t), 1, len(strikes) - 1)
            prev, nxt = strikes[idx - 1], strikes[idx]
            u = np.clip((t - prev) / np.maximum(nxt - prev, 1e-3), 0, 1)
            u = np.where((t < strikes[0]) | (t > strikes[-1]), 0.35, u)
        wind = np.sin(np.pi * np.clip(u, 0, 1) ** 1.6)
        right_hand = impact + (raised - impact) * wind[:, None]
        left_hand = impact - rr * 0.12 + DOWN * 0.03 - fr * 0.02
        hand = np.stack([left_hand, right_hand], axis=1)
        return hand, default_pole

    def _fall_dir(self, t_on: float) -> np.ndarray:
        t_end = self.ctx.first_after(("impact", "free_fall_end", "arrest_start", "slide_start"), t_on)
        return self.ctx.fall_direction(t_on, t_end)

    # ===================================================================================
    # Head
    # ===================================================================================
    def conscious_level(self) -> np.ndarray:
        """Consciousness per frame, 1 awake .. 0 unconscious, eased over CONSCIOUS_BLEND_S at
        each change so the head tone does not snap."""
        level = np.ones(self.N)
        for t_change, value in sorted(self.ctx.rec.conscious_log, key=lambda e: e[0]):
            after = self.t >= t_change
            if not after.any():
                continue
            k = int(np.argmax(after))
            start = level[k - 1] if k > 0 else 1.0
            target = 1.0 if value else 0.0
            ramp = smoothstep((self.t[after] - t_change) / CONSCIOUS_BLEND_S)
            level[after] = start + (target - start) * ramp
        return level

    def build_head(self):
        d = self.dims
        R3 = Rotation.from_matrix(self.M[:, 3])
        level = Rotation.from_euler("YX", np.column_stack([self.heading, np.full(self.N, -0.12)]))
        tilt = np.degrees(np.arccos(np.clip(self.M[:, 3, 1, 1], -1, 1)))
        conscious = self.conscious_level()
        k = 0.65 * (1 - smoothstep((tilt - 35.0) / 45.0)) * (0.25 + 0.75 * conscious)
        rel = (R3.inv() * level).as_rotvec()
        head_rot = R3 * Rotation.from_rotvec(rel * k[:, None])
        neck = self.P[:, 3] + self.M[:, 3] @ np.array([0.0, d.neck, 0.0])
        head_rest = np.array([0.0, 0.1 * d.H, -0.022 * d.s])
        head_rot = self._head_above_support(neck, head_rot, head_rest)
        head = neck + head_rot.apply(head_rest)
        self.neck, self.head_pos, self.head_q = neck, head, head_rot.as_quat()

    def _head_above_support(self, neck: np.ndarray, head_rot: Rotation, head_rest: np.ndarray,
                            max_lift_deg: float = 80.0, rate_deg: float = 4.0) -> Rotation:
        """Lift the head about the neck (neck extension, in the vertical plane of the head's
        direction) wherever its centre would sit less than HEAD_RADIUS * s above the support
        under it, e.g. face down after a fall. Only surfaces at or below the neck count (a neck
        inside a drawn prop is a scene mismatch, not a pillow). The lift angle per frame is the
        smallest envelope above the requirement that changes by at most rate_deg per frame,
        capped at max_lift_deg, so the head never snaps. A neck itself below the support (the
        simulated trunk below the floor, ground truth) cannot be fixed by the head: the lift
        fades out between 0 and 10 cm of such depth."""
        offset = head_rot.apply(head_rest)
        length = float(np.linalg.norm(head_rest))
        head = neck + offset
        support = np.array([self.scene.support(h[0], h[2], b) for h, b in zip(head, neck[:, 1] + 0.02, strict=True)])
        need = support + HEAD_RADIUS * self.dims.s
        current = offset / length
        elevation = np.arcsin(np.clip(current[:, 1], -1.0, 1.0))
        wanted = np.arcsin(np.clip((need - neck[:, 1]) / length, -1.0, 1.0))
        lift = np.clip(wanted - elevation, 0.0, np.radians(max_lift_deg))
        lift *= smoothstep((neck[:, 1] - support + 0.1) / 0.1)
        if not (lift > 0).any():
            return head_rot
        rate = np.radians(rate_deg)
        for i in range(1, len(lift)):                     # decay no faster than `rate` ...
            lift[i] = max(lift[i], lift[i - 1] - rate)
        for i in range(len(lift) - 2, -1, -1):            # ... and start rising early enough
            lift[i] = max(lift[i], lift[i + 1] - rate)
        lift = np.minimum(lift, np.radians(max_lift_deg))
        lift = np.minimum(lift, np.maximum(np.pi / 2 - elevation, 0.0))   # never past pointing straight up
        active = lift > 1e-6
        anterior = unit(flat(-self.M[active, 3, :, 2]) + 1e-9 * self.f[active])   # thorax anterior
        horiz = unit(flat(current[active]) + 0.05 * anterior)   # continuous even for a vertical head
        axis = np.cross(horiz, UP)                        # turning about it raises the head direction
        turn = Rotation.from_rotvec(axis * lift[active][:, None])
        quats = head_rot.as_quat()
        quats[active] = (turn * Rotation.from_quat(quats[active])).as_quat()
        return Rotation.from_quat(quats)


def build_rig(ctx, scene, frames: np.ndarray, body_idx: np.ndarray, strip_anchors: np.ndarray,
              shift: np.ndarray | None = None, mocap=None) -> Rig:
    """mocap: a mocap.Session (clip-driven limbs where a clip applies) or None (procedural)."""
    rb = RigBuilder(ctx, scene, frames, body_idx, shift, mocap)
    rb.anchors_strip = strip_anchors
    rb.build_head()
    rb.build_legs()
    rb.build_arms()
    joints = np.stack([rb.hips[:, 0], rb.hips[:, 1], rb.knee[:, 0], rb.knee[:, 1], rb.ankle[:, 0], rb.ankle[:, 1],
                       rb.heel[:, 0], rb.heel[:, 1], rb.toe[:, 0], rb.toe[:, 1], rb.shoulders[:, 0], rb.shoulders[:, 1],
                       rb.elbow[:, 0], rb.elbow[:, 1], rb.hand[:, 0], rb.hand[:, 1], rb.neck, rb.head_pos], axis=1)
    dring = rb.P[:, 3] + np.einsum("nij,j->ni", rb.M[:, 3], np.array([0.0, -0.02 * rb.dims.H, 0.09]))
    rig = Rig(joints, rb.head_q, rb.P[:, 0], rb.P, dring, strip_anchors[:, 3], rb.head_pos, rb.hand[:, 0], rb.hand[:, 1],
              rb.shoulders[:, 1], rb.tool, [(it.t0, it.t1, it.mode) for it in rb.leg_items],
              [(it.t0, it.t1, it.mode) for it in rb.arm_items])
    rig.dims = rb.dims
    # IK targets before two_bone (for the reach gate) and the mocap log (payload meta.mocap).
    rig.targets = {"ankle": rb.ankle_target, "hand": rb.hand_target, "planted": rb.planted}
    rig.mocap = mocap.meta() if mocap is not None else None
    rig.mocap_session = mocap
    rig.builder = rb                   # for the contact pass (contact.py); not exported
    return rig
