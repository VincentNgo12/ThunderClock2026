# File: physics.py
# Purpose: Small deterministic position-based dynamics engine (XPBD) for the 3D viewer, the
#          ragdoll pass that lets the worker's limbs and head fall, collide and settle around
#          the simulated spine, and the render-only "rest lift" for a trunk below the floor.
# Contributors: Unassigned
"""Viewer physics. Render only: the simulated spine, IMU signals and events never change.

Engine (World)
--------------
Particles (mass, position, previous position) under gravity (model_api.GRAVITY along -Y)
with a small linear air damping. Every substep runs, in order:

    predict (inertia + gravity; driven particles' velocity damped toward their target's)
    -> soft pose drive -> kinematic pins -> distance / range constraints (graph-coloured
    batches) -> bend-side and cone limits -> centroid target -> collisions with friction

Velocities are implicit, v = (x - x_prev) / dt. A contact removes the approach velocity and
leaves no velocity away from the surface (restitution 0, even after pushing a deep
penetration out; Colliders.max_push can also cap how fast boxes and capsules push out). The
bend-side and cone limits are inelastic in the same way: their corrections are bounded per
substep, add no velocity and remove the velocity that pushes further past the limit. The
pose drive is a critically damped spring by default (no overshoot), optionally with a cap
on how fast it closes in on its target. This is XPBD with "small steps" (Macklin et al.
2019): many substeps and one constraint pass each (World(iterations=n) for stiffer chains:
a 10-link rope stretches ~2.5 mm with 1, ~0.6 mm with 4), stable without per-scene tuning. Deterministic: fixed
substep, fixed order, no random numbers.

Substep rate 250 Hz: 5 substeps per 50 Hz output frame (an integer, so pins and drive targets
interpolate exactly between frames); at a 10 m/s impact a particle moves 4 cm per substep,
less than the 4.5-9 cm limb radii, so limbs do not tunnel through the floor; and it costs a
quarter of the simulator's 1 kHz. A broadphase once per step keeps only the shapes that the
particles can reach within that step; moving capsules and the net dent are interpolated
across the substeps like pins.

Generic use (a rope pinned to a moving anchor, a cloth on a frame...):

    col = Colliders(ground_y=0.0)
    col.add_box(centre, size, yaw=0.3)
    world = World(col)
    ids = world.add_particles(points, mass=0.1, radius=0.01)
    world.add_distance(ids[:-1], ids[1:], compliance=1e-6)    # chain links
    pin = world.add_pin(ids[:1])
    for frame in frames:
        world.set_pin_targets(pin, anchor[frame])                # moved linearly over the step
        world.step(1 / 50)
        positions = world.x

Other tools: set_drive (soft pull toward moving targets), set_centroid (keep the weighted
centre of a set of particles on a moving target), add_bend / add_cone (joint limits),
set_distance (change rest lengths), world.touched (particles that touched a shape last step).

Limb ragdoll (apply_limb_physics)
---------------------------------
Hips, shoulders and neck stay exactly where the rig puts them (rigid on the simulated spine).
Knees, ankles, heels, toes, elbows, hands and the head are particles joined by exact bones,
pulled toward the procedural pose by a drive whose strength follows a muscle-tone schedule
ctrl(t) in [0, 1] built from ground-truth events, phases and consciousness. ctrl = 1 (normal
activity) tracks the pose; a fall ramps it down to a limp tone, so limbs swing, hit the floor
and settle; it ramps back up for struggling (after a short stunned spell), getting up and
recovery. When tone comes back from a limp state the drive target cross-fades in time from
the ragdoll's pose to the procedural one (RISE_S) and the limbs close in on it at most at
DRIVE_MAX_SPEED, so they move there at a human pace instead of whipping. Feet keep a passive
ankle stiffness (they never spin about the shank). During free flight the limbs also keep the
drawn whole-body centre of mass on the simulator's ballistic path (a soft hold that fades in
and out; they take the reaction of the prescribed trunk tumble); feet-first flights plan the
legs from the takeoff pose to the touchdown pose. A limb whose root is itself below the floor
(the simulated trunk in it) follows it down instead of fighting its bone.

Only time windows that need it are simulated (low tone, or lying in the safety net); outside
them the procedural pose is kept exactly. Inside them a final pass keeps bone lengths exact
and limb joints on or above nearby surfaces (only surfaces within 5 cm count, and the pass
fades out with the window blend), and the head turns at most HEAD_STEP_DEG per frame
relative to the upper thorax.

Rest lift (rest_lift)
---------------------
The simulated trunk sometimes rests a few cm below the support while lying (ground truth).
The viewer lifts the whole drawn body (and the worn strip) by a slow vertical offset, only
while lying at rest, so the torso rests on the floor. IMU signals are untouched.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.spatial.transform import Rotation

from ..model_api import GRAVITY
from ..spine import root_rotation
from .geom import UP, fwd, right, smoothstep

ENGINE = "thundersim-pbd"
VERSION = "1.1"
SUBSTEP_HZ = 250.0
DENSE_MAX = 64          # worlds up to this many particles solve each batch with one small matmul
# Friction scales with the push-out depth (the normal impulse). A kinematic pin can drag a
# particle deep into a surface, which would make it stick and then slip each step; counting at
# most this much depth per substep (~12x a resting particle's) keeps friction strong but finite.
FRICTION_DEPTH_CAP_M = 0.002
# Default pose drive damping ratio (1 = critical): the drive is a spring toward the target, so
# it also damps the particle's velocity toward the target's velocity, or a limb whips around
# its pose when the tone comes back (an undamped drive overshoots 50-90 %).
DRIVE_DAMPING = 1.0
CONE_STEP_RAD = 0.05     # a cone limit turns a limb back by at most this much per substep
BEND_STEP_M = 0.01       # a bend-side limit moves a knee or elbow by at most this much per substep


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.add.reduce(a * b, axis=-1)


def _norm(v: np.ndarray) -> np.ndarray:
    return np.sqrt(_dot(v, v))


def unit_rows(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(_norm(v), 1e-12)[..., None]


# =============================================================================================
# Engine: colliders
# =============================================================================================
class Colliders:
    """Shapes the particles collide with (world metres, +Y up).

    - ground: infinite floor plane at ground_y (None = no floor).
    - boxes: oriented boxes (yaw, or any 3x3 rotation). A particle deeper than `max_depth`
      inside a box is ignored: that is a scene/trajectory mismatch, not a contact to resolve.
    - capsules: segments with a radius; set_capsule() moves one per step (a kinematic shape,
      for example a body part that a cloth or a limb drapes over).
    - sheets: horizontal one-sided rectangles with an optional Gaussian dent that can move per
      step (a safety net under load).
    Friction is Coulomb-like with static coefficient mu and kinetic 0.8 mu. A particle collides
    with a shape only when (particle group & shape mask) != 0. `max_push` (m per substep,
    default unlimited) caps how far a box or capsule pushes a particle out in one substep; a
    deeper particle also loses its velocity into the shape and comes out over a few substeps
    (a limb dragged deep into a wall by its pinned root then slides out instead of popping).
    """

    def __init__(self, ground_y: float | None = 0.0, ground_mu: float = 0.7, ground_mask: int = -1):
        self.ground_y = None if ground_y is None else float(ground_y)
        self.ground_mu = float(ground_mu)
        self.ground_mask = int(ground_mask)
        self._boxes: list = []
        self._caps: list = []
        self.sheets: list[dict] = []
        self.max_push = np.inf
        self._box_arrays = None
        self._cap_arrays = None

    # --- boxes ------------------------------------------------------------------------------
    def add_box(self, centre, size, yaw: float = 0.0, mu: float = 0.7, mask: int = 1, rotation=None,
                max_depth: float = 0.25) -> int:
        """Box with full sizes `size` (along its local x, y, z) centred at `centre`. The local
        axes are the columns of `rotation`, or a yaw about +Y (three.js rotation.y)."""
        if rotation is None:
            rotation = Rotation.from_rotvec([0.0, float(yaw), 0.0]).as_matrix()
        self._boxes.append((np.asarray(centre, float), np.asarray(rotation, float), np.asarray(size, float) / 2,
                            float(mu), int(mask), float(max_depth)))
        self._box_arrays = None
        return len(self._boxes) - 1

    def box_arrays(self):
        """(centre, rotation, half size, mu, mask, max depth, AABB min, AABB max) per box."""
        if self._box_arrays is None and self._boxes:
            c, R, h, mu, mask, deep = (np.array(v) for v in zip(*self._boxes, strict=True))
            ext = np.einsum("bij,bj->bi", np.abs(R), h)            # world half extents
            self._box_arrays = (c, R, h, mu, mask, deep, c - ext, c + ext)
        return self._box_arrays

    # --- capsules ---------------------------------------------------------------------------
    def add_capsule(self, a, b, radius: float, mu: float = 0.5, mask: int = 1) -> int:
        self._caps.append([np.asarray(a, float).copy(), np.asarray(b, float).copy(), float(radius), float(mu),
                           int(mask)])
        self._cap_arrays = None
        return len(self._caps) - 1

    def set_capsule(self, cid: int, a, b) -> None:
        """Move a capsule: it reaches (a, b) at the end of the coming step, moving linearly
        from where it was (a kinematic shape, like a pin)."""
        self.cap_arrays()
        self._caps[cid][0] = np.asarray(a, float)
        self._caps[cid][1] = np.asarray(b, float)
        self._cap_arrays[0][cid] = self._caps[cid][0]
        self._cap_arrays[1][cid] = self._caps[cid][1] - self._caps[cid][0]

    def cap_arrays(self):
        """(start a, axis b - a, radius, mu, mask, a and axis at the start of the step) per capsule."""
        if self._cap_arrays is None and self._caps:
            a = np.array([c[0] for c in self._caps])
            ab = np.array([c[1] for c in self._caps]) - a
            self._cap_arrays = (a, ab, np.array([c[2] for c in self._caps]), np.array([c[3] for c in self._caps]),
                                np.array([c[4] for c in self._caps]), a.copy(), ab.copy())
        return self._cap_arrays

    def end_step(self) -> None:
        """The moving shapes' end-of-step poses become the next step's start poses."""
        if self._cap_arrays is not None:
            self._cap_arrays[5][:] = self._cap_arrays[0]
            self._cap_arrays[6][:] = self._cap_arrays[1]
        for sh in self.sheets:
            sh["dent_prev"] = sh["dent"]

    # --- sheets -----------------------------------------------------------------------------
    def add_sheet(self, centre, yaw: float, size_rf, mu: float = 0.8, mask: int = 1, thickness: float = 0.4) -> int:
        """Horizontal rectangle (size along right and fwd of yaw) with its surface at centre[1]."""
        self.sheets.append({"c": np.asarray(centre, float), "r": right(yaw), "f": fwd(yaw),
                            "half": (float(size_rf[0]) / 2, float(size_rf[1]) / 2), "mu": float(mu), "mask": int(mask),
                            "thick": float(thickness), "dent": (0.0, 0.0, 0.0), "dent_prev": (0.0, 0.0, 0.0),
                            "dent_w": 1.1, "sag": 0.0, "sag_w": 6.0})
        return len(self.sheets) - 1

    def set_dent(self, sid: int, x: float, z: float, depth: float, width: float = 1.1, sag: float = 0.0,
                 sag_width: float = 6.0) -> None:
        """Surface lowered by depth * exp(-d^2 / width) + sag * exp(-d^2 / sag_width), d = distance
        from (x, z) in metres, reached at the end of the coming step (moving linearly)."""
        sh = self.sheets[sid]
        sh["dent"] = (float(x), float(z), float(depth))
        sh["dent_w"], sh["sag"], sh["sag_w"] = float(width), float(sag), float(sag_width)

    def sheet_top(self, sid: int, x: np.ndarray, z: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Surface height at (x, z), a fraction alpha through the current step."""
        sh = self.sheets[sid]
        dx, dz, depth = (p0 + (p1 - p0) * alpha for p0, p1 in zip(sh["dent_prev"], sh["dent"], strict=True))
        d2 = (x - dx) ** 2 + (z - dz) ** 2
        return sh["c"][1] - depth * np.exp(-d2 / sh["dent_w"]) - sh["sag"] * np.exp(-d2 / sh["sag_w"])

    @property
    def n_boxes(self) -> int:
        return len(self._boxes)

    @property
    def n_caps(self) -> int:
        return len(self._caps)


# =============================================================================================
# Engine: particle world
# =============================================================================================
class World:
    """Particle system with XPBD constraints and collisions. See the module docstring.

    Units: metres, seconds, kilograms. A particle with infinite mass is kinematic: constraints
    and collisions never move it (pins move it). After each step, `touched` marks the
    particles that touched a shape whose mask shares a bit with `touch_mask`. `ground_drop`
    (per particle, m) lowers the floor plane for a particle (a limb whose pinned root is
    itself below the floor follows it down instead of fighting its bone).
    """

    def __init__(self, colliders: Colliders | None = None, substep_hz: float = SUBSTEP_HZ, gravity: float = GRAVITY,
                 air_damping: float = 0.01, iterations: int = 1):
        self.col = colliders if colliders is not None else Colliders(None)
        self.dt = 1.0 / float(substep_hz)
        self.gravity = float(gravity)
        self.air_damping = float(air_damping)   # 1/s, linear: v *= 1 - air_damping * dt per substep
        self.iterations = int(iterations)
        self.touch_mask = -1
        self.x = np.zeros((0, 3))
        self.xp = np.zeros((0, 3))
        self.w = np.zeros(0)                    # inverse masses
        self.radius = np.zeros(0)
        self.group = np.zeros(0, int)
        self.ground_drop = np.zeros(0)          # per particle: the floor is this much lower for it (m)
        self.touched = np.zeros(0, bool)
        # distance / range constraints: i, j, lo, hi, compliance
        self._ci, self._cj = [], []
        self._clo, self._chi, self._calpha = [], [], []
        # bend-side limits (root, mid, end) and cone limits (root, child, half angle)
        self._bend = np.zeros((0, 3), int)
        self._bend_ref = np.zeros((0, 3))
        self._cone = np.zeros((0, 2), int)
        self._cone_cs = np.zeros((0, 2))
        self._cone_axis = np.zeros((0, 3))
        # pins, drive and centroid target
        self._pin_idx = np.zeros(0, int)
        self._pin_from = np.zeros((0, 3))
        self._pin_to = np.zeros((0, 3))
        self._pin_path = None
        self._drv_from = np.zeros((0, 3))
        self._drv_to = np.zeros((0, 3))
        self._drv_frac = np.zeros(0)
        self._drv_zeta = np.zeros(0)
        self._drv_cap = np.zeros(0)
        self._centroid = None
        self._dirty = True

    # --- particles --------------------------------------------------------------------------
    def add_particles(self, positions, mass, radius=0.0, group=1) -> np.ndarray:
        """Add particles at rest. mass (kg) and radius (m, collision sphere) may be scalars or
        per particle; mass = np.inf makes a particle kinematic. Returns their indices."""
        pos = np.atleast_2d(np.asarray(positions, float))
        n = len(pos)
        mass = np.broadcast_to(np.asarray(mass, float), (n,))
        start = len(self.x)
        self.x = np.vstack([self.x, pos])
        self.xp = np.vstack([self.xp, pos])
        self.w = np.concatenate([self.w, np.where(np.isfinite(mass), 1.0 / np.maximum(mass, 1e-12), 0.0)])
        self.radius = np.concatenate([self.radius, np.broadcast_to(np.asarray(radius, float), (n,))])
        self.group = np.concatenate([self.group, np.broadcast_to(np.asarray(group, int), (n,))])
        self.ground_drop = np.concatenate([self.ground_drop, np.zeros(n)])
        self.touched = np.zeros(len(self.x), bool)
        self._drv_from = np.vstack([self._drv_from, pos])
        self._drv_to = np.vstack([self._drv_to, pos])
        self._drv_frac = np.concatenate([self._drv_frac, np.zeros(n)])
        self._drv_zeta = np.concatenate([self._drv_zeta, np.full(n, DRIVE_DAMPING)])
        self._drv_cap = np.concatenate([self._drv_cap, np.full(n, np.inf)])
        self._dirty = True
        return np.arange(start, start + n)

    def set_state(self, indices, positions, velocities=None) -> None:
        """Teleport particles (velocities in m/s; default at rest)."""
        idx = np.asarray(indices, int)
        pos = np.asarray(positions, float)
        vel = np.zeros_like(pos) if velocities is None else np.asarray(velocities, float)
        self.x[idx] = pos
        self.xp[idx] = pos - vel * self.dt

    @property
    def velocities(self) -> np.ndarray:
        return (self.x - self.xp) / self.dt

    # --- constraints ------------------------------------------------------------------------
    def add_distance(self, i, j, length=None, compliance: float = 0.0, lo=None, hi=None) -> np.ndarray:
        """Distance constraints between particles i and j (arrays). With lo/hi it is a range
        (joint limit): the distance may move freely inside [lo, hi]. compliance in m/N
        (0 = rigid, as for bones; ~1e-6 for a stretchy rope). Returns constraint ids."""
        i, j = np.atleast_1d(np.asarray(i, int)), np.atleast_1d(np.asarray(j, int))
        if length is None:
            length = _norm(self.x[j] - self.x[i])
        length = np.broadcast_to(np.asarray(length, float), i.shape)
        lo = length if lo is None else np.broadcast_to(np.asarray(lo, float), i.shape)
        hi = length if hi is None else np.broadcast_to(np.asarray(hi, float), i.shape)
        start = len(self._ci)
        self._ci += i.tolist()
        self._cj += j.tolist()
        self._clo += np.asarray(lo, float).tolist()
        self._chi += np.asarray(hi, float).tolist()
        self._calpha += np.broadcast_to(np.asarray(compliance, float), i.shape).tolist()
        self._dirty = True
        return np.arange(start, start + len(i))

    def set_distance(self, ids, length=None, lo=None, hi=None) -> None:
        """Change rest lengths (or range bounds) of existing constraints."""
        ids = np.atleast_1d(np.asarray(ids, int))
        if self._dirty:
            self._build()
        lo = np.broadcast_to(np.asarray(length if lo is None else lo, float), ids.shape)
        hi = np.broadcast_to(np.asarray(length if hi is None else hi, float), ids.shape)
        for k, c in enumerate(ids):
            batch, slot = self._where[int(c)]
            self._batches[batch]["lo"][slot] = lo[k]
            self._batches[batch]["hi"][slot] = hi[k]

    def add_bend(self, root, mid, end) -> np.ndarray:
        """Bend-side limit for a hinge (knee, elbow): `mid` must stay on the side of the
        root->end line given by set_bend_reference (it may lie on the line: a straight limb)."""
        rows = np.stack([np.atleast_1d(root), np.atleast_1d(mid), np.atleast_1d(end)], axis=1).astype(int)
        start = len(self._bend)
        self._bend = np.vstack([self._bend, rows])
        self._bend_ref = np.vstack([self._bend_ref, np.zeros((len(rows), 3))])
        self._dirty = True
        return np.arange(start, start + len(rows))

    def set_bend_reference(self, ids, directions) -> None:
        self._bend_ref[np.asarray(ids, int)] = directions

    def add_cone(self, root, child, half_angle) -> np.ndarray:
        """Cone limit: the root->child direction stays within half_angle (rad) of an axis given
        by set_cone_axis (a ball joint's range of motion). It yields to the floor."""
        rows = np.stack([np.atleast_1d(root), np.atleast_1d(child)], axis=1).astype(int)
        ang = np.broadcast_to(np.asarray(half_angle, float), (len(rows),))
        start = len(self._cone)
        self._cone = np.vstack([self._cone, rows])
        self._cone_cs = np.vstack([self._cone_cs, np.column_stack([np.cos(ang), np.sin(ang)])])
        self._cone_axis = np.vstack([self._cone_axis, np.tile(UP, (len(rows), 1))])
        self._dirty = True
        return np.arange(start, start + len(rows))

    def set_cone_axis(self, ids, axes) -> None:
        self._cone_axis[np.asarray(ids, int)] = axes

    # --- pins, drive and centroid -------------------------------------------------------------
    def add_pin(self, indices) -> np.ndarray:
        """Make particles kinematic and moved by set_pin_targets. Returns pin ids."""
        idx = np.atleast_1d(np.asarray(indices, int))
        self.w[idx] = 0.0
        start = len(self._pin_idx)
        self._pin_idx = np.concatenate([self._pin_idx, idx])
        self._pin_from = np.vstack([self._pin_from, self.x[idx]])
        self._pin_to = np.vstack([self._pin_to, self.x[idx]])
        self._dirty = True
        return np.arange(start, start + len(idx))

    def set_pin_targets(self, pin_ids, end, start=None) -> None:
        """Pinned particles move linearly from `start` (default: where they are) to `end`
        over the next step."""
        pin_ids = np.asarray(pin_ids, int)
        self._pin_to[pin_ids] = end
        self._pin_from[pin_ids] = self.x[self._pin_idx[pin_ids]] if start is None else start

    def set_pin_path(self, path) -> None:
        """Exact pinned positions at the end of each substep of the next step: (substeps, pins, 3)."""
        self._pin_path = np.asarray(path, float)

    def set_drive(self, indices, end, fraction, start=None, damping=None, max_speed=None) -> None:
        """Soft pose drive: each substep removes `fraction` (0..1) of the offset between a
        particle and its target, which moves linearly from `start` (default `end`) to `end`
        over the next step, and damps the particle's velocity toward the target's by
        min(1, 2 * damping * sqrt(fraction)) of the difference: a spring with damping ratio
        `damping` (default DRIVE_DAMPING = 1, critical: no overshoot). fraction 1 tracks the
        target exactly; under gravity alone a particle hangs about
        GRAVITY * dt^2 * (1 - fraction) / fraction below it. max_speed (m/s, default none):
        the particle closes in on its target at most this fast (relative to the target's own
        motion), so a limb held back by an obstacle does not shoot to its pose when it slips
        free; motion away from or across the target is not limited."""
        idx = np.asarray(indices, int)
        self._drv_to[idx] = end
        self._drv_from[idx] = end if start is None else start
        self._drv_frac[idx] = fraction
        self._drv_zeta[idx] = DRIVE_DAMPING if damping is None else damping
        self._drv_cap[idx] = np.inf if max_speed is None else max_speed

    def set_centroid(self, indices, weights, end, start=None, share=None, stiffness: float = 1.0,
                     max_step: float = 0.01) -> None:
        """Keep sum(weights * x) / sum(weights) of these (dynamic) particles on a target moving
        linearly from `start` to `end` over the next step (a linear momentum constraint,
        solved after the distance and limit constraints). Each substep moves particle i by
        share_i * s, with s chosen so the weighted centre lands on the target (|s| at most
        `max_step` metres). share defaults to 1 (a uniform shift); larger shares on distal
        particles of a chain make the next substep's bones turn the shift into joint flexion,
        which also works along a hanging limb. stiffness (0..1] is the fraction of the error
        removed per substep (below 1 a soft constraint that fights the bones less at the
        substep rate). Lasts until clear_centroid()."""
        idx = np.asarray(indices, int)
        weights = np.asarray(weights, float) / float(np.sum(weights))
        share = np.ones(len(idx)) if share is None else np.asarray(share, float)
        wfull = np.zeros(len(self.x))
        wfull[idx] = weights
        move = np.zeros((len(self.x), 1))
        move[idx, 0] = share * (self.w[idx] > 0)
        end = np.asarray(end, float)
        self._centroid = {"w": wfull, "move": move, "gain": float(stiffness) / max(float(weights @ move[idx, 0]), 1e-9),
                          "from": end if start is None else np.asarray(start, float), "to": end, "max": float(max_step)}

    def clear_centroid(self) -> None:
        self._centroid = None

    # --- stepping ---------------------------------------------------------------------------
    def _build(self) -> None:
        """Pack constraints into batches; a particle moves in at most one constraint per batch
        (kinematic particles never move, so they may repeat). Small worlds also get each
        batch's scatter matrix, so a batch applies its corrections with one matmul."""
        n = len(self.x)
        ci, cj = np.array(self._ci, int), np.array(self._cj, int)
        clo, chi, calpha = np.array(self._clo), np.array(self._chi), np.array(self._calpha)
        used: list[set] = []
        members: list[list] = []
        for c in range(len(ci)):
            movers = {int(p) for p in (ci[c], cj[c]) if self.w[p] > 0}
            for b, u in enumerate(used):
                if not (movers & u):
                    break
            else:
                used.append(set())
                members.append([])
                b = len(used) - 1
            used[b] |= movers
            members[b].append(c)
        self._batches, self._where = [], {}
        for b, cs in enumerate(members):
            cs = np.array(cs, int)
            i, j = ci[cs], cj[cs]
            wi, wj = self.w[i], self.w[j]
            denom = wi + wj + calpha[cs] / self.dt ** 2
            inv = np.where(denom > 0, 1.0 / np.where(denom > 0, denom, 1.0), 0.0)
            scatter = None
            if n <= DENSE_MAX:
                scatter = np.zeros((n, len(cs)))
                cols = np.arange(len(cs))
                scatter[i, cols] += wi
                scatter[j, cols] -= wj
            self._batches.append({"i": i, "j": j, "lo": clo[cs].copy(), "hi": chi[cs].copy(), "inv": inv,
                                  "wi": wi[:, None], "wj": wj[:, None], "scatter": scatter})
            for slot, c in enumerate(cs):
                self._where[int(c)] = (b, slot)
        dyn = self.w > 0
        self._dyn_idx = np.flatnonzero(dyn)
        self._dyn_col = dyn[:, None].astype(float)
        self._ground_ok = dyn & ((self.group & self.col.ground_mask) != 0)
        self._bend_wm = self.w[self._bend[:, 1]] > 0 if len(self._bend) else np.zeros(0, bool)
        self._cone_wc = self.w[self._cone[:, 1]] > 0 if len(self._cone) else np.zeros(0, bool)
        self._rmax = float(self.radius.max()) if n else 0.0
        self._dirty = False

    def _broadphase(self, frame_dt: float):
        """Shapes the dynamic particles can reach within this step: (ground limit per particle
        or None, [(candidate particles, box...)], [(candidate particles, capsule...)], sheet ids)."""
        col = self.col
        idx = self._dyn_idx
        if not len(idx):
            return None, [], [], []
        x = self.x[idx]
        speed = float(np.sqrt(_dot(x - self.xp[idx], x - self.xp[idx]).max())) / self.dt
        pin_move = float(np.abs(self._pin_to - self._pin_from).max()) if len(self._pin_idx) else 0.0
        reach = max(speed * frame_dt, pin_move) + pin_move + self.gravity * frame_dt ** 2 + self._rmax + 0.05
        lo, hi = x.min(axis=0) - reach, x.max(axis=0) + reach
        ground = None
        if col.ground_y is not None and lo[1] <= col.ground_y:
            ground = np.where(self._ground_ok, col.ground_y + self.radius - self.ground_drop, -np.inf)
        # Per shape: the dynamic particles of a matching group whose reach box overlaps it.
        plo, phi = self.x[idx] - reach, self.x[idx] + reach
        group = self.group[idx]

        def candidates(mask, shape_lo, shape_hi):
            return idx[((group & mask) != 0) & np.all(phi >= shape_lo, axis=1) & np.all(plo <= shape_hi, axis=1)]
        boxes = []
        if col.n_boxes:
            c, R, h, mu, mask, deep, blo, bhi = col.box_arrays()
            for b in np.flatnonzero(np.all(bhi >= lo, axis=1) & np.all(blo <= hi, axis=1)):
                cand = candidates(mask[b], blo[b], bhi[b])
                if len(cand):
                    boxes.append((cand, c[b], R[b], h[b], mu[b], deep[b], bool(mask[b] & self.touch_mask)))
        caps = []
        if col.n_caps:
            a, ab, rad, mu, mask, a0, ab0 = col.cap_arrays()
            ends = np.stack([a, a + ab, a0, a0 + ab0])                 # both poses of the step
            clo = ends.min(axis=0) - rad[:, None]
            chi = ends.max(axis=0) + rad[:, None]
            for k in np.flatnonzero(np.all(chi >= lo, axis=1) & np.all(clo <= hi, axis=1)):
                cand = candidates(mask[k], clo[k], chi[k])
                if len(cand):
                    caps.append((cand, a0[k].copy(), a[k] - a0[k], ab0[k].copy(), ab[k] - ab0[k], rad[k], mu[k],
                                 bool(mask[k] & self.touch_mask)))
        sheets = []
        for sid, sh in enumerate(col.sheets):
            half = abs(sh["r"]) * sh["half"][0] + abs(sh["f"]) * sh["half"][1]
            # vertical range: the surface down to a 2 m dent (a safety net under load)
            centre, flat = sh["c"], [0, 2]
            if (lo[1] <= centre[1] and hi[1] >= centre[1] - sh["thick"] - 2.0
                    and np.all(centre[flat] + half[flat] >= lo[flat])
                    and np.all(centre[flat] - half[flat] <= hi[flat])):
                sheets.append((sid, (self.w > 0) & ((self.group & sh["mask"]) != 0)))
        return ground, boxes, caps, sheets

    def step(self, frame_dt: float) -> None:
        """Advance by frame_dt seconds in fixed substeps (frame_dt is rounded to a whole number
        of substeps)."""
        if self._dirty:
            self._build()
        n_sub = max(1, int(round(frame_dt / self.dt)))
        dyn = self._dyn_col
        keep_v = dyn * max(0.0, 1.0 - self.air_damping * self.dt)
        fall = np.array([0.0, -self.gravity * self.dt ** 2, 0.0]) * dyn
        driven = bool(self._drv_frac.any())
        if driven:
            pull = self._drv_frac[:, None] * dyn
            damp = np.minimum(1.0, 2.0 * self._drv_zeta * np.sqrt(self._drv_frac))[:, None] * dyn
            drv0, drv_d = self._drv_from, self._drv_to - self._drv_from
            drv_v = drv_d / n_sub                    # the target's motion per substep
            cap = self._drv_cap * self.dt            # largest approach per substep (m)
            capped = bool(np.isfinite(cap).any())
            cap = np.where(np.isfinite(cap), cap, 1e9)
        pinned = len(self._pin_idx) > 0
        pin_path = self._pin_path
        pin0, pin_d = self._pin_from, self._pin_to - self._pin_from
        cen = self._centroid
        shapes = self._broadphase(frame_dt)
        self.touched = np.zeros(len(self.x), bool)
        x, xp = self.x, self.xp
        for s in range(n_sub):
            a = (s + 1) / n_sub
            v = (x - xp) * keep_v
            if driven:
                v += damp * (drv_v - v)
            p = x + v + fall
            xp = x
            if driven:
                goal = drv0 + drv_d * a
                p += pull * (goal - p)
                if capped:
                    # approach toward the target this substep, beyond the target's own motion
                    toward = unit_rows(goal - x)
                    excess = np.maximum(_dot(p - x - drv_v, toward) - cap, 0.0)
                    p -= toward * excess[:, None]
            if pinned:
                p[self._pin_idx] = pin_path[s] if pin_path is not None else pin0 + pin_d * a
            for _ in range(self.iterations):
                for batch in self._batches:
                    self._solve_distance(p, batch)
                if len(self._bend):
                    self._solve_bend(p, xp)
                if len(self._cone):
                    self._solve_cone(p, xp)
                if cen is not None:
                    self._solve_centroid(p, cen, a)
                self._collide(p, xp, shapes, a)
            x = p
        self.x, self.xp = x, xp
        self.col.end_step()
        self._pin_from = self._pin_to.copy()
        self._pin_path = None
        self._drv_from = self._drv_to.copy()
        if cen is not None:
            cen["from"] = cen["to"]

    # --- constraint solvers -----------------------------------------------------------------
    @staticmethod
    def _solve_distance(p: np.ndarray, batch: dict) -> None:
        d = p.take(batch["j"], axis=0) - p.take(batch["i"], axis=0)
        length = np.sqrt(_dot(d, d))
        err = length - np.minimum(np.maximum(length, batch["lo"]), batch["hi"])
        corr = d * (err * batch["inv"] / np.maximum(length, 1e-9))[:, None]
        if batch["scatter"] is not None:
            p += batch["scatter"] @ corr
        else:
            p[batch["i"]] += corr * batch["wi"]
            p[batch["j"]] -= corr * batch["wj"]

    @staticmethod
    def _solve_centroid(p: np.ndarray, cen: dict, a: float) -> None:
        goal = cen["from"] + (cen["to"] - cen["from"]) * a
        shift = (goal - cen["w"] @ p) * cen["gain"]
        size = float(np.sqrt(shift @ shift))
        if size > cen["max"]:
            shift *= cen["max"] / size
        p += cen["move"] * shift

    @staticmethod
    def _limit_move(p: np.ndarray, xp: np.ndarray, idx: np.ndarray, delta: np.ndarray) -> None:
        """Apply a joint-limit correction as an inelastic stop: move the particles by delta,
        keep their velocity (xp moves too, so the correction adds none) and remove the part of
        it that points against the correction (further past the limit). A limit violated by a
        lot is then worked off over a few substeps without launching the limb."""
        p[idx] += delta
        xp[idx] += delta
        size = _norm(delta)
        normal = delta / np.maximum(size, 1e-12)[:, None]
        into = np.minimum(_dot(p[idx] - xp[idx], normal), 0.0) * (size > 1e-12)
        xp[idx] += normal * into[:, None]

    def _solve_bend(self, p: np.ndarray, xp: np.ndarray) -> None:
        root, mid, end = self._bend[:, 0], self._bend[:, 1], self._bend[:, 2]
        pa = p.take(root, axis=0)
        u = unit_rows(p.take(end, axis=0) - pa)
        off = p.take(mid, axis=0) - pa
        off -= u * _dot(off, u)[:, None]
        ref = self._bend_ref - u * _dot(self._bend_ref, u)[:, None]
        ref_len = _norm(ref)
        side = _dot(off, ref) / np.maximum(ref_len, 1e-9)
        push = np.where((side < 0) & (ref_len > 1e-3) & self._bend_wm, np.minimum(-side, BEND_STEP_M), 0.0)
        if push.any():
            k = np.flatnonzero(push > 0)
            self._limit_move(p, xp, mid[k], ref[k] * (push[k] / ref_len[k])[:, None])

    def _solve_cone(self, p: np.ndarray, xp: np.ndarray) -> None:
        """Turn each violating child back toward its cone by at most CONE_STEP_RAD per substep
        (inelastic, see _limit_move). The floor wins over the limit: a turn that would end
        below the floor is swung up about the root to the floor height instead (the limb lies
        on the floor at the nearest allowed angle), so violations cannot build up there."""
        root, child = self._cone[:, 0], self._cone[:, 1]
        d = p.take(child, axis=0) - p.take(root, axis=0)
        length = _norm(d)
        axis = self._cone_axis
        along = _dot(d, axis)
        viol = (along < self._cone_cs[:, 0] * length) & self._cone_wc & (length > 1e-9)
        if not viol.any():
            return
        v = np.flatnonzero(viol)
        axis, length = axis[v], length[v]
        perp = d[v] - axis * along[v][:, None]
        perp_len = _norm(perp)
        if (perp_len <= 1e-9).any():       # exactly opposite the axis: any perpendicular will do
            perp = np.where((perp_len > 1e-9)[:, None], perp, np.cross(axis, np.array([1.0, 0.0, 0.0])))
            perp_len = _norm(perp)
        perp /= np.maximum(perp_len, 1e-12)[:, None]
        angle = np.arctan2(perp_len, along[v])
        half = np.arctan2(self._cone_cs[v, 1], self._cone_cs[v, 0])
        turn = np.maximum(angle - CONE_STEP_RAD, half)
        u = axis * np.cos(turn)[:, None] + perp * np.sin(turn)[:, None]
        base = p[root[v]]
        if self.col.ground_y is not None:
            # A joint limit is soft tissue, the floor is not: never turn a limb into the floor.
            floor = self.col.ground_y + self.radius[child[v]] - self.ground_drop[child[v]]
            need = (floor - base[:, 1]) / np.maximum(length, 1e-9)
            low = self._ground_ok[child[v]] & (u[:, 1] < need)
            if low.any():
                uy = np.minimum(need[low], 1.0)
                old = d[v][low] / length[low][:, None]
                flat = u[low][:, [0, 2]]
                flat = np.where((np.abs(flat).sum(axis=1) > 1e-9)[:, None], flat, old[:, [0, 2]] + [1e-9, 0.0])
                flat /= np.sqrt(np.sum(flat * flat, axis=1))[:, None]
                scale = np.sqrt(np.maximum(1.0 - uy * uy, 0.0))
                u[low] = np.column_stack([flat[:, 0] * scale, uy, flat[:, 1] * scale])
                stuck = need[low] > 1.0                     # root itself too low: leave the child
                if stuck.any():
                    rows = np.flatnonzero(low)[stuck]
                    u[rows] = old[stuck]
        self._limit_move(p, xp, child[v], base + u * length[:, None] - p[child[v]])

    # --- collisions -------------------------------------------------------------------------
    @staticmethod
    def _friction(p: np.ndarray, xp: np.ndarray, idx: np.ndarray, normal: np.ndarray, depth: np.ndarray, mu) -> None:
        """Coulomb-like: cancel this substep's tangential motion if it is below mu * depth
        (static), otherwise reduce it by 0.8 mu * depth (kinetic)."""
        dp = p[idx] - xp[idx]
        tang = dp - normal * _dot(dp, normal)[:, None]
        slide = _norm(tang)
        limit = mu * np.minimum(depth, FRICTION_DEPTH_CAP_M)
        keep = np.where(slide <= limit, 1.0, np.minimum(0.8 * limit / np.maximum(slide, 1e-12), 1.0))
        p[idx] -= tang * keep[:, None]

    def _push_out(self, p: np.ndarray, xp: np.ndarray, idx: np.ndarray, normal: np.ndarray, depth: np.ndarray,
                  mu) -> None:
        """Push particles out along normal by depth (at most Colliders.max_push), with friction
        and no rebound; a particle left inside also loses its velocity into the shape."""
        push = np.minimum(depth, self.col.max_push)
        p[idx] += normal * push[:, None]
        self._friction(p, xp, idx, normal, push, mu)
        self._stop_rebound(p, xp, idx, normal)
        deep = push < depth
        if deep.any():
            rows, n = idx[deep], normal[deep]
            xp[rows] += n * np.minimum(_dot(p[rows] - xp[rows], n), 0.0)[:, None]

    @staticmethod
    def _stop_rebound(p: np.ndarray, xp: np.ndarray, idx: np.ndarray, normal: np.ndarray) -> None:
        """Restitution 0: a contact leaves no velocity away from the surface. Pushing a deep
        penetration out would otherwise launch the particle (implicit v = (p - xp) / dt)."""
        out = _dot(p[idx] - xp[idx], normal)
        xp[idx] += normal * np.maximum(out, 0.0)[:, None]

    def _collide(self, p: np.ndarray, xp: np.ndarray, shapes, alpha: float) -> None:
        """Collisions at a fraction alpha through the step (moving shapes are interpolated)."""
        ground, boxes, caps, sheets = shapes
        if ground is not None:
            self._collide_ground(p, xp, ground)
        for box in boxes:
            self._collide_box(p, xp, *box)
        for cand, a0, da, ab0, dab, rad, mu, touch in caps:
            self._collide_capsule(p, xp, cand, a0 + da * alpha, ab0 + dab * alpha, rad, mu, touch)
        for sid, allowed in sheets:
            self._collide_sheet(p, xp, sid, allowed, alpha)

    def _collide_ground(self, p: np.ndarray, xp: np.ndarray, limit: np.ndarray) -> None:
        """Floor plane; whole-array masks (cheaper than index lists for small worlds)."""
        depth = limit - p[:, 1]
        hit = depth > 0
        if not hit.any():
            return
        depth = np.where(hit, depth, 0.0)
        p[:, 1] += depth
        tx, tz = p[:, 0] - xp[:, 0], p[:, 2] - xp[:, 2]
        slide = np.sqrt(tx * tx + tz * tz)
        lim = self.col.ground_mu * np.minimum(depth, FRICTION_DEPTH_CAP_M)
        keep = np.where(hit, np.where(slide <= lim, 1.0, np.minimum(0.8 * lim / np.maximum(slide, 1e-12), 1.0)), 0.0)
        p[:, 0] -= tx * keep
        p[:, 2] -= tz * keep
        xp[:, 1] = np.where(hit & (p[:, 1] > xp[:, 1]), p[:, 1], xp[:, 1])     # no rebound velocity
        if self.col.ground_mask & self.touch_mask:
            self.touched |= hit

    def _collide_box(self, p: np.ndarray, xp: np.ndarray, cand, centre, rot, half, mu, deep, touch) -> None:
        """Push candidate particles out through the nearest face of one oriented box."""
        local = (p[cand] - centre) @ rot
        gap = half + self.radius[cand][:, None] - np.abs(local)
        depth = gap.min(axis=1)
        ok = (depth > 0) & (depth < deep)
        if not ok.any():
            return
        idx, gap, local, depth = cand[ok], gap[ok], local[ok], depth[ok]
        axis = gap.argmin(axis=1)
        sign = np.where(local[np.arange(len(idx)), axis] >= 0, 1.0, -1.0)
        normal = rot[:, axis].T * sign[:, None]
        self._push_out(p, xp, idx, normal, depth, mu)
        if touch:
            self.touched[idx] = True

    def _collide_capsule(self, p: np.ndarray, xp: np.ndarray, cand, a, ab, rad, mu, touch) -> None:
        """Push candidate particles radially out of one capsule."""
        rel = p[cand] - a
        u = np.clip((rel @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        d = rel - u[:, None] * ab
        dist = _norm(d)
        depth = rad + self.radius[cand] - dist
        ok = (depth > 0) & (dist > 1e-6)
        if not ok.any():
            return
        idx = cand[ok]
        normal = d[ok] / dist[ok][:, None]
        self._push_out(p, xp, idx, normal, depth[ok], mu)
        if touch:
            self.touched[idx] = True

    def _collide_sheet(self, p: np.ndarray, xp: np.ndarray, sid: int, allowed: np.ndarray, alpha: float) -> None:
        """One-sided horizontal sheet with its dent; whole-array masks like the ground.
        `allowed`: dynamic particles of a matching group."""
        sh = self.col.sheets[sid]
        top = self.col.sheet_top(sid, p[:, 0], p[:, 2], alpha)
        depth = top + self.radius - p[:, 1]
        ok = (depth > 0) & (depth < sh["thick"]) & allowed
        if not ok.any():
            return
        rel = p[ok] - sh["c"]
        ok[ok] = (np.abs(rel @ sh["r"]) <= sh["half"][0]) & (np.abs(rel @ sh["f"]) <= sh["half"][1])
        if not ok.any():
            return
        depth = np.where(ok, depth, 0.0)
        p[:, 1] += depth
        tx, tz = p[:, 0] - xp[:, 0], p[:, 2] - xp[:, 2]
        slide = np.sqrt(tx * tx + tz * tz)
        lim = sh["mu"] * np.minimum(depth, FRICTION_DEPTH_CAP_M)
        keep = np.where(ok, np.where(slide <= lim, 1.0, np.minimum(0.8 * lim / np.maximum(slide, 1e-12), 1.0)), 0.0)
        p[:, 0] -= tx * keep
        p[:, 2] -= tz * keep
        xp[:, 1] = np.where(ok & (p[:, 1] > xp[:, 1]), p[:, 1], xp[:, 1])     # no rebound velocity
        if sh["mask"] & self.touch_mask:
            self.touched |= ok


# =============================================================================================
# Scene geometry for the physics
# =============================================================================================
GROUND_MU = {"dry": 0.7, "wet": 0.45, "icy": 0.15, "oily": 0.25, "muddy": 0.4, "dusty": 0.55}
SCENE_MASK, ARM_MASK = 1, 2      # scene shapes; the moving trunk capsule (hits arms only)


def support_points(scene, x: np.ndarray, z: np.ndarray, below: np.ndarray) -> np.ndarray:
    """Vectorised Scene.support: top of the highest walkable under each (x, z) that is not
    above `below`, else the ground."""
    x, z, below = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float), np.asarray(below, float))
    best = np.full(x.shape, float(scene.ground_y))
    for w in scene.walkables:
        r, f = right(w.yaw), fwd(w.yaw)
        dx, dz = x - w.centre[0], z - w.centre[2]
        a = dx * r[0] + dz * r[2]
        b = dx * f[0] + dz * f[2]
        ok = (np.abs(a) <= w.half[0]) & (np.abs(b) <= w.half[1]) & (w.y <= below + 1e-6) & (w.y > best)
        best = np.where(ok, w.y, best)
    return best


def scene_colliders(scene, lo: np.ndarray | None = None, hi: np.ndarray | None = None) -> Colliders:
    """Colliders for the drawn scene: ground, boxes and slabs, tubes (rails, scaffold), ladder
    rails and rungs, roof/bank wedges and the safety net. With lo/hi (world AABB corners), only
    shapes that reach into that box are kept."""
    col = Colliders(scene.ground_y, GROUND_MU.get(str(scene.condition), 0.7))

    def near(points, pad) -> bool:
        if lo is None:
            return True
        pts = np.atleast_2d(points)
        return bool(np.all(pts.max(axis=0) + pad >= lo) and np.all(pts.min(axis=0) - pad <= hi))

    for prop in scene.props:
        kind = prop.get("t")
        if kind == "box":
            c, s = np.asarray(prop["p"], float), np.asarray(prop["s"], float)
            if near(c, 0.5 * float(np.linalg.norm(s))):
                col.add_box(c, s, yaw=float(prop.get("yaw", 0.0)), mu=0.6, mask=SCENE_MASK)
        elif kind == "tube":
            a, b = np.asarray(prop["a"], float), np.asarray(prop["b"], float)
            if near(np.stack([a, b]), float(prop["r"])):
                col.add_capsule(a, b, float(prop["r"]), mu=0.5, mask=SCENE_MASK)
        elif kind == "ladder":
            base, yaw, w = np.asarray(prop["base"], float), float(prop["yaw"]), float(prop.get("w", 0.46))
            height = float(prop["top_y"]) - base[1]
            f, r = fwd(yaw), right(yaw)
            top = base + f * height / 4.0 + UP * height
            if not near(np.stack([base, top]), w):
                continue
            for side in (-1.0, 1.0):
                col.add_capsule(base + r * side * w / 2, top + r * side * w / 2, 0.03, mu=0.5, mask=SCENE_MASK)
            for y in np.arange(0.3, height - 1e-6, 0.3):  # rungs every 0.3 m on a 1:4 lean (world.js ladderMesh)
                c = base + f * y / 4.0 + UP * y
                col.add_capsule(c - r * w / 2, c + r * w / 2, 0.016, mu=0.5, mask=SCENE_MASK)
        elif kind == "wedge":
            # Sloped slab (roof, trench bank): top edge at p, descending along fwd(yaw) by `slope`.
            p0, yaw = np.asarray(prop["p"], float), float(prop["yaw"])
            slope = np.deg2rad(float(prop["slope"]))
            f, r = fwd(yaw), right(yaw)
            along = f * np.cos(slope) - UP * np.sin(slope)
            normal = UP * np.cos(slope) + f * np.sin(slope)
            length, thick = float(prop["len"]) / max(np.cos(slope), 1e-3), 0.3
            centre = p0 + along * length / 2 - normal * thick / 2
            if near(centre, length / 2 + float(prop["w"]) / 2):
                col.add_box(centre, (float(prop["w"]), thick, length), rotation=np.column_stack([r, normal, along]),
                            mu=0.5, mask=SCENE_MASK)
        elif kind == "net":
            y = float(scene.dyn.get("net", {}).get("y", prop["p"][1]))
            centre = np.array([prop["p"][0], y, prop["p"][2]])
            col.add_sheet(centre, float(prop["yaw"]), (float(prop["w"]), float(prop["l"])), mu=0.8, mask=SCENE_MASK)
    return col


# =============================================================================================
# Body model shared by the ragdoll, the COM target and the rest lift
# =============================================================================================
# Particles are the 18 rig joints in rig.JOINTS order.
PINNED = (0, 1, 10, 11, 16)                       # hips, shoulders, neck: rigid on the simulated spine
LEGS = (2, 3, 4, 5, 6, 7, 8, 9)                   # knees, ankles, heels, toes
ARMS = (12, 13, 14, 15)                           # elbows, hands
HEAD = (17,)
FOOT_JOINTS = (4, 6, 8, 5, 7, 9)                  # ankle, heel, toe: left then right
DYNAMIC = LEGS + ARMS + HEAD
LIMB_ROOT = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 10, 11, 10, 11, 10, 11, 16, 16])   # hip, shoulder or neck
# Segment masses, fraction of body mass (de Leva 1996, male). Each limb segment is lumped half
# onto each end joint (the foot a third each on ankle, heel and toe); the rest is the trunk.
THIGH, SHANK, FOOT, UPPER_ARM, FOREARM, HAND, HEAD_M = 0.1416, 0.0433, 0.0137, 0.0271, 0.0162, 0.0061, 0.0694
TRUNK_M = 1.0 - HEAD_M - 2 * (THIGH + SHANK + FOOT + UPPER_ARM + FOREARM + HAND)
JOINT_MASS = np.array([THIGH / 2] * 2 + [THIGH / 2 + SHANK / 2] * 2 + [SHANK / 2 + FOOT / 3] * 2 + [FOOT / 3] * 4
                      + [UPPER_ARM / 2] * 2 + [UPPER_ARM / 2 + FOREARM / 2] * 2 + [FOREARM / 2 + HAND] * 2
                      + [0.0, HEAD_M])
# Simulator centre of mass relative to the pelvis anchor in the root frame (falls.Faller.r_p):
# COM = pelvis - R_root @ (0, -0.03 H, 0.10).
SIM_COM_R_P = (0.0, -0.03, 0.10)
# body.js torso: half width a and half depth b along the spine (fraction of the curve), times s*w
# and s*sqrt(w); the tube centre is (b - 0.04) m anterior of the spine line.
TORSO_PROFILE = np.array([[0.0, 0.135, 0.105], [0.07, 0.172, 0.13], [0.2, 0.165, 0.118], [0.32, 0.148, 0.104],
                          [0.45, 0.158, 0.112], [0.62, 0.173, 0.124], [0.8, 0.184, 0.12], [0.88, 0.196, 0.112],
                          [0.94, 0.16, 0.098]])


def _torso_rings(P: np.ndarray, M: np.ndarray, H: float, s: float, w: float):
    """Cross-sections of the drawn torso (body.js), as straight segments along bottom ->
    pelvis -> lumbar -> thoracic -> upper -> neck instead of the Hermite curve. Yields per
    ring: spine-line point, ellipse centre, half width, half depth, lateral and posterior axes."""
    bottom = P[:, 0] + M[:, 0, :, 1] * (-0.085 * H)
    neck = P[:, 3] + M[:, 3, :, 1] * (0.03 * H)
    pts = np.stack([bottom, P[:, 0], P[:, 1], P[:, 2], P[:, 3], neck], axis=1)        # (N, 6, 3)
    seg_of_pt = (0, 0, 1, 2, 3, 3)
    knots = np.array([0.0, 0.2, 0.4, 0.62, 0.9, 1.0])
    for sr, a0, b0 in TORSO_PROFILE:
        k = int(np.clip(np.searchsorted(knots, sr, side="right") - 1, 0, 4))
        u = (sr - knots[k]) / (knots[k + 1] - knots[k])
        line = pts[:, k] + (pts[:, k + 1] - pts[:, k]) * u
        seg = seg_of_pt[k] if u < 0.5 else seg_of_pt[k + 1]
        xr, zr = M[:, seg, :, 0], M[:, seg, :, 2]
        a, bd = a0 * s * w, b0 * s * np.sqrt(w)
        yield line, line - zr * (bd - 0.04), a, bd, xr, zr


def torso_centroid(P: np.ndarray, M: np.ndarray, H: float, s: float, w: float) -> np.ndarray:
    """Centroid of the drawn torso (N, 3): ring centres weighted by their ellipse areas."""
    total, weight = np.zeros((len(P), 3)), 0.0
    for _, centre, a, bd, _, _ in _torso_rings(P, M, H, s, w):
        total += centre * (a * bd)
        weight += a * bd
    return total / weight


def _ring_low(centre: np.ndarray, a: float, bd: float, xr: np.ndarray, zr: np.ndarray) -> np.ndarray:
    """Lowest point (y) of one torso cross-section ellipse; the back is flattened (body.js)."""
    back = np.where(zr[:, 1] < 0, 0.92, 1.0)
    return centre[:, 1] - np.sqrt((a * xr[:, 1]) ** 2 + (bd * back * zr[:, 1]) ** 2)


def torso_bottom(P: np.ndarray, M: np.ndarray, H: float, s: float, w: float) -> np.ndarray:
    """Lowest point (y) of the drawn torso per frame (N,)."""
    low = np.full(len(P), np.inf)
    for _, centre, a, bd, xr, zr in _torso_rings(P, M, H, s, w):
        low = np.minimum(low, _ring_low(centre, a, bd, xr, zr))
    return low


def body_com(joints: np.ndarray, P: np.ndarray, M: np.ndarray, H: float, s: float, w: float) -> np.ndarray:
    """Whole-body centre of mass of the drawn figure (N, 3): de Leva segment masses on the
    rendered joints plus the trunk mass at the drawn torso's centroid."""
    return TRUNK_M * torso_centroid(P, M, H, s, w) + np.einsum("j,njk->nk", JOINT_MASS, joints)


# =============================================================================================
# Limb ragdoll: muscle tone schedule
# =============================================================================================
# Collision radii as a fraction of the size scale s (body.js draws knee 0.058 s, hand 0.047 s,
# elbow 0.043 s, skull 0.08-0.1 s); heel and toe are points on the sole line.
RADIUS = {2: 0.055, 4: 0.045, 6: 0.012, 8: 0.012, 12: 0.042, 14: 0.045, 17: 0.085}
# Lowest allowed joint centre above the support in the final pass (m).
CLEAR = {2: 0.04, 4: 0.03, 6: 0.0, 8: 0.0, 12: 0.03, 14: 0.03, 17: 0.075}
# Drive: the per-substep fraction is F_LIMP ** (1 - ctrl); F_LIMP lets a limb hang ~0.8 m
# below its target, i.e. no tone at all. ctrl 0.1 ~ 33 cm, 0.5 ~ 1 cm, 1 = exact.
LIMP_SAG_M = 0.8
# Muscle tone (legs, arms, head) per state.
TONE = {
    "normal": (1.0, 1.0, 1.0),
    "falling": (0.30, 0.60, 0.60),          # conscious topple: some bracing
    "collapse": (0.25, 0.15, 0.20),         # fainting sag
    "air": (0.25, 0.30, 0.50),              # tumbling fall
    "air_feet": (0.60, 0.30, 0.50),         # feet-first fall: legs held for the landing
    "landing": (0.80, 0.60, 0.85),          # feet-first touchdown: legs take the load
    "jump": (0.80, 0.80, 1.0),              # voluntary jump: takeoff and landing
    "jump_air": (0.55, 0.55, 0.80),         # voluntary jump: in the air
    "down_unconscious": (0.08, 0.08, 0.10),
    "down_conscious": (0.15, 0.20, 0.45),   # conscious but still (or stunned just after impact)
    "down_struggle": (1.0, 1.0, 1.0),       # moving on their own again: the procedural pose drives
    "hang_unconscious": (0.10, 0.10, 0.12),
    "hang_conscious": (0.25, 0.50, 0.60),
    "hang_struggle": (1.0, 1.0, 1.0),
    "rest": (0.25, 0.30, 0.60),             # voluntary lying, conscious
    "rest_asleep": (0.10, 0.12, 0.30),
}
RAMP_DOWN_S, RAMP_UP_S = 0.25, 0.8     # time for a full 1 -> 0 drop / 0 -> 1 recovery of tone
STUN_S = 1.5                           # limp after an impact or arrest before conscious movement resumes
# Drive damping ratio: critical from DAMPED_TONE up; a limp limb (tone 0) keeps LIMP_DAMPING so it
# still drops and swings nearly freely, but no longer rings around its pose for seconds.
LIMP_DAMPING, DAMPED_TONE = 0.4, 0.3
# When tone comes back from below RISE_FROM (waking up, struggling, getting up), the drive target
# moves from the ragdoll's pose (carried with the trunk) to the procedural pose over RISE_S
# (smoothstep in time), so the limbs move there at a human pace instead of snapping.
RISE_FROM, RISE_S = 0.5, 0.8
# A limb group with tone >= RISE_FROM, or cross-fading to its pose, closes in on its drive target
# at most this fast (relative to the target's motion): held back by an obstacle (the trunk, a
# rung, the floor) it does not shoot to its pose when it slips free.
DRIVE_MAX_SPEED = 2.5
TAKEOFF_S = 0.25                       # contacts this soon after takeoff (feet leaving the edge) do not end a flight
# How far each joint moves for a whole-body COM correction in flight: distal joints move most,
# so the bones turn a correction into knee/elbow/hip/shoulder rotation (per joint, rig order).
CENTROID_SHARE = np.array([0, 0, 0.5, 0.5, 1, 1, 1, 1, 1, 1, 0, 0, 0.5, 0.5, 1, 1, 0, 1.0])
CENTROID_STIFFNESS = 0.3               # per substep: a soft hold that fights the bones less
CENTROID_MAX_STEP_M = 0.004            # largest COM correction per substep
# The hold fades in after takeoff and out after a limb touches something after takeoff (s).
CENTROID_FADE_IN_S, CENTROID_FADE_OUT_S = 0.04, 0.1
# Props (boxes, tubes, the trunk capsule) push a limb out by at most this per substep (1.25 m/s):
# a limb dragged into a wall or over an edge by the trunk slides out instead of popping.
PROP_PUSH_M = 0.005
HEAD_STEP_DEG = 8.0                    # largest head turn per frame relative to the upper thorax (physics windows)
# Tone at or above which the drive tracks the pose within ~0.2 mm: no simulation needed, and a
# limb group that tone may follow its pose while the rest of the ragdoll sleeps.
FULL_TONE = 0.9
# Passive ankle stiffness: heel and toe are pulled toward the posed foot carried on the
# simulated shank with at least this tone (the foot never spins about the shank axis).
FOOT_TONE = 0.5
NET_SAG_M = 0.04                       # the ragdoll's net sheet: extra sag under the body (set_dent sag)
MARGIN_BEFORE_S, MARGIN_AFTER_S, MERGE_GAP_S = 0.3, 0.5, 1.0
BLEND_S = 0.25                         # blend in/out of a window (s), at least
BLEND_SPEED = 2.0                      # m/s: a window ends with a longer blend when the ragdoll is far from the pose
STANDING_BAND = (0.7, 1.3)             # pelvis height over the support, x standing pelvis height
LAST_RUN: dict = {}                    # diagnostics of the latest apply_limb_physics call (not in the payload)


def _side_of(frames_t, items) -> np.ndarray:
    """Mode name active at each time from a (t0, t1, mode) list sorted by t0."""
    if not items:
        return np.full(len(frames_t), "", dtype=object)
    t0 = np.array([it[0] for it in items])
    k = np.clip(np.searchsorted(t0, frames_t, side="right") - 1, 0, len(items) - 1)
    return np.array([it[2] for it in items], dtype=object)[k]


def flight_list(events) -> list[tuple[float, float, bool, bool]]:
    """Free flights from the ground-truth events: (start s, end s, voluntary jump, ends feet first)."""
    evs = sorted(((float(te), name, info) for te, name, info in events), key=lambda e: e[0])
    out = []
    for te, name, info in evs:
        if name != "free_fall_start":
            continue
        ends = [(t2, i2) for t2, n2, i2 in evs if n2 == "free_fall_end" and t2 > te]
        if not ends:
            continue
        t_end, info_end = ends[0]
        jump = info.get("kind") == "jump" or info_end.get("kind") == "jump"
        out.append((te, t_end, jump, jump or info_end.get("reason") == "feet"))
    return out


def control_schedule(ctx, scene, rig, frames: np.ndarray, M: np.ndarray, flights=None) -> dict:
    """Muscle tone ctrl(t) in [0, 1] for legs, arms and head (N,) each, from ground truth:
    fall_onset/impact/arrest/free-fall events, phases, the consciousness log and the scenario's
    post-fall behaviour. Also returns the per-frame state names."""
    t = frames / ctx.fs
    n = len(t)
    dt = float(t[1] - t[0]) if n > 1 else 0.02
    phase = ctx.track.phase[frames]
    conscious = np.ones(n, bool)
    for tc, value in ctx.rec.conscious_log:
        conscious[t >= tc] = bool(value)
    behaviour = str(ctx.params.get("behaviour", ""))
    events = sorted(((float(te), name, info) for te, name, info in ctx.events), key=lambda e: e[0])
    flights = flight_list(ctx.events) if flights is None else flights
    t_end = float(t[-1]) + 1.0

    def first(names, after, before=np.inf, cond=lambda info: True):
        hits = [te for te, name, info in events if name in names and after <= te < before and cond(info)]
        return min(hits) if hits else None

    # Standing on the feet: trunk upright and the pelvis at about standing height over the
    # support (not far above it: hanging in a harness or bouncing in a net is not standing).
    pel = rig.pelvis
    height = pel[:, 1] - support_points(scene, pel[:, 0], pel[:, 2], pel[:, 1] - 0.25)
    stand_h = ctx.worker.pelvis_height
    upright = (M[:, 0, 1, 1] > 0.6) & (height > STANDING_BAND[0] * stand_h) & (height < STANDING_BAND[1] * stand_h)

    state = np.full(n, "normal", dtype=object)
    episode = np.zeros(n, bool)
    onsets = [(te, info) for te, name, info in events if name == "fall_onset"]
    starts_rec = sorted([c.t0 for c in ctx.rec.calls if c.name in ("get_up", "stand_up", "walk")])
    for k, (t_on, info) in enumerate(onsets):
        t_next = onsets[k + 1][0] if k + 1 < len(onsets) else t_end
        t_down = first(("impact",), t_on, t_next, lambda i: i.get("mode") != "feet_first")
        t_arrest = first(("arrest_start",), t_on, t_next)
        if t_arrest is not None and (t_down is None or t_arrest < t_down):
            t_down = t_arrest
        t_land = first(("free_fall_end",), t_on, t_next, lambda i: i.get("reason") == "feet")
        fall_end = min(x for x in (t_down, t_land, t_next) if x is not None)
        sel = (t >= t_on) & (t < fall_end)
        state[sel] = "collapse" if info.get("kind") == "collapse" else "falling"
        episode |= sel
        for t_ff, t_ffe, jump, feet in flights:
            if t_on <= t_ff < fall_end and not jump:
                sel = (t >= t_ff) & (t < min(t_ffe, fall_end))
                state[sel] = "air_feet" if feet else "air"
        if t_land is not None and (t_down is None or t_land < t_down):
            sel = (t >= t_land) & (t < t_land + 0.5)
            state[sel] = "landing"
            episode |= sel
        if t_down is not None:
            t_rec = min([s for s in starts_rec if s > t_down] + [t_next, t_end])
            rec_phase = np.flatnonzero((t > t_down) & (phase == 6))
            if len(rec_phase):
                t_rec = min(t_rec, float(t[rec_phase[0]]))
            sel = (t >= t_down) & (t < t_rec)
            hang = phase == 5
            # Conscious movement bouts (falls.post_fall 'struggle', or 'still_then_move' once it
            # wakes; both set conscious = True when the bouts start) after a short stunned
            # spell: tone comes back and the procedural pose moves the limbs.
            t_awake = min([tc for tc, value in ctx.rec.conscious_log if value and tc >= t_down - 1e-6] + [np.inf])
            moving = conscious & (behaviour in ("struggle", "still_then_move")) & (t >= max(t_awake, t_down + STUN_S))
            down = np.where(conscious, np.where(moving, "down_struggle", "down_conscious"), "down_unconscious")
            hung = np.where(conscious, np.where(moving, "hang_struggle", "hang_conscious"), "hang_unconscious")
            state[sel] = np.where(hang[sel], hung[sel], down[sel])
            episode |= sel & (t < t_down + 0.5)
    # Voluntary jump (no fall episode): firm at takeoff and touchdown, softer in the air.
    for t_ff, t_ffe, jump, _ in flights:
        if jump:
            sel = (t >= t_ff - 0.15) & (t < t_ffe + 0.45) & (state == "normal")
            state[sel] = np.where((t[sel] >= t_ff) & (t[sel] < t_ffe), "jump_air", "jump")
            episode |= sel
    # Voluntary lying (lying work, a nap): limbs rest on the floor unless the arms work.
    arm_mode = _side_of(t, getattr(rig, "arm_modes", []))
    rest = np.isin(phase, (4, 8)) & (state == "normal") & ~upright
    state[rest] = np.where(conscious[rest], "rest", "rest_asleep")
    # Standing on the feet always means full tone, whatever the phase label says.
    state[upright & ~episode] = "normal"

    tone = np.array([TONE[s] for s in state])                       # (n, 3)
    tone[(state == "rest") & (arm_mode == "liework"), 1] = 1.0      # working overhead while lying
    # Slew-limited ramps: tone drops within RAMP_DOWN_S and recovers within RAMP_UP_S.
    ctrl = np.empty_like(tone)
    ctrl[0] = tone[0]
    down_step, up_step = dt / RAMP_DOWN_S, dt / RAMP_UP_S
    for k in range(1, n):
        ctrl[k] = ctrl[k - 1] + np.clip(tone[k] - ctrl[k - 1], -down_step, up_step)
    return {"legs": ctrl[:, 0], "arms": ctrl[:, 1], "head": ctrl[:, 2], "state": state,
            "on_net": _lying_on_net(scene, rig.pelvis, phase)}


def _lying_on_net(scene, pelvis: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Frames (N,) where the body lies in the safety net. The procedural lying pose does not
    know the sagging net and would hang the limbs below the torso (the drawn cloth then sags
    under them, away from the torso), so the ragdoll keeps running there at any tone: with full
    tone it follows the pose but rests on the net."""
    net = scene.dyn.get("net")
    if net is None:
        return np.zeros(len(pelvis), bool)
    rel = pelvis - np.asarray(net["centre"], float)
    yaw = float(net["yaw"])
    inside = (np.abs(rel @ right(yaw)) <= net["w"] / 2) & (np.abs(rel @ fwd(yaw)) <= net["l"] / 2)
    return inside & np.isin(phase, (4, 5, 8)) & (pelvis[:, 1] < float(net["y"]) + 0.6)


def physics_windows(ctrl: dict, t: np.ndarray) -> list[tuple[int, int]]:
    """Frame ranges [k0, k1) that need simulating: wherever any tone is below FULL_TONE or the
    body lies in the safety net (ctrl['on_net']), plus margins (merged when close)."""
    n = len(t)
    dt = float(t[1] - t[0]) if n > 1 else 0.02
    active = np.minimum(np.minimum(ctrl["legs"], ctrl["arms"]), ctrl["head"]) < FULL_TONE
    active |= ctrl.get("on_net", np.zeros(n, bool))
    if not active.any():
        return []
    edges = np.flatnonzero(np.diff(np.concatenate([[0], active.astype(int), [0]])))
    runs = list(zip(edges[::2], edges[1::2], strict=True))
    before, after, gap = (int(round(x / dt)) for x in (MARGIN_BEFORE_S, MARGIN_AFTER_S, MERGE_GAP_S))
    merged: list[list[int]] = []
    for a, b in runs:
        a, b = max(a - before, 0), min(b + after, n)
        if merged and a - merged[-1][1] < gap:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= 3]


# =============================================================================================
# Limb ragdoll: world, targets and simulation
# =============================================================================================
def _limb_world(target0: np.ndarray, dims, body_mass: float, colliders: Colliders) -> tuple[World, dict]:
    """The 18-particle limb ragdoll: pinned hips, shoulders, neck; exact bones; joint limits."""
    s = dims.s
    world = World(colliders)
    world.touch_mask = SCENE_MASK
    mass = np.full(18, np.inf)
    radius = np.zeros(18)
    group = np.full(18, SCENE_MASK)
    for j in DYNAMIC:
        mass[j] = JOINT_MASS[j] * body_mass
        radius[j] = RADIUS[j - (j % 2) if j < 16 else j] * s
    group[list(ARMS)] = SCENE_MASK | ARM_MASK
    world.add_particles(target0, mass, radius, group)
    h = {"pins": world.add_pin(np.array(PINNED))}
    L, R = np.array([0, 1]), np.array([2, 3])
    head_len = float(np.linalg.norm(target0[17] - target0[16]))
    h["bones"] = np.concatenate([
        world.add_distance(L, R, dims.thigh),                               # hip-knee
        world.add_distance(R, R + 2, dims.shank),                           # knee-ankle
        world.add_distance(L + 10, L + 12, dims.upper_arm),                 # shoulder-elbow
        world.add_distance(L + 12, L + 14, dims.forearm),                   # elbow-hand
        world.add_distance([16], [17], head_len)])                          # neck-head
    h["foot"] = np.concatenate([world.add_distance([4, 5], [6, 7]), world.add_distance([4, 5], [8, 9]),
                                world.add_distance([6, 7], [8, 9])])        # ankle-heel, ankle-toe, heel-toe

    # Joint limits: knee and elbow flexion range as root-to-end distances (max flexion ~145 deg).
    def reach(l1, l2, inner_deg):
        return float(np.sqrt(l1 * l1 + l2 * l2 - 2 * l1 * l2 * np.cos(np.radians(inner_deg))))
    world.add_distance([0, 1], [4, 5], lo=reach(dims.thigh, dims.shank, 35), hi=0.9995 * dims.leg)
    world.add_distance([10, 11], [14, 15], lo=reach(dims.upper_arm, dims.forearm, 35), hi=0.9995 * dims.arm)
    # Ankle range (dorsiflexion 20 deg .. plantarflexion 50 deg from a right angle) as knee-toe.
    toe_len = float(np.linalg.norm(target0[8] - target0[4]))
    world.add_distance([2, 3], [8, 9], lo=reach(dims.shank, toe_len, 75), hi=reach(dims.shank, toe_len, 145))
    h["bend"] = world.add_bend([0, 1, 10, 11], [2, 3, 12, 13], [4, 5, 14, 15])
    h["cone_hip"] = world.add_cone([0, 1], [2, 3], np.radians(80))
    h["cone_head"] = world.add_cone([16], [17], np.radians(55))
    return world, h


def _bend_refs(J: np.ndarray, M: np.ndarray, dt: float) -> np.ndarray:
    """Bend-side reference per frame (N, 4, 3) for knees, then elbows. Knees: the pelvis's
    anterior (a hinge bends forward whatever the procedural IK did; the solver projects it
    across the hip-ankle line). Elbows: the target pose's bend direction, biased posterior
    for a nearly straight arm, kept continuous in time (a flip of the procedural elbow is
    ignored) and smoothed over ~0.1 s."""
    out = np.zeros((len(J), 4, 3))
    out[:, 0] = out[:, 1] = -M[:, 0, :, 2]
    for k, (a, m, c) in ((2, (10, 12, 14)), (3, (11, 13, 15))):
        u = unit_rows(J[:, c] - J[:, a])
        off = J[:, m] - J[:, a]
        off -= u * _dot(off, u)[:, None]
        ref = unit_rows(off + 0.03 * M[:, 3, :, 2])
        for i in range(1, len(ref)):                    # keep the side chosen so far
            if ref[i] @ ref[i - 1] < 0:
                ref[i] = ref[i - 1]
        out[:, k] = gaussian_filter1d(ref, 0.1 / dt, axis=0, mode="nearest")
    return out


def _shank_frame(knee: np.ndarray, ankle: np.ndarray, anterior: np.ndarray) -> np.ndarray:
    """Columns: shank axis (knee -> ankle), the anterior direction across it, their normal."""
    axis = unit_rows((ankle - knee)[None])[0]
    across = anterior - axis * (anterior @ axis)
    if across @ across < 1e-8:                       # shank along the anterior: any perpendicular
        across = np.cross(axis, [1.0, 0.0, 0.0])
    across /= np.linalg.norm(across)
    return np.column_stack([axis, across, np.cross(axis, across)])


def _carried_feet(drive_k: np.ndarray, x: np.ndarray, anterior: np.ndarray) -> np.ndarray:
    """Goals for ankle, heel and toe of both feet (6, 3, in FOOT_JOINTS order): the posed
    foot, as seen from the posed shank, carried on the simulated shank, then shifted so the
    three particles keep their mass centre (the pull only turns the foot; no net force)."""
    goals = np.zeros((6, 3))
    for side in (0, 1):
        knee, ankle = 2 + side, 4 + side
        joints = [ankle, 6 + side, 8 + side]
        posed = _shank_frame(drive_k[knee], drive_k[ankle], anterior)
        now = _shank_frame(x[knee], x[ankle], anterior)
        want = x[ankle] + (drive_k[joints] - drive_k[ankle]) @ posed @ now.T
        m = JOINT_MASS[joints] / JOINT_MASS[joints].sum()
        goals[3 * side:3 * side + 3] = want + (m @ x[joints] - m @ want)
    return goals


def _trunk_capsule(spine: np.ndarray, M: np.ndarray, dims) -> tuple[np.ndarray, np.ndarray]:
    """Moving capsule inside the drawn torso (body.js centres it ~0.08 m anterior of the spine
    line), so limp arms rest on the trunk instead of passing through it."""
    ant0, ant3 = -M[:, 0, :, 2], -M[:, 3, :, 2]
    a = spine[:, 0] + M[:, 0, :, 1] * (-0.04 * dims.H) + ant0 * 0.07
    b = spine[:, 3] - M[:, 3, :, 1] * 0.03 + ant3 * 0.06
    return a, b


def _flight_frames(flights, t: np.ndarray) -> list[tuple[int, int, bool]]:
    """Flights as frame indices: (first frame at/after takeoff, first frame at/after touchdown,
    feet first)."""
    out = []
    for t0, t1, _, feet in flights:
        k0 = int(np.searchsorted(t, t0 - 1e-9))
        k1 = min(int(np.searchsorted(t, t1 - 1e-9)), len(t) - 1)
        if k1 - k0 >= 2:
            out.append((k0, k1, feet))
    return out


def plan_flight_legs(target: np.ndarray, pelvis: np.ndarray, R0: np.ndarray, flights_k) -> np.ndarray:
    """Drive targets with planned legs in feet-first flights: knees, ankles, heels and toes
    move with the pelvis from their takeoff pose to their touchdown pose (pelvis frame,
    smoothstep), instead of the procedural air pose, so the feet leave and meet the support
    with the trunk's own timing."""
    out = target.copy()
    legs = np.array(LEGS)
    for k0, k1, feet in flights_k:
        if not feet or k0 < 1:
            continue
        ka, kb = k0 - 1, k1
        rel_a = (target[ka, legs] - pelvis[ka]) @ R0[ka]           # world -> pelvis frame
        rel_b = (target[kb, legs] - pelvis[kb]) @ R0[kb]
        ks = np.arange(ka + 1, kb)
        u = smoothstep((ks - ka) / (kb - ka))[:, None, None]
        rel = rel_a[None] + (rel_b - rel_a)[None] * u
        out[ks[:, None], legs[None]] = pelvis[ks][:, None] + np.einsum("kij,klj->kli", R0[ks], rel)
    return out


GROUPS = (("legs", LEGS), ("arms", ARMS), ("head", HEAD))


def _damping(ctrl: np.ndarray) -> np.ndarray:
    """Drive damping ratio for a tone: LIMP_DAMPING at no tone, critical from DAMPED_TONE up."""
    return LIMP_DAMPING + (1.0 - LIMP_DAMPING) * smoothstep(np.clip(ctrl, 0.0, 1.0) / DAMPED_TONE)


def _rise_starts(ctrl_g: np.ndarray, k0: int, k1: int) -> np.ndarray:
    """Frames in (k0, k1) where a group's tone starts rising from below RISE_FROM."""
    ks = np.arange(max(k0 + 1, 2), k1)
    rising = ctrl_g[ks] > ctrl_g[ks - 1] + 1e-9
    was_flat = ctrl_g[ks - 1] <= ctrl_g[ks - 2] + 1e-9
    return ks[rising & was_flat & (ctrl_g[ks - 1] < RISE_FROM)]


def _cross_fade(drive: np.ndarray, x: np.ndarray, idx, k: int, k1: int, carry, rise_k: int) -> None:
    """Drive targets of joints idx from frame k on (in place): from their simulated positions x
    (end of frame k - 1), carried with the trunk frame (origin, rotation), to the procedural
    pose over rise_k frames (smoothstep in time)."""
    origin, R = carry
    idx = np.asarray(idx)
    rel = (x[idx] - origin[k - 1]) @ R[k - 1]                     # trunk frame at the start
    kk = np.arange(k, min(k + rise_k, k1))
    carried = origin[kk][:, None] + np.einsum("kij,lj->kli", R[kk], rel)
    u = smoothstep((kk - k + 1) / rise_k)[:, None, None]
    rows = kk[:, None]
    drive[rows, idx] = carried + (drive[rows, idx] - carried) * u
    drive[k - 1, idx] = x[idx]          # the coming step's target starts where the limb is


def _simulate_window(k0: int, k1: int, target: np.ndarray, drive: np.ndarray, ctrl: dict,
                     aux: dict) -> tuple[np.ndarray, dict]:
    """Run the ragdoll over frames [k0, k1). `target` is the procedural pose (pins, limits),
    `drive` the pose the limbs are pulled toward. Returns joint positions (k1 - k0, 18, 3)."""
    dt = aux["dt"]
    dims = aux["dims"]
    seg = target[k0:k1]
    drive = drive.copy()                    # tone rises cross-fade it (_cross_fade)
    lo, hi = seg.reshape(-1, 3).min(axis=0) - 1.2, seg.reshape(-1, 3).max(axis=0) + 1.2
    colliders = scene_colliders(aux["scene"], lo, hi)
    colliders.max_push = PROP_PUSH_M
    trunk = colliders.add_capsule(aux["trunk_a"][k0], aux["trunk_b"][k0], 0.11 * dims.s * np.sqrt(dims.w), mu=0.3,
                                  mask=ARM_MASK)
    if colliders.sheets:                    # the net starts with its current dent
        colliders.set_dent(0, *aux["net"][k0], sag=NET_SAG_M)
        colliders.end_step()
    world, h = _limb_world(target[k0], dims, aux["mass"], colliders)
    ground_y = float(aux["scene"].ground_y)
    vel0 = (target[k0] - target[k0 - 1]) / dt if k0 > 0 else np.zeros((18, 3))
    world.set_state(np.arange(18), target[k0], vel0)
    f_limp = GRAVITY * world.dt ** 2 / LIMP_SAG_M
    frac = np.zeros((k1 - k0, 18))
    zeta = np.ones((k1 - k0, 18))
    cap = np.full((k1 - k0, 18), np.inf)
    rise_k = max(int(round(RISE_S / dt)), 1)
    rises: dict[int, list] = {}
    for name, idx in GROUPS:
        tone = np.clip(ctrl[name][k0:k1], 0.0, 1.0)
        frac[:, list(idx)] = (f_limp ** (1.0 - tone))[:, None]
        zeta[:, list(idx)] = _damping(tone)[:, None]
        cap[np.ix_(tone >= RISE_FROM, list(idx))] = DRIVE_MAX_SPEED
        for k in _rise_starts(ctrl[name], k0, k1):
            rises.setdefault(int(k), []).append(name)
            cap[k - k0:k - k0 + rise_k, list(idx)] = DRIVE_MAX_SPEED
    group_idx = dict(GROUPS)
    dyn, pins = np.array(DYNAMIC), np.array(PINNED)
    foot_frac = f_limp ** (1.0 - FOOT_TONE)
    # Limb groups at full tone follow their pose; they keep the ragdoll awake only while
    # something holds them off it (see the sleep test below).
    follow = frac[:, dyn] >= f_limp ** (1.0 - FULL_TONE)
    m_dyn = float(JOINT_MASS[dyn].sum())
    flight_at = aux["flight_at"]
    out = np.empty_like(seg)
    out[0] = target[k0]
    asleep, quiet = False, 0
    offsets: dict[int, np.ndarray] = {}
    released: dict[int, int] = {}           # flight -> frame a limb first touched something
    feet_goal = None
    stats = {"frames": k1 - k0, "asleep": 0, "flights": 0, "released": 0, "rises": 0}
    quiet_need = int(round(0.4 / dt))
    fade_in, fade_out = CENTROID_FADE_IN_S / dt, CENTROID_FADE_OUT_S / dt
    for k in range(k0 + 1, k1):
        i = k - k0
        flight = int(flight_at[k])
        for name in rises.get(k, ()):
            _cross_fade(drive, world.x, group_idx[name], k, k1, aux["carry"][name], rise_k)
            stats["rises"] += 1
        free = ~follow[i]
        driven = free & (frac[i, dyn] > 1e-3)
        if asleep:
            wake = (flight >= 0 or float(np.abs(target[k, pins] - sleep_pins).max()) > 0.01
                    or bool(np.any(follow[i] != sleep_follow))
                    or float(np.abs(frac[i, dyn][free] - sleep_frac[free]).max(initial=0.0)) > 1e-9
                    or float(np.abs(drive[k, dyn][driven] - sleep_drive[driven]).max(initial=0.0)) > 0.01)
            if not wake:
                out[i] = world.x
                out[i, pins] = target[k, pins]
                out[i, dyn[~free]] = drive[k, dyn[~free]]
                stats["asleep"] += 1
                continue
            asleep, quiet = False, 0
            world.set_state(dyn[free], world.x[dyn[free]])       # wake at rest
            posed = dyn[~free]                                   # full-tone groups resume on their pose
            world.set_state(posed, drive[k - 1, posed], (drive[k - 1, posed] - drive[k - 2, posed]) / dt)
            feet_goal = None
        world.set_pin_targets(h["pins"], target[k, pins], start=target[k - 1, pins])
        # A hip, shoulder or neck below the floor (the simulated trunk in it, ground truth):
        # its limb may follow it down rather than be pinned against the floor by its bone.
        world.ground_drop = np.maximum(ground_y - target[k, LIMB_ROOT, 1], 0.0)
        world.set_drive(dyn, drive[k, dyn], frac[i, dyn], start=drive[k - 1, dyn], damping=zeta[i, dyn],
                        max_speed=cap[i, dyn])
        if frac[i, 6] < foot_frac:                     # limp legs: passive ankle keeps the foot
            feet_now = _carried_feet(drive[k], world.x, -aux["M0"][k, :, 2])
            world.set_drive(FOOT_JOINTS, feet_now, foot_frac, start=feet_now if feet_goal is None else feet_goal)
            feet_goal = feet_now                       # next step's start: the goal moves with the shank
        else:
            feet_goal = None
        heel, toe = target[k, 6:8] - target[k, 4:6], target[k, 8:10] - target[k, 4:6]
        sole = target[k, 8:10] - target[k, 6:8]
        world.set_distance(h["foot"], np.concatenate([_norm(heel), _norm(toe), _norm(sole)]))
        world.set_bend_reference(h["bend"], aux["bend"][k])
        world.set_cone_axis(h["cone_hip"], aux["hip_axis"][k])
        world.set_cone_axis(h["cone_head"], aux["head_axis"][k][None])
        colliders.set_capsule(trunk, aux["trunk_a"][k], aux["trunk_b"][k])
        if colliders.sheets:
            colliders.set_dent(0, *aux["net"][k], sag=NET_SAG_M)
        # Free flight: the limbs keep the drawn whole-body COM on a ballistic path: the
        # simulator's COM plus the offset at takeoff plus the pose's own steady drift from it
        # (aux['com_path']). The hold fades in after takeoff and out once a limb touches
        # something after takeoff (a soft hold: no kick at either end).
        hold = 0.0
        if flight >= 0:
            kf0 = aux["flights_k"][flight][0]
            hold = smoothstep((k - kf0) / fade_in)
            if flight in released:
                hold *= 1.0 - smoothstep((k - released[flight]) / fade_out)
        if hold > 0.0:
            path = aux["com_path"][flight]
            if flight not in offsets:
                offsets[flight] = JOINT_MASS[dyn] @ world.x[dyn] - path[0]
                stats["flights"] += 1

            def goal(kk):
                return (path[min(kk - kf0, len(path) - 1)] + offsets[flight]) / m_dyn
            world.set_centroid(dyn, JOINT_MASS[dyn], goal(k), start=goal(k - 1), share=CENTROID_SHARE[dyn],
                               stiffness=CENTROID_STIFFNESS * hold, max_step=CENTROID_MAX_STEP_M * hold)
        else:
            world.clear_centroid()
        before = world.x[dyn].copy()
        world.step(dt)
        if (flight >= 0 and flight not in released and world.touched[dyn].any()
                and (k - aux["flights_k"][flight][0]) * dt > TAKEOFF_S):
            released[flight] = k
            stats["released"] += 1
            LAST_RUN.setdefault("released_at", []).append((flight, k, [int(j) for j in dyn[world.touched[dyn]]]))
        out[i] = world.x
        # Sleep when the limbs that are not at full tone stayed still for 0.4 s while the trunk,
        # their drive targets and the tone stayed put: they keep their positions until one of
        # those changes (cheap long holds); full-tone groups keep following their pose (so they
        # must be on it: one held back by an obstacle keeps the ragdoll awake).
        still = (flight < 0 and float(np.abs(world.x[dyn][free] - before[free]).max(initial=0.0)) < 0.002 * dt / 0.02
                 and float(np.abs(world.x[dyn][~free] - drive[k, dyn][~free]).max(initial=0.0)) < 0.01
                 and float(np.abs(target[k, pins] - target[k - 1, pins]).max()) < 0.002
                 and bool(np.all(follow[i] == follow[i - 1]))
                 and float(np.abs(frac[i, dyn][free] - frac[i - 1, dyn][free]).max(initial=0.0)) < 1e-9
                 and float(np.abs(drive[k, dyn][driven] - drive[k - 1, dyn][driven]).max(initial=0.0)) < 0.002)
        quiet = quiet + 1 if still else 0
        if quiet >= quiet_need:
            asleep = True
            sleep_pins, sleep_follow = target[k, pins].copy(), follow[i].copy()
            sleep_frac, sleep_drive = frac[i, dyn].copy(), drive[k, dyn].copy()
    return out, stats


# =============================================================================================
# Limb ragdoll: final pass and entry point
# =============================================================================================
def _place(parent: np.ndarray, child: np.ndarray, length, floor: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Child at exactly `length` from parent along the current direction, swung up about the
    parent (length kept) if it would end below `floor`."""
    d = child - parent
    dist = _norm(d)
    u = np.where((dist > 1e-9)[:, None], d / np.maximum(dist, 1e-12)[:, None], fallback)
    length = np.broadcast_to(np.asarray(length, float), dist.shape)
    need = (floor - parent[:, 1]) / np.maximum(length, 1e-9)
    low = u[:, 1] < need
    if low.any():
        uy = np.clip(need[low], -1.0, 1.0)
        horiz = u[low][:, [0, 2]]
        hn = np.sqrt(np.sum(horiz * horiz, axis=1))
        flat_dir = np.where((hn > 1e-9)[:, None], horiz / np.maximum(hn, 1e-12)[:, None], np.array([[0.0, 1.0]]))
        scale = np.sqrt(np.maximum(1.0 - uy * uy, 0.0))
        u[low] = np.column_stack([flat_dir[:, 0] * scale, uy, flat_dir[:, 1] * scale])
    return parent + u * length[:, None]


SUPPORT_REACH_M = 0.05     # final pass: only surfaces this close above a joint (or its target) hold it up


def finalize_limbs(J: np.ndarray, target: np.ndarray, dims, scene, weight: np.ndarray | None = None) -> np.ndarray:
    """Exact bone lengths from each root outward, and every limb joint centre at least CLEAR
    above the support under it. Only surfaces at most SUPPORT_REACH_M above the joint or its
    procedural target (the lower of the two) count, so a joint beside a step block, rail or
    platform is never put on top of it (as rig._sole_lift). Where the limb's root (hip,
    shoulder, neck) is itself below that support (the simulated trunk in the floor), the
    floor is lowered by that depth, so the limb is not swung up out of the floor around it.
    `weight` (N,), the physics blend
    weight: where it is below 1 the joint may also sink as deep as the procedural pose's joint
    does, scaled by 1 - weight, so the pass leaves the procedural pose itself unchanged and the
    output is continuous where a window blends back into it."""
    J = J.copy()
    reach = np.minimum(J[..., 1], target[..., 1]) + SUPPORT_REACH_M        # from the input pose
    slack = np.zeros(len(J)) if weight is None else 1.0 - np.clip(weight, 0.0, 1.0)

    def settle(j, parent, length, fallback):
        """Place joint j at `length` from its parent, then again if the swing moved it over
        a higher surface (the support is read where the joint ends up; 3 passes)."""
        clear = CLEAR[j - (j % 2) if j < 16 else j]
        root = J[:, LIMB_ROOT[j], 1]
        for _ in range(3):
            support = support_points(scene, J[:, j, 0], J[:, j, 2], reach[:, j])
            floor = support + clear - np.maximum(support - root, 0.0)     # a root in the floor: follow it
            floor -= slack * np.maximum(floor - target[:, j, 1], 0.0)
            J[:, j] = _place(J[:, parent], J[:, j], length, floor, fallback)

    for side in (0, 1):
        hip, knee, ankle, heel, toe = side, 2 + side, 4 + side, 6 + side, 8 + side
        settle(knee, hip, dims.thigh, unit_rows(target[:, knee] - target[:, hip]))
        settle(ankle, knee, dims.shank, unit_rows(target[:, ankle] - target[:, knee]))
        for j in (heel, toe):
            settle(j, ankle, _norm(target[:, j] - target[:, ankle]), unit_rows(target[:, j] - target[:, ankle]))
        sh, el, hand = 10 + side, 12 + side, 14 + side
        settle(el, sh, dims.upper_arm, unit_rows(target[:, el] - target[:, sh]))
        settle(hand, el, dims.forearm, unit_rows(target[:, hand] - target[:, el]))
    settle(17, 16, _norm(target[:, 17] - target[:, 16]), unit_rows(target[:, 17] - target[:, 16]))
    return J


def _turn_head(head_q: np.ndarray, before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Rotate each head orientation by the shortest arc taking the neck->head direction
    `before` to `after` (xyzw). Continuous unless the two are nearly opposite, which the head
    cone (55 deg about the rest direction) and the rate limit (_rate_limited) rule out."""
    a, b = unit_rows(before), unit_rows(after)
    axis = np.cross(a, b)
    w = 1.0 + _dot(a, b)
    opposite = w < 1e-6
    axis[opposite] = np.cross(a[opposite], np.array([1.0, 0.0, 0.0]))
    w[opposite] = 0.0
    q = np.column_stack([axis, w])
    q /= np.maximum(_norm(q), 1e-12)[:, None]
    return (Rotation.from_quat(q) * Rotation.from_quat(head_q)).as_quat()


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Quaternion product a * b (xyzw)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def _qconj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vector v rotated by the unit quaternion q (xyzw)."""
    u, w = q[:3], q[3]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _limit_head(rig, out: np.ndarray, target: np.ndarray, M: np.ndarray, windows, scene, catch_up: int) -> None:
    """Head orientation from the ragdoll's neck->head direction (the shortest arc from the
    procedural one), turning at most HEAD_STEP_DEG per frame relative to the upper thorax
    inside the physics windows and up to `catch_up` frames after them (slerp toward the
    ragdoll's). A limited head that would end below the support is swung up about the neck in
    the same frame, and the limiter goes on from there, so no lag builds up to be released at
    once. The head joint (out[:, 17], in place) follows the limited orientation."""
    head_q = _turn_head(rig.head_q, target[:, 17] - target[:, 16], out[:, 17] - out[:, 16])
    upper = Rotation.from_matrix(M[:, 3]).as_quat()
    dims = rig.dims
    head_rest = np.array([0.0, 0.1 * dims.H, -0.022 * dims.s])
    length = float(np.linalg.norm(head_rest))
    half = np.radians(HEAD_STEP_DEG) / 2
    cos_half = np.cos(half)
    # support under the ragdoll's head (the limited head stays near it)
    reach = np.minimum(out[:, 17, 1], target[:, 17, 1]) + SUPPORT_REACH_M
    floor = support_points(scene, out[:, 17, 0], out[:, 17, 2], reach) + CLEAR[17]
    for k0, k1 in windows:
        a, b = max(k0 - 1, 0), min(k1 + catch_up, len(head_q))
        cur = _qmul(_qconj(upper[a]), head_q[a])
        for k in range(a + 1, b):
            want = _qmul(_qconj(upper[k]), head_q[k])
            c = float(cur @ want)
            if c < 0:
                want, c = -want, -c
            if c >= cos_half:
                cur = want
                continue
            theta = float(np.arccos(min(c, 1.0)))
            cur = (np.sin(theta - half) * cur + np.sin(half) * want) / np.sin(theta)
            q = _qmul(upper[k], cur / np.linalg.norm(cur))
            d = _qrot(q, head_rest) / length
            need = (floor[k] - out[k, 16, 1]) / length
            axis = np.cross(d, UP)
            size = float(np.linalg.norm(axis))
            if d[1] < need <= 1.0 and size > 1e-6:
                # As rig._head_above_support: a neck itself below the support (the simulated
                # trunk in the floor) cannot be fixed by the head, so the lift fades out over
                # 0-10 cm of neck depth; and it turns the head by at most another limit step.
                depth = floor[k] - CLEAR[17] - out[k, 16, 1]
                lift = (np.arcsin(need) - np.arcsin(np.clip(d[1], -1.0, 1.0))) * smoothstep(1.0 - depth / 0.1)
                lift = min(lift, 2 * half) / 2
                q = _qmul(np.concatenate([axis / size * np.sin(lift), [np.cos(lift)]]), q)
            cur = _qmul(_qconj(upper[k]), q)
            head_q[k] = q
            out[k, 17] = out[k, 16] + _qrot(q, head_rest)
    rig.head_q = head_q


def apply_limb_physics(ctx, scene, frames: np.ndarray, rig, body_idx: np.ndarray) -> dict:
    """Ragdoll post-pass on the rig (in place): simulate the limbs and head in the windows that
    need it, blend in/out, then (inside the windows) exact bones, a support clamp and a head
    turn-rate limit; outside them the procedural pose is kept as is. Updates rig.joints,
    rig.hand_l, rig.hand_r, rig.head and rig.head_q. Returns a summary for meta.physics."""
    t_start, cpu_start = time.perf_counter(), time.process_time()
    LAST_RUN.clear()
    n = len(frames)
    info = {"engine": ENGINE, "version": VERSION, "substep_hz": SUBSTEP_HZ, "enabled": True, "windows": []}
    if n < 3 or rig.joints.shape[1:] != (18, 3):
        info["enabled"] = False
        return info
    t = frames / ctx.fs
    dt = float(t[1] - t[0])
    dims = rig.dims
    M = np.stack([ctx.kin.seg_rot[i][body_idx].as_matrix() for i in range(4)], axis=1)   # (N, 4, 3, 3)
    target = rig.joints.copy()
    flights = flight_list(ctx.events)
    ctrl = control_schedule(ctx, scene, rig, frames, M, flights)
    windows = physics_windows(ctrl, t)
    out = target.copy()
    stats_all = {"sleep_frames": 0, "flights": 0, "flights_released": 0, "tone_rises": 0}
    final_move = np.full(n, np.nan)          # diagnostics: how far the final pass moved the ragdoll
    blend = np.zeros(n)
    if windows:
        flights_k = _flight_frames(flights, t)
        drive = plan_flight_legs(target, rig.pelvis, M[:, 0], flights_k)
        track = ctx.track
        R_root = root_rotation(track.heading[body_idx], track.pitch[body_idx], track.roll[body_idx])
        r_p = np.array(SIM_COM_R_P) * np.array([1.0, ctx.H, 1.0])
        flight_at = np.full(n, -1)
        for f, (kf0, kf1, _) in enumerate(flights_k):
            flight_at[kf0 + 1:kf1 + 1] = f          # steps that end inside the flight
        trunk_a, trunk_b = _trunk_capsule(rig.spine, M, dims)
        head_rest = np.array([0.0, 0.1 * dims.H, -0.022 * dims.s])
        pins = list(PINNED)
        upper = (rig.spine[:, 3], M[:, 3])
        aux = {"dt": dt, "dims": dims, "scene": scene, "mass": float(ctx.worker.mass_kg), "trunk_a": trunk_a,
               "trunk_b": trunk_b, "bend": _bend_refs(target, M, dt),
               "hip_axis": np.repeat(unit_rows(-M[:, 0, :, 1] - 0.84 * M[:, 0, :, 2])[:, None], 2, axis=1),
               "head_axis": unit_rows(np.einsum("nij,j->ni", M[:, 3], head_rest)),
               "c_sim": rig.pelvis - R_root.apply(r_p), "flight_at": flight_at, "flights_k": flights_k, "M0": M[:, 0],
               "known": TRUNK_M * torso_centroid(rig.spine, M, dims.H, dims.s, dims.w)
               + np.einsum("j,njk->nk", JOINT_MASS[pins], target[:, pins]),
               "carry": {"legs": (rig.pelvis, M[:, 0]), "arms": upper, "head": upper}}
        # Per flight, the limbs' mass moment the COM hold aims at (plus a constant set at
        # takeoff): the simulator's COM plus the posed body's steady drift from it (least-squares
        # slope, kept so the limbs only remove the non-ballistic curvature) minus the trunk's part.
        aux["com_path"] = []
        for kf0, kf1, _ in flights_k:
            ks = np.arange(kf0, kf1 + 1)
            gap = body_com(drive[ks], rig.spine[ks], M[ks], dims.H, dims.s, dims.w) - aux["c_sim"][ks]
            drift = np.polyfit((ks - kf0) * dt, gap, 1)[0]
            path = aux["c_sim"][ks] + drift * ((ks - kf0) * dt)[:, None] - aux["known"][ks]
            aux["com_path"].append(path)
        if "net" in scene.dyn:
            # The limbs lie on a dent whose bottom is the drawn torso's lowest point, centred
            # under the torso, so they rest on the cloth around it (net_motion lets the body push
            # the cloth down) instead of sinking below the torso.
            centre = torso_centroid(rig.spine, M, dims.H, dims.s, dims.w)
            bottom = torso_bottom(rig.spine, M, dims.H, dims.s, dims.w)
            dent = np.clip(float(scene.dyn["net"]["y"]) - NET_SAG_M - bottom, 0.0, 1.5)
            aux["net"] = np.column_stack([centre[:, 0], centre[:, 2], dent])
        blend_k = max(int(round(BLEND_S / dt)), 1)
        for k0, k1 in windows:
            phys, stats = _simulate_window(k0, k1, target, drive, ctrl, aux)
            stats_all["sleep_frames"] += stats["asleep"]
            stats_all["flights"] += stats["flights"]
            stats_all["flights_released"] += stats["released"]
            stats_all["tone_rises"] += stats["rises"]
            w = np.ones(k1 - k0)
            ramp = smoothstep(np.arange(1, blend_k + 1) / blend_k)
            if k0 > 0:
                w[:blend_k] = np.minimum(w[:blend_k], ramp[: k1 - k0])
            if k1 < n:
                # Back to the procedural pose: the smoothstep's peak speed (1.5 x gap / time)
                # stays under BLEND_SPEED where the ragdoll ended up far from it (held by a stair).
                tail = slice(max(k1 - k0 - 2 * blend_k, 0), k1 - k0)
                gap = float(_norm(phys[tail] - target[k0:k1][tail]).max())
                end_k = int(np.clip(round(max(BLEND_S, 1.5 * gap / BLEND_SPEED) / dt), blend_k, k1 - k0))
                down = smoothstep(np.arange(end_k, 0, -1) / end_k)
                w[-end_k:] = np.minimum(w[-end_k:], down)
            blended = target[k0:k1] + (phys - target[k0:k1]) * w[:, None, None]
            out[k0:k1] = finalize_limbs(blended, target[k0:k1], dims, scene, w)
            final_move[k0:k1] = _norm(out[k0:k1] - blended).max(axis=1)
            blend[k0:k1] = w
            info["windows"].append([round(float(t[k0]), 3), round(float(t[k1 - 1]), 3)])
        _limit_head(rig, out, target, M, windows, scene, int(round(1.0 / dt)))
    rig.joints = out
    rig.hand_l, rig.hand_r, rig.head = out[:, 14].copy(), out[:, 15].copy(), out[:, 17].copy()
    info["sim_frames"] = int(sum(b - a for a, b in windows))
    info.update(stats_all)
    LAST_RUN["seconds"] = time.perf_counter() - t_start     # timing stays out of the payload (determinism)
    LAST_RUN["cpu"] = time.process_time() - cpu_start
    LAST_RUN["ctrl"] = ctrl
    LAST_RUN["target"] = target
    LAST_RUN["final_move"], LAST_RUN["blend"] = final_move, blend
    return info


# =============================================================================================
# Render-only rest lift
# =============================================================================================
LIFT_CAP_M, LIFT_EASE_S, LIFT_MIN_REST_S = 0.15, 1.0, 2.2


def torso_clearance(scene, P: np.ndarray, M: np.ndarray, H: float, s: float, w: float) -> np.ndarray:
    """Lowest drawn torso point minus the support under it, per frame (negative = in the floor)."""
    worst = np.full(len(P), np.inf)
    for line, c, a, bd, xr, zr in _torso_rings(P, M, H, s, w):
        low = _ring_low(c, a, bd, xr, zr)
        support = support_points(scene, c[:, 0], c[:, 2], line[:, 1] + 0.3)
        worst = np.minimum(worst, low - support)
    return worst


def rest_lift(ctx, scene, frames: np.ndarray, body_idx: np.ndarray, shift: np.ndarray) -> tuple[np.ndarray, list]:
    """Render-only vertical offset (N,) for the whole body while it lies at rest with the drawn
    torso below the support (ground truth puts the spine a few cm into the floor at times).
    Capped at LIFT_CAP_M, eased in and out over LIFT_EASE_S inside rest spans of at least
    LIFT_MIN_REST_S, smooth in between. Returns the offset and display notes."""
    n = len(frames)
    lift = np.zeros(n)
    if n < 3:
        return lift, []
    t = frames / ctx.fs
    dt = float(t[1] - t[0])
    kin = ctx.kin
    M = np.stack([kin.seg_rot[i][body_idx].as_matrix() for i in range(4)], axis=1)
    P = kin.anchors[body_idx] - np.einsum("nsij,j->nsi", M, np.array([0.0, 0.0, ctx.worker.sensor_offset_m]))
    P = P + shift[:, None, :]
    H = ctx.H
    s = H / 1.75
    w = float(np.clip(np.sqrt(ctx.worker.mass_kg / H ** 2 / 24.0), 0.88, 1.25))     # rig.Dims.w
    need = np.clip(-torso_clearance(scene, P, M, H, s, w), 0.0, LIFT_CAP_M)
    # At rest: lying phase, pelvis and upper trunk slow (0.2 s box-filtered speed).
    k = max(int(round(0.2 / dt)), 1)
    speed = np.max([np.convolve(_norm(np.gradient(P[:, j], dt, axis=0)), np.ones(k) / k, mode="same")
                    for j in (0, 3)], axis=0)
    phase = ctx.track.phase[frames]
    rest = np.isin(phase, (4, 8)) & (speed < 0.2)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], rest.astype(int), [0]])))
    ease = int(round(LIFT_EASE_S / dt))
    for a, b in zip(edges[::2], edges[1::2], strict=True):
        if (b - a) * dt < LIFT_MIN_REST_S or need[a:b].max() < 0.005:
            continue
        span = need[a:b]
        level = maximum_filter1d(span, size=int(round(1.0 / dt)), mode="nearest")
        level = gaussian_filter1d(level, 0.4 / dt, mode="nearest")
        level = np.maximum(level, span)          # never leave the torso in the floor inside the span
        level = gaussian_filter1d(level, 0.2 / dt, mode="nearest")
        u = np.arange(b - a)
        # ease in after the body comes to rest and out before it moves again (not at the ends
        # of the recording, where nothing moves next)
        ease_in = smoothstep((u + 1) / ease) if a > 0 else np.ones(b - a)
        ease_out = smoothstep((b - a - u) / ease) if b < n else np.ones(b - a)
        envelope = np.minimum(ease_in, ease_out)
        lift[a:b] = np.minimum(level * envelope, LIFT_CAP_M)
    notes = []
    if lift.max() > 0.005:
        notes.append(f"While lying at rest the simulated spine sits up to {lift.max() * 100:.0f} cm into the support "
                     "(ground truth). The drawn body and strip are lifted by that much (eased over 1 s, render only); "
                     "IMU signals are unchanged.")
    return lift, notes
