# File: contact.py
# Purpose: Whole-body contact and balance pass of the 3D viewer, between the rig (rig.py, procedural
#          or mocap limbs) and the ragdoll (physics.py): feet locked on their plants, hands locked on
#          the handholds of held objects that move rigidly with them, no carried load through the
#          legs, static balance and joint limits. Render only.
# Contributors: Unassigned
"""Contact pass (off unless THUNDERSIM_CONTACT=1, ?contact=1 or run_payload(contact=True)).

The simulated spine, pelvis, signals and events never change. Only the free degrees of freedom
move: foot placement and roll, the swing path, arm pose and where a held object sits in the
hands. The pass works on active frames (contact_model.ACTIVE_PHASES, outside the ragdoll's
windows); corrections fade in and out over EDGE_S at their edges. Gates: contact_gates.py.

Legs (per foot, per planted stance of the rig):
  - lock: the foot keeps the sole point it landed on, with the rig's own roll (and a mocap clip's
    heel strike). Where the rig would drag it (its 'no step' items, emergency toe slides) or the
    hip pulls it out of reach even rolled over the ball, the foot lifts instead: the swing to its
    next landing starts earlier, or a short step lands under the hip and locks there. The rig's
    following swing starts from the locked pose (redrawn when it is far off);
  - hip extension: a planted trailing leg rolls further onto the ball (heel up, knee bending), and
    a foot just after toe-off lifts a little, so the thigh stays within HIP_EXT_LIMIT_DEG;
  - before each lift and put-down the feet step to where the simulator's lift geometry puts the
    ankles (motions.lift_root), so the drawn legs, the load and the balance match the simulated
    pelvis; a quasi-static foot left on its ball or heel by a staggered stop takes a settle step;
  - mode changes (standing <-> kneeling, onto a ladder): feet step in an arc instead of sliding,
    one foot after the other;
  - kneeling: the feet lie plantarflexed with the toes on the floor (the rig left them 5 cm up);
  - every swing the pass draws (early lifts, short steps, redrawn swings) keeps the foot's lowest
    point SWING_CLEARANCE_M above the floor in mid-swing, so a foot rolled far onto the ball does not
    drag its toe forward along the floor.

Arms and held objects (handholds in contact_model):
  - sack: rests where the simulated grip point is at lift_grasp, is held from lift_off to
    put_down_touch and rests where it is set down. The hands lock on its handholds from the end
    of the reach to the release, at the lift's simulated grip height until the worker is upright,
    then one hand after the other slides to the put-down's grip height (contact_model.Sack.grips).
    While held, the sack hangs from the shoulders on the rise (on the near-straight arms of the
    simulated posture), then follows the rig's carry hand path. Where it would touch the drawn
    thighs, shanks or torso, the arms swing forward about the shoulders just enough, at their
    current length (no curl). A carry that picks up the sack the previous carry set down moves the
    same sack.
  - ladder rails: hand over hand. Each hand grips its rail at a fixed height and re-grips higher
    (lower going down) before the shoulder leaves its comfort band, one hand at a time; a hand
    with no rail in reach (standing high on a stepladder) hangs free until one is.
  - stepladder: the right hand reaches the upper rail at mid length (lower along the rail when
    mid length is out of reach) before the pick-up; the stepladder then turns and folds about
    that handhold, which stays in the hand (scene.stepladder_motion's poses and timing).
  - hammer strokes: shortened where the simulated strike interval would make the hand faster than
    HAND_SPEED_MAX. Hand and foot targets are speed-limited just under the pop limits.
Handholds out of the arm's reach are reported (meta.contact.unreachable), never faked. Static
balance is measured, not forced (contact_gates.py): where the simulated trunk puts the centre of
mass outside every reachable support (kneeling floor work), the gate says so.
"""

from __future__ import annotations

import os

import numpy as np

from .contact_model import (ACTIVE_PHASES, SACK_HALF, base_handhold, box_signed_distance, handhold_offset, leg_spheres,
                            rail_point, sack_axes, sack_handholds, sack_plan, stepladder_grip_local, torso_spheres)
from .geom import UP, flat, fwd, right, smoothstep, unit
from .mocap import CLEARANCE_EASE, SWING_CLEARANCE_M
from .rig import MIN_JOINT_ANGLE, PLANTED_ROLL_STEP, SIDES, speed_limited, swing_profile, two_bone, wrap_angle
from .scene import body_scale

VERSION = "1"
ENV = "THUNDERSIM_CONTACT"
DEFAULT_ON = False             # on by default only once every gate passes (contact_gates.py)

# --- tuning ([design] unless a source is named; lengths m, times s, angles deg) --------------
EDGE_S = 0.3                   # corrections fade in / out over this at the edges of active frames
LOCK_DRIFT_M = 0.001           # the rig moved a plant this far: the stance ends there
REACH_EXCESS_M = 0.002         # hip this far beyond a locked foot's reach (rolled): the stance ends
STEP_S = 0.32                  # duration of a short corrective step
STEP_LIFT = 0.07               # its apex / s
LANDING_WINDOW_S = 1.2         # an early lift joins the next landing if that is at most this far off
FOLLOW_REDRAW_M = 0.05         # a re-posed foot this far from the rig's next swing start: new swing
SWING_SPEED_MAX = 4.0          # m/s: peak ankle speed of a redrawn swing (lifts earlier if needed)
SET_STEP_M = 0.04              # before a lift, a foot this far from its place under the lift steps there
FOOT_STEP_MAX = 0.105          # m per frame: foot targets relative to the pelvis (rig MAX_FOOT_SPEED)
HAND_STEP_MAX = 0.18           # m per frame: hand targets relative to the shoulder (pop limit 0.20)
HIP_EXT_LIMIT_DEG = -28.0      # thigh extension limit used by the heel rise (gate -30, 2 deg margin)
BALL_ROLL_MAX = 1.1            # rad, as rig.RigBuilder
SETTLE_ROLL = np.deg2rad(4.0)  # a quasi-static foot rolled more than this ...
SETTLE_AFTER_S = 0.4           # ... for this long takes a settle step
STATIC_SPEED = 0.15            # m/s: pelvis slower than this horizontally is quasi-static
KNEEL_TOE_CLEAR = 0.004        # kneeling: lowest boot point this far above the floor
HAND_SPEED_MAX = 6.0           # m/s: hammer hand speed relative to the shoulder (hand of a nailing stroke)
SACK_CLEAR_M = 0.012           # held sack this far clear of the drawn legs and torso
SACK_SWING_MAX_DEG = 70.0      # the arms swing forward at most this far to clear the legs
SACK_SWING_RATE_DEG = 1.5      # ... changing by at most this per frame (75 deg/s at 50 fps)
ARM_REACH = 0.995              # handhold within this share of the arm's length
RAIL_BAND = (-0.20, 0.42)      # grip height relative to the shoulder that a climber keeps
RAIL_REGRIP_ABOVE = 0.34       # climbing up: new grip this far above the shoulder (at the landing)
RAIL_REGRIP_BELOW = -0.12      # climbing down: new grip this far relative to the shoulder
RAIL_MOVE_S = 0.4              # a hand's move from grip to grip
RAIL_AWAY_M = 0.06             # mid-move, the hand is this far off the rail toward the climber
REACH_PHASE_S = 0.6            # the right hand reaches the stepladder this long before the pick-up
LADDER_LIFT_FIRST = 0.35       # share of a stepladder pick-up spent lifting it upright before it turns


def requested(flag: bool | None = None) -> bool:
    """The contact pass for this run: an explicit flag wins, else THUNDERSIM_CONTACT, else DEFAULT_ON."""
    if flag is not None:
        return bool(flag)
    value = os.environ.get(ENV, "").strip()
    if value in ("0", "1"):
        return value == "1"
    return DEFAULT_ON


# =============================================================================================
# Frames
# =============================================================================================
def active_frames(ctx, scene, rig, frames: np.ndarray) -> np.ndarray:
    """Frames (N,) the pass may change: active phases, outside the ragdoll's windows."""
    from .physics import control_schedule, flight_list, physics_windows

    rb = rig.builder
    t = frames / ctx.fs
    act = np.isin(ctx.track.phase[frames], ACTIVE_PHASES)
    ctrl = control_schedule(ctx, scene, rig, frames, rb.M, flight_list(ctx.events))
    for k0, k1 in physics_windows(ctrl, t):
        act[k0:k1] = False
    return act


def edge_weight(active: np.ndarray, dt: float) -> np.ndarray:
    """1 inside active runs, easing to 0 over EDGE_S at their edges (0 outside)."""
    n = len(active)
    ramp = max(int(round(EDGE_S / dt)), 1)
    dist = np.full(n, np.inf)
    last = -np.inf
    for k in range(n):                                  # frames since the last inactive frame
        if not active[k]:
            last = k
        dist[k] = k - last
    nxt = np.inf
    back = np.full(n, np.inf)
    for k in range(n - 1, -1, -1):
        if not active[k]:
            nxt = k
        back[k] = nxt - k
    return np.where(active, smoothstep(np.minimum(dist, back) / ramp), 0.0)


def runs(mask: np.ndarray) -> list:
    """[(k0, k1), ...] of consecutive True frames."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


# =============================================================================================
# Legs
# =============================================================================================
class LegPass:
    """Feet re-posed from the rig's plants: lock, early lifts, settle steps, heel rise, kneeling."""

    def __init__(self, ctx, scene, rig, active: np.ndarray, log: dict):
        self.ctx, self.scene, self.rig, self.rb = ctx, scene, rig, rig.builder
        self.active = active
        self.log = log
        rb = self.rb
        self.N, self.dt, self.t = rb.N, rb.dt, rb.t
        self.ankle = rb.ankle_target.copy()
        self.sole_u, self.sole_n = rb.sole_u.copy(), rb.sole_n.copy()
        self.toe_bend = rb.toe_bend.copy()
        self.pole = rb.leg_pole.copy()
        self.edited = np.zeros((self.N, 2), bool)
        self.planted = rb.planted.copy()
        self.plant = {key: value.copy() for key, value in rb.plant.items()}   # edited by settle steps
        speed = np.linalg.norm(rb.vel, axis=1)
        self.quasi_static = speed < STATIC_SPEED

    # --- foot poses ------------------------------------------------------------------------
    def stance_pose(self, s: int, contact, yaw: float, k: int, ball: bool, pitch_min: float = 0.0):
        """(ankle, pitch, excess) of foot s locked on `contact` at frame k: rolled as the hip
        requires (rig.RigBuilder._planted) and at least -pitch_min onto the ball; excess = how far
        (m) the hip is still beyond the leg's reach."""
        rb = self.rb
        logged = self.plant["pitch"][k, s]
        if np.isfinite(logged) and np.linalg.norm(rb.plant["contact"][k, s] - contact) < LOCK_DRIFT_M:
            pitch = float(logged)          # the rig's own roll on this plant (rate-limited, mocap heel strike)
            ankle = rb._ankle_at(contact, yaw, pitch)
            if np.linalg.norm(rb.hips[k, s] - ankle) > rb.dims.leg * 0.995:
                ankle, pitch = rb._planted(contact, yaw, rb.hips[k, s], "ball" if ball else "auto")
        else:
            ankle, pitch = rb._planted(contact, yaw, rb.hips[k, s], "ball" if ball else "auto")
        if pitch_min < 0.0 and pitch > pitch_min:
            pitch = pitch_min
            ankle = rb._ankle_at(contact, yaw, pitch)
        excess = max(0.0, float(np.linalg.norm(rb.hips[k, s] - ankle)) - rb.dims.leg * 0.995)
        return ankle, pitch, excess

    def set_stance(self, s: int, k: int, contact, yaw: float, ball: bool, pitch_min: float = 0.0) -> float:
        ankle, pitch, excess = self.stance_pose(s, contact, yaw, k, ball, pitch_min)
        self.ankle[k, s] = ankle
        self.sole_u[k, s], self.sole_n[k, s] = self.rb._foot_axes(yaw, pitch)
        self.toe_bend[k, s] = max(-pitch, 0.0)
        self.planted[k, s] = True
        self.edited[k, s] = True
        return excess

    def set_swing(self, s: int, k0: int, k1: int, start: tuple, end: tuple, lift: float) -> None:
        """Swing frames [k0, k1): from start = (ankle, yaw, pitch) to end = (ankle, yaw, pitch), as
        rig.RigBuilder._foot_pose draws a procedural swing. The foot's lowest point clears the
        floor by SWING_CLEARANCE_M in mid-swing: a swing that starts rolled far onto the ball
        (a trailing foot after the heel rise) would otherwise drag its toe along the floor."""
        a0, yaw0, pitch0 = start
        a1, yaw1, pitch1 = end
        span = max(k1 - k0, 1)
        for k in range(k0, k1):
            u = (k - k0 + 1) / span
            w = swing_profile(u)
            bump = float(np.sin(np.pi * u))
            yaw = yaw0 + float(wrap_angle(yaw1 - yaw0)) * w
            pitch = pitch0 + (pitch1 - pitch0) * w
            sole_u, sole_n = self.rb._foot_axes(yaw, pitch)
            bend = max(-pitch0, 0.0) * float(1.0 - smoothstep(u / 0.4))
            ankle = a0 + (a1 - a0) * w + UP * (lift * bump)
            ankle = ankle + UP * self.clearance_lift(ankle, sole_u, sole_n, bend, k, u)
            self.ankle[k, s] = ankle
            self.sole_u[k, s], self.sole_n[k, s] = sole_u, sole_n
            self.toe_bend[k, s] = bend
            self.planted[k, s] = False
            self.edited[k, s] = True

    def clearance_lift(self, ankle, sole_u, sole_n, bend: float, k: int, u: float) -> float:
        """Lift (m) that puts the foot's lowest drawn point (heel, ball or bent toe tip, as solve()
        draws them) SWING_CLEARANCE_M above the floor under it, easing in and out over the first
        and last CLEARANCE_EASE of the swing (the same rule as mocap.Session.clearance_lift)."""
        need = SWING_CLEARANCE_M * float(smoothstep(u / CLEARANCE_EASE)) \
            * (1.0 - float(smoothstep((u - 1.0 + CLEARANCE_EASE) / CLEARANCE_EASE)))
        if need <= 0.0:
            return 0.0
        d = self.rb.dims
        sole = ankle - sole_n * d.ankle_h
        ball = sole + sole_u * d.ball
        toe = ball + (sole_u * np.cos(bend) + sole_n * np.sin(bend)) * (d.toe - d.ball)
        heel = sole - sole_u * d.heel
        lowest = min(float(heel[1]), float(ball[1]), float(toe[1]))
        return max(self.rb.floor_at(ankle, k) + need - lowest, 0.0)

    def pose_at(self, s: int, k: int) -> tuple:
        """(ankle, yaw, pitch) of the current foot target at frame k."""
        u = self.sole_u[k, s]
        yaw = float(np.arctan2(-u[0], -u[2])) if np.hypot(u[0], u[2]) > 1e-6 else float(self.rb.heading[k])
        pitch = float(np.arcsin(np.clip(u[1], -1.0, 1.0)))
        return self.ankle[k, s].copy(), yaw, pitch

    # --- hip extension ---------------------------------------------------------------------
    def hip_extension(self, s: int, ks: np.ndarray, ankles: np.ndarray) -> np.ndarray:
        """Hip flexion (deg, NaN out of the sagittal plane) of leg s at frames ks for ankle targets."""
        rb = self.rb
        knee, _ = two_bone(rb.hips[ks, s], ankles, self.pole[ks, s], rb.dims.thigh, rb.dims.shank)
        R = rb.M[ks, 0]
        thigh = np.einsum("nji,nj->ni", R, knee - rb.hips[ks, s])
        hip = np.degrees(np.arctan2(-thigh[:, 2], -thigh[:, 1]))
        hip = np.where(hip < -90.0, hip + 360.0, hip)
        in_plane = np.hypot(thigh[:, 1], thigh[:, 2]) >= 0.5 * np.linalg.norm(thigh, axis=1)
        return np.where(in_plane, hip, np.nan)

    def heel_rise_needed(self, s: int, k0: int, k1: int, contact, yaw: float, ball: bool) -> np.ndarray:
        """Least ball roll (rad, <= 0 pitch) per frame of a locked stance [k0, k1) that keeps the hip
        extension within HIP_EXT_LIMIT_DEG; 0 where no extra roll is needed, NaN where even the
        largest roll does not."""
        rb = self.rb
        ks = np.arange(k0, k1)
        base = np.array([self.stance_pose(s, contact, yaw, k, ball)[1] for k in ks])
        ankles = np.array([rb._ankle_at(contact, yaw, p) for p in base])
        hip = self.hip_extension(s, ks, ankles)
        need = np.zeros(len(ks))
        bad = np.flatnonzero(hip < HIP_EXT_LIMIT_DEG)
        for i in bad:
            k = ks[i]
            found = np.nan
            for roll in np.linspace(max(-base[i], 0.0), BALL_ROLL_MAX, 23)[1:]:
                a = rb._ankle_at(contact, yaw, -roll)
                hip_now = self.hip_extension(s, np.array([k]), a[None])[0]
                if not hip_now < HIP_EXT_LIMIT_DEG:          # within the limit (or out of plane)
                    found = -roll
                    break
            need[i] = found
        return need

    # --- the per-foot pass -------------------------------------------------------------------
    def lift_stance(self) -> list:
        """Per foot, [(frame, contact, yaw), ...]: before each lift and put-down a foot that is not
        where the simulator's lift geometry puts the ankles (motions.lift_root: the ankles under the
        standing pelvis, feet a hip width apart) steps there, the farther foot first."""
        from ..motions import knee_at_stand, lift_root

        rb, ctx = self.rb, self.ctx
        out: list = [[], []]
        H = ctx.H
        k_stand = knee_at_stand(ctx.worker.pelvis_height / H)
        short = int(round(STEP_S / self.dt))
        for c in ctx.rec.calls:
            if c.name not in ("lift", "put_down"):
                continue
            k0 = rb.ix(c.t0)
            if not self.active[k0] or not rb.planted[k0].all():
                continue
            i = ctx.idx(c.t0)
            heading = float(ctx.track.heading[i])
            shift = rb.P[k0, 0] - ctx.spine_point(c.t0, 0)              # the viewer's render shift
            root = ctx.track.root_pos[i] + shift
            ankle = root - fwd(heading) * float(lift_root(k_stand, float(ctx.track.pitch[i]), H)[0])
            targets, dists = [], []
            for side in (0, 1):
                p = flat(ankle) + right(heading) * (SIDES[side] * 0.1 * rb.dims.s)
                p[1] = rb.floor_at(p, k0)
                targets.append(p)
                dists.append(float(np.linalg.norm(flat(self.plant["contact"][k0, side] - p))))
            k_step = k0 + int(round(0.1 / self.dt))
            for side in sorted((0, 1), key=lambda x: -dists[x]):
                if dists[side] > SET_STEP_M:
                    out[side].append((k_step, targets[side], heading))
                    k_step += short + 3
        return out

    def run(self) -> None:
        self.set_steps = self.lift_stance()
        stand = self.stand_frames()
        # A foot the rig calls planted inside a blend between modes is being lerped, not standing.
        self.planted &= stand[:, None] | ~self.active[:, None]
        for s in (0, 1):
            for k0, k1 in runs(self.rb.planted[:, s] & stand & self.active):
                self.stance(s, k0, k1)
        self.swing_hip()
        self.no_slide(stand)
        self.kneeling()

    def stand_frames(self) -> np.ndarray:
        """Frames (N,) in a stand item and not in its blend from another mode (kneel, ladder...):
        there the rig lerps the feet between poses, and a lock would snap at the blend's middle."""
        rb = self.rb
        out = np.zeros(self.N, bool)
        self.blend_end = np.full(self.N, -1)
        items = rb.leg_items
        for i, item in enumerate(items):
            k0 = rb.ix(item.t0) if i > 0 else 0
            k1 = rb.ix(items[i + 1].t0) if i + 1 < len(items) else self.N
            kb = rb.ix(item.t1) if i > 0 else 0
            self.blend_end[k0:max(kb, k0)] = kb
            if item.mode != "stand":
                continue
            prev = items[i - 1].mode if i > 0 else "stand"
            out[(kb if prev != "stand" else k0):k1] = True
        return out

    def no_slide(self, stand: np.ndarray) -> None:
        """A foot that the rig moves along the floor outside a stance (the blend between modes, e.g.
        standing <-> kneeling or onto a ladder) steps instead of sliding: it lifts in an arc. When
        both feet would move at once, they go one after the other (the farther-travelling foot
        first) and each waits on the floor while the other moves."""
        rb = self.rb
        d = rb.dims
        blend = (self.blend_end >= 0) & ~stand          # blending between two modes
        moving = []
        for s in (0, 1):
            low = np.array([boot_low(self.ankle[k, s], self.sole_u[k, s], self.sole_n[k, s], d) - rb.floor_at(self.ankle[k, s], k)
                            for k in range(self.N)])
            speed = np.zeros(self.N)
            speed[1:] = np.linalg.norm(np.diff(self.ankle[:, s][:, [0, 2]], axis=0), axis=1) / self.dt
            free = self.active & ~self.planted[:, s]
            sliding = free & (low <= 0.015) & (speed > 0.1) & blend
            moving.append([(k0, k1, speed) for k0, k1 in runs(sliding)])
        done: set = set()
        for s in (0, 1):
            for k0, k1, speed in moving[s]:
                if (s, k0) in done:
                    continue
                other = [(a, b) for a, b, _ in moving[1 - s] if a < k1 and b > k0 and (1 - s, a) not in done]
                if other:
                    a, b = min(k0, other[0][0]), max(k1, other[0][1])
                    done |= {(s, k0), (1 - s, other[0][0])}
                    self.one_after_other(a, b)
                    continue
                done.add((s, k0))
                self.arc(s, k0, k1, float(np.sum(speed[k0:k1]) * self.dt))

    def arc(self, s: int, k0: int, k1: int, travel: float) -> None:
        """Lift foot s in an arc over [k0, k1) that moves `travel` m along the floor."""
        if travel < 0.02 or k1 - k0 < 3:
            return
        lift = min(0.4 * travel, 0.12)
        u = (np.arange(k0, k1) - k0 + 0.5) / (k1 - k0)
        self.ankle[k0:k1, s, 1] += lift * np.sin(np.pi * u)
        self.edited[k0:k1, s] = True
        self.log["slides_lifted"] = self.log.get("slides_lifted", 0) + 1

    def one_after_other(self, a: int, b: int) -> None:
        """Both feet move over [a, b): re-time each foot's path so the farther-travelling one moves
        in the first half and the other in the second, each holding still (planted) otherwise."""
        if b - a < 6:
            return
        travel = [float(np.linalg.norm(np.diff(self.ankle[a:b, s][:, [0, 2]], axis=0), axis=1).sum()) for s in (0, 1)]
        first = int(np.argmax(travel))
        m = (a + b) // 2
        orig = {key: getattr(self, key)[a:b].copy() for key in ("ankle", "sole_u", "sole_n", "toe_bend")}
        for s, (s0, s1) in ((first, (a, m)), (1 - first, (m, b))):
            for k in range(a, b):
                if k < s0:
                    x = 0.0
                elif k >= s1:
                    x = float(b - 1 - a)
                else:
                    x = (k - s0 + 1) / (s1 - s0) * (b - 1 - a)
                i0 = int(np.floor(x))
                i1 = min(i0 + 1, b - 1 - a)
                w = x - i0
                for key in ("ankle", "sole_u", "sole_n", "toe_bend"):
                    getattr(self, key)[k, s] = orig[key][i0, s] * (1 - w) + orig[key][i1, s] * w
                self.sole_u[k, s], self.sole_n[k, s] = unit(self.sole_u[k, s]), unit(self.sole_n[k, s])
                self.planted[k, s] = not (s0 <= k < s1)
                self.edited[k, s] = True
            self.arc(s, s0, s1, travel[s])
        self.log["blends_stepped"] = self.log.get("blends_stepped", 0) + 1

    def swing_hip(self) -> None:
        """Swinging feet in stand mode whose thigh is still extended past HIP_EXT_LIMIT_DEG (just
        after toe-off) are raised a little, so the knee flexes and the thigh comes forward."""
        rb = self.rb
        stand = self.mode_mask("stand")
        for s in (0, 1):
            ks = np.flatnonzero(self.active & stand & ~self.planted[:, s])
            if not len(ks):
                continue
            hip = self.hip_extension(s, ks, self.ankle[ks, s])
            bad = hip < HIP_EXT_LIMIT_DEG
            if not bad.any():
                continue
            need = np.zeros(self.N)
            for i in np.flatnonzero(bad):
                k = ks[i]
                for lift in np.arange(0.01, 0.161, 0.01):
                    if not self.hip_extension(s, np.array([k]), (self.ankle[k, s] + UP * lift)[None])[0] < HIP_EXT_LIMIT_DEG:
                        need[k] = lift
                        break
                else:
                    need[k] = 0.16
            raise_m = envelope(need, 0.01)
            raise_m[self.planted[:, s] | ~self.active] = 0.0       # never lift a planted foot
            self.ankle[:, s, 1] += raise_m
            self.edited[raise_m > 0, s] = True
            self.log["swing_raises"] = self.log.get("swing_raises", 0) + int((need > 0).sum())

    def mode_mask(self, mode: str) -> np.ndarray:
        """Frames (N,) whose leg item (rig timeline, after its blend) is `mode`."""
        rb = self.rb
        out = np.zeros(self.N, bool)
        items = rb.leg_items
        for i, item in enumerate(items):
            if item.mode != mode:
                continue
            k0 = rb.ix(item.t0) if i > 0 else 0
            k1 = rb.ix(items[i + 1].t0) if i + 1 < len(items) else self.N
            out[k0:k1] = True
        return out

    def next_landing(self, s: int, k: int):
        """First frame >= k where the rig plants foot s again, or None."""
        later = np.flatnonzero(self.rb.planted[k:, s])
        return int(k + later[0]) if len(later) else None

    def stance(self, s: int, a: int, b: int) -> None:
        """Re-pose one planted stance [a, b) of foot s."""
        rb = self.rb
        plant = self.plant
        k = a
        while k < b:
            contact = plant["contact"][k, s].copy()
            yaw = float(plant["yaw"][k, s])
            ball = bool(plant["ball"][k, s])
            if np.isnan(contact).any():
                return
            # Frames this lock can hold: the rig keeps the plant and the hip stays in reach.
            end = b
            need = self.heel_rise_needed(s, k, b, contact, yaw, ball)
            for j in range(k, b):
                moved = np.linalg.norm(plant["contact"][j, s] - contact) > LOCK_DRIFT_M
                floor = need[j - k] if np.isfinite(need[j - k]) else 0.0
                _, _, excess = self.stance_pose(s, contact, yaw, j, ball, floor)
                if moved or excess > REACH_EXCESS_M or np.isnan(need[j - k]):
                    end = j
                    break
            settle = self.settle_frame(s, k, end, contact, yaw, ball)
            set_step = next(((ks, c, y) for ks, c, y in self.set_steps[s] if k + 2 <= ks < end), None)
            if set_step is not None and (settle is None or set_step[0] <= settle):
                settle = None
                end = set_step[0]
            elif settle is not None and settle < end:
                end = settle
            pitch_floor = smooth_roll(np.nan_to_num(need[: end - k], nan=-BALL_ROLL_MAX))
            for j in range(k, end):
                self.set_stance(s, j, contact, yaw, ball, pitch_floor[j - k])
            if end >= b:
                k_next = min(b, self.N - 1)
                into_blend = rb.planted[k_next, s] and self.blend_end[k_next] > b   # into another mode
                self.follow_on(s, b - 1, self.blend_end[k_next] if into_blend else None)
                return
            # The stance ends early: lift now and land at the next plant, or step under the hip.
            start = self.pose_at(s, end - 1) if end > k else self.pose_at(s, end)
            land = self.next_landing(s, b)
            short = int(round(STEP_S / self.dt))
            if set_step is not None and set_step[0] == end and end + short < b:
                k_land = end + short
                _, target, yaw_new = set_step
                self.log["lift_set_steps"] = self.log.get("lift_set_steps", 0) + 1
                self.place_step(s, end, k_land, start, target, yaw_new)
                plant["contact"][k_land:b, s] = target
                plant["yaw"][k_land:b, s] = yaw_new
                plant["ball"][k_land:b, s] = False
                self.set_steps[s] = [x for x in self.set_steps[s] if x[0] != set_step[0]]
                k = k_land
                continue
            if settle is not None and settle == end:
                k_land = min(end + short, b - 1)
                target = rb._ground_point(rb.home(k_land, s), k_land)
                if target is None or k_land <= end:
                    return
                self.log["settle_steps"] += 1
                self.place_step(s, end, k_land, start, target, float(rb.heading[k_land]))
                plant["contact"][k_land:b, s] = target              # the new plant holds to b
                plant["yaw"][k_land:b, s] = float(rb.heading[k_land])
                plant["ball"][k_land:b, s] = False
                k = k_land
                continue
            if land is not None and (land - end) * self.dt <= LANDING_WINDOW_S and self.active[end:land + 1].all():
                self.log["early_lifts"] += 1
                end_pose = self.pose_at(s, land)
                self.set_swing(s, end, land, start, end_pose, STEP_LIFT * rb.dims.s)
                return
            if b - end < 3:
                self.follow_on(s, end - 1, b)                    # no room for a step: the rig's drag
                self.log["drag_frames"] = self.log.get("drag_frames", 0) + (b - end)
                return
            k_land = min(end + short, b - 1)
            target = plant["contact"][k_land, s].copy()          # a quick step goes where the rig drags
            hip_floor = rb._ground_point(rb.home(k_land, s), k_land)
            if hip_floor is not None and k_land - end >= short:
                target = hip_floor
            self.log["catch_steps"] += 1
            self.place_step(s, end, k_land, start, target, float(rb.heading[k_land]))
            plant["contact"][k_land:b, s] = target
            plant["yaw"][k_land:b, s] = float(rb.heading[k_land])
            plant["ball"][k_land:b, s] = False
            k = k_land

    def follow_on(self, s: int, k_last: int, k_end: int | None = None) -> None:
        """The rig's own swing (or blend into another mode, up to k_end) after a re-posed stance
        starts from the re-posed foot: the offset at the stance's last frame fades out over it (its
        shape, e.g. a mocap swing, is kept)."""
        rb = self.rb
        offset = self.ankle[k_last, s] - rb.ankle_target[k_last, s]
        if np.linalg.norm(offset) < 1e-4 or k_last + 1 >= self.N:
            return
        land = self.next_landing(s, k_last + 1) if k_end is None else None
        slowest = k_last + int(round(max(0.4, np.linalg.norm(offset) / 0.8) / self.dt))   # fade <= 0.8 m/s
        if k_end is not None:
            k_end = min(max(int(k_end), slowest), self.N)
        elif land is not None and (land - k_last) * self.dt <= LANDING_WINDOW_S:
            if np.linalg.norm(offset) > FOLLOW_REDRAW_M and self.active[k_last:land + 1].all():
                # Far from where the rig's swing starts (it had dragged its foot): a new swing,
                # lifting earlier if it would otherwise be faster than SWING_SPEED_MAX.
                start = self.pose_at(s, k_last)
                end = self.pose_at(s, land)
                frames_needed = int(np.ceil(1.43 * np.linalg.norm(end[0] - start[0]) / (SWING_SPEED_MAX * self.dt)))
                k_lift = min(k_last + 1, max(land - frames_needed, k_last - int(round(0.3 / self.dt))))
                start = self.pose_at(s, k_lift - 1)
                self.set_swing(s, k_lift, land, start, end, STEP_LIFT * rb.dims.s)
                self.log["swings_redrawn"] = self.log.get("swings_redrawn", 0) + 1
                return
            k_end = land
        else:
            k_end = min(slowest, self.N)
        for k in range(k_last + 1, k_end):
            u = (k - k_last) / (k_end - k_last)
            self.ankle[k, s] = self.ankle[k, s] + offset * (1.0 - smoothstep(u))
            self.edited[k, s] = True

    def place_step(self, s: int, k0: int, k1: int, start: tuple, contact, yaw: float) -> None:
        ankle, pitch, _ = self.stance_pose(s, contact, yaw, k1, False)
        self.set_swing(s, k0, k1, start, (ankle, yaw, pitch), STEP_LIFT * self.rb.dims.s)

    def settle_frame(self, s: int, k0: int, k1: int, contact, yaw: float, ball: bool):
        """First frame of a settle step inside a lock [k0, k1): the foot has been rolled by more than
        SETTLE_ROLL for SETTLE_AFTER_S while the body is quasi-static and the other foot is planted
        (a staggered stop leaves a heel up); None if no step is needed."""
        if ball:
            return None
        rb = self.rb
        other = 1 - s
        need = int(round(SETTLE_AFTER_S / self.dt))
        short = int(round(STEP_S / self.dt))
        count = 0
        for j in range(k0, k1):
            _, pitch, _ = self.stance_pose(s, contact, yaw, j, ball)
            calm = self.quasi_static[j] and rb.planted[j, other] and abs(pitch) > SETTLE_ROLL
            count = count + 1 if calm else 0
            if count >= need:
                start = max(j - need + 1, k0 + int(round(0.1 / self.dt)))   # not right at the landing
                # The step must fit while the other foot stays planted and the body stays calm.
                if start + short < k1 and rb.planted[start:start + short + 1, other].all() \
                        and self.quasi_static[start:start + short + 1].all():
                    home = rb._ground_point(rb.home(start + short, s), start + short)
                    if home is not None and np.linalg.norm(flat(home - contact)) > 0.03:
                        return start
                count = 0
        return None

    # --- kneeling ----------------------------------------------------------------------------
    def kneeling(self) -> None:
        """Kneel / quad items: the feet lie plantarflexed with the toe end on the floor. The tilt
        eases in after the rig's blend into the item and out before the next item."""
        rb = self.rb
        for item_i, item in enumerate(rb.leg_items):
            if item.mode not in ("kneel", "quad"):
                continue
            k0 = rb.ix(item.t1)
            k1 = rb.ix(rb.leg_items[item_i + 1].t0) if item_i + 1 < len(rb.leg_items) else rb.N
            if k1 - k0 < 3:
                continue
            ramp = edge_weight(np.ones(k1 - k0, bool) & self.active[k0:k1], self.dt)
            ramp = np.minimum(ramp, edge_weight(np.r_[False, np.ones(k1 - k0 - 2, bool), False], self.dt))
            for k in range(k0, k1):
                if ramp[k - k0] <= 0.0:
                    continue
                for s in (0, 1):
                    self._toe_down(s, k, ramp[k - k0])
            self.log["kneel_frames"] += int(np.sum(ramp > 0))

    def _toe_down(self, s: int, k: int, amount: float) -> None:
        """Tilt a plantarflexed foot (sole facing up, toes back) about its across axis until the
        boot's lowest point rests KNEEL_TOE_CLEAR above the floor; `amount` (0..1) of that tilt."""
        rb = self.rb
        d = rb.dims
        u0, n0 = self.sole_u[k, s], self.sole_n[k, s]
        if n0[1] > -0.5:
            return                                        # not a foot lying sole up
        ankle = self.ankle[k, s]
        floor = rb.floor_at(ankle, k)
        axis = unit(np.cross(u0, n0))
        if np.cross(axis, u0)[1] > 0.0:                   # turn the toe end down, not up
            axis = -axis
        tilt = 0.0
        for g in np.radians(np.arange(0.0, 61.0, 0.5)):
            tilt = g
            if boot_low(ankle, rotate(u0, axis, g), rotate(n0, axis, g), d) <= floor + KNEEL_TOE_CLEAR:
                break
        self.sole_u[k, s] = rotate(u0, axis, tilt * amount)
        self.sole_n[k, s] = rotate(n0, axis, tilt * amount)
        self.edited[k, s] = True

    # --- output --------------------------------------------------------------------------------
    def solve(self, weight: np.ndarray) -> dict:
        """Leg joints from the edited targets, blended with the rig's by `weight` (N,)."""
        rb = self.rb
        d = rb.dims
        w = weight[:, None, None]
        ankle_t = rb.ankle_target + (self.ankle - rb.ankle_target) * w
        pelvis = rb.P[:, 0, None, :]
        ankle_t = pelvis + speed_limited(ankle_t - pelvis, FOOT_STEP_MAX)
        u = unit(rb.sole_u + (self.sole_u - rb.sole_u) * w)
        nrm = unit(rb.sole_n + (self.sole_n - rb.sole_n) * w)
        bend = rb.toe_bend + (self.toe_bend - rb.toe_bend) * weight[:, None]
        knee, ankle = two_bone(rb.hips, ankle_t, self.pole, d.thigh, d.shank, continuous=True)
        sole = ankle - nrm * d.ankle_h
        heel = sole - u * d.heel
        ball = sole + u * d.ball
        toe_dir = u * np.cos(bend)[..., None] + nrm * np.sin(bend)[..., None]
        toe = ball + toe_dir * (d.toe - d.ball)
        return {"knee": knee, "ankle": ankle, "heel": heel, "toe": toe, "ankle_target": ankle_t,
                "planted": np.where(weight[:, None] >= 0.5, self.planted, rb.planted)}


def smooth_roll(pitch_min: np.ndarray) -> np.ndarray:
    """Envelope of the needed ball rolls (pitch <= 0) that changes by at most PLANTED_ROLL_STEP / 2
    per frame and is never less rolled than needed."""
    out = np.minimum(pitch_min, 0.0).copy()
    rate = PLANTED_ROLL_STEP / 2
    for i in range(1, len(out)):
        out[i] = min(out[i], out[i - 1] + rate)
    for i in range(len(out) - 2, -1, -1):
        out[i] = min(out[i], out[i + 1] + rate)
    return out


def rotate(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation of v about a unit axis."""
    return v * np.cos(angle) + np.cross(axis, v) * np.sin(angle) + axis * (axis @ v) * (1 - np.cos(angle))


def boot_low(ankle, u, nrm, d) -> float:
    """Lowest point (y) of the drawn boot (body.js: the heel-toe line raised toward the ankle by the
    boot's thickness) for a foot at ankle with sole axes u, nrm."""
    from .scene import BOOT_R

    sole = ankle - nrm * d.ankle_h
    heel = sole - u * d.heel
    toe = sole + u * d.toe
    up = unit(ankle - heel - u * float((ankle - heel) @ u))
    lows = [p[1] + up[1] * BOOT_R * d.s - BOOT_R * d.s for p in (heel, toe)]
    return float(min(lows))


# =============================================================================================
# Arms and held objects
# =============================================================================================
class ArmPass:
    """Hand targets for held objects (sack, ladder rails, stepladder) and hammer strokes."""

    def __init__(self, ctx, scene, rig, active: np.ndarray, legs: dict, log: dict):
        self.ctx, self.scene, self.rig, self.rb = ctx, scene, rig, rig.builder
        self.active, self.legs, self.log = active, legs, log
        rb = self.rb
        self.N, self.dt, self.t = rb.N, rb.dt, rb.t
        self.hand = rb.hand_target.copy()
        self.arm_len = rb.dims.upper_arm + rb.dims.forearm
        self.held: dict = {}
        self.unreachable: dict = {}
        self.rail_grips: list = []
        self.sack_channels: list = []            # sack channel (bag<k>) of each carry
        self.rail_end: dict = {}                 # ladder index -> time the hands leave its rails
        self.stepladder_along: list = []         # [[pick frame, handhold along the rail], ...]

    # --- helpers ---------------------------------------------------------------------------
    def items(self, mode: str) -> list:
        return [(i, it) for i, it in enumerate(self.rb.arm_items) if it.mode == mode]

    def next_item(self, i: int):
        items = self.rb.arm_items
        return items[i + 1] if i + 1 < len(items) else None

    def reachable(self, s: int, k: int, point: np.ndarray, margin: float = ARM_REACH) -> bool:
        """The hand reaches `point` at frame k: within `margin` of the arm's length and not closer
        to the shoulder than the elbow's flexion limit allows (rig.MIN_JOINT_ANGLE)."""
        d = self.rb.dims
        dist = float(np.linalg.norm(point - self.rb.shoulders[k, s]))
        folded = np.sqrt(d.upper_arm ** 2 + d.forearm ** 2 - 2 * d.upper_arm * d.forearm * np.cos(MIN_JOINT_ANGLE))
        return folded + 0.01 <= dist <= margin * self.arm_len

    def note_unreachable(self, key: str, s: int, k: int, point: np.ndarray) -> None:
        if not self.reachable(s, k, point):
            self.unreachable.setdefault(f"{key}:{s}", []).append(int(k))

    def fade_offset(self, s: int, k_edge: int, offset: np.ndarray, k_other: int) -> None:
        """Add `offset` to hand s at k_edge, fading to 0 at k_other (either side), smoothly."""
        if k_other == k_edge:
            return
        lo, hi = sorted((k_edge, k_other))
        for k in range(max(lo, 0), min(hi + 1, self.N)):
            u = abs(k - k_other) / abs(k_edge - k_other)
            self.hand[k, s] += offset * float(smoothstep(u))

    # --- sacks -----------------------------------------------------------------------------
    def sacks(self) -> None:
        ctx, scene, rb = self.ctx, self.scene, self.rb
        plan = sack_plan(ctx, scene)
        if not plan:
            return
        t, N = self.t, self.N
        owner: list = []                         # carries (plan indices) of each physical sack
        for sk in plan:
            same = None
            for ch, prev in enumerate(owner):
                if np.linalg.norm(flat(plan[prev[-1]].drop - sk.pick)) < 0.05:
                    same = ch
            if same is None:
                owner.append([plan.index(sk)])
            else:
                owner[same].append(plan.index(sk))
        poses = []
        self.sack_channels = [0] * len(plan)
        for ch, chain in enumerate(owner):
            pose = np.zeros((N, 4))
            first = plan[chain[0]]
            pose[:, :3], pose[:, 3] = first.pick, first.yaw_pick
            for ci, idx in enumerate(chain):
                sk = plan[idx]
                self.sack_channels[idx] = ch
                if ci > 0:                       # the same sack, lying where it was set down
                    prev = plan[chain[ci - 1]]
                    sk.pick = prev.extra["drop_pose"][:3].copy()
                    sk.yaw_pick = float(prev.extra["drop_pose"][3])
                self.carry_sack(sk, pose)
            poses.append(pose)
        for ch, pose in enumerate(poses):
            pose[:, 3] = np.unwrap(pose[:, 3])
            self.held[f"bag{ch}"] = pose
        self.held["bag_mass"] = [round(float(max(plan[i].mass for i in chain)), 1) for chain in owner]
        self.log["sacks"] = len(poses)

    def carry_sack(self, sk, pose: np.ndarray) -> None:
        """Pose of one sack over its carry (written into pose from its grasp on) and the hand targets."""
        rb, t = self.rb, self.t
        lift_item = next(((i, it) for i, it in self.items("grip") if abs(it.t1 - sk.t_grasp) < 0.05), None)
        k_grasp, k_off = rb.ix(sk.t_grasp), rb.ix(sk.t_off)
        k_touch, k_rel = rb.ix(sk.t_touch), rb.ix(sk.t_release)
        k_reach0 = rb.ix(lift_item[1].t0) if lift_item else max(k_grasp - 25, 0)
        grips = sk.grips(t)                                 # (N, 2) handhold heights on the sack
        half_up = UP * (grips.mean(axis=1) - SACK_HALF[1])[:, None]   # sack centre -> handhold midpoint
        sh_mid = rb.shoulders.mean(axis=1)
        heading = rb.heading
        # --- path of the handhold midpoint while held ---------------------------------------
        h_pick = sk.pick + UP * (sk.grip_pick - SACK_HALF[1])
        put_item = next(((i, it) for i, it in self.items("hold") if it.t0 > sk.t_off + 0.1 and it.t1 <= sk.t_touch + 0.05), None)
        carry_item = next(((i, it) for i, it in self.items("carry") if sk.t_off < it.t0 < sk.t_touch), None)
        ks = np.arange(k_off, k_touch + 1)
        hang_pick = sh_mid[ks] + (h_pick - sh_mid[k_off])          # hanging from the shoulders
        drop_info = next(((tt, info) for tt, n, info in self.ctx.events if n == "put_down_touch" and abs(tt - sk.t_touch) < 1e-6), None)
        c_drop = np.array([drop_info[1]["gx"], 0.0, drop_info[1]["gz"]])
        c_drop[1] = self.scene.support(c_drop[0], c_drop[2], float(drop_info[1]["gy"]) + 0.3) + SACK_HALF[1]
        h_drop = c_drop + UP * (sk.grip_drop - SACK_HALF[1])
        hang_drop = sh_mid[ks] + (h_drop - sh_mid[k_touch])
        # Carried: the rig's carry hand path holds the sack by its handholds.
        carry = rb.hand_target[ks].mean(axis=1) if carry_item else hang_pick
        w_carry = np.zeros(len(ks))
        if carry_item:
            w_carry = smoothstep((t[ks] - carry_item[1].t0) / max(carry_item[1].t1 - carry_item[1].t0, 1e-3))
        w_put = np.zeros(len(ks))
        if put_item:
            it = put_item[1]
            w_put = smoothstep((t[ks] - it.t0) / max(it.t1 - it.t0, 1e-3))
        path = hang_pick + (carry - hang_pick) * w_carry[:, None]
        path = path + (hang_drop - path) * w_put[:, None]
        yaw = sk.yaw_pick + (heading[ks] - heading[k_off])
        # --- clearance and reach (only the sack in the air can move) -------------------------
        path = self.clear_sack(ks, path, yaw, grips[ks], fixed_ends=True)
        # --- write the sack pose ---------------------------------------------------------------
        pose[k_grasp:k_off, :3], pose[k_grasp:k_off, 3] = sk.pick, sk.yaw_pick
        pose[ks, :3] = path - half_up[ks]
        pose[ks, 3] = yaw
        drop_pose = np.r_[pose[k_touch, :3], pose[k_touch, 3]]
        pose[k_touch:, :3], pose[k_touch:, 3] = drop_pose[:3], drop_pose[3]
        sk.extra["drop_pose"] = drop_pose
        # --- hands -----------------------------------------------------------------------------
        hh = sack_handholds(pose[:, :3], pose[:, 3], grips)
        for s in (0, 1):
            rest_pick = hh[k_grasp, s]
            # reach: from the rig's target to the handhold, arriving at lift_grasp
            for k in range(k_reach0, k_grasp):
                u = smoothstep((t[k] - t[k_reach0]) / max(t[k_grasp] - t[k_reach0], 1e-3))
                self.hand[k, s] = rb.hand_target[k, s] + (rest_pick - rb.hand_target[k, s]) * u
            for k in range(k_grasp, k_rel + 1):
                self.hand[k, s] = hh[k, s]
                self.note_unreachable(f"sack{sk.index}", s, k, hh[k, s])
            # release: back to the rig's pose over its blend (the 'idle' item after the release)
            idle = next((it for _, it in self.items("idle") if abs(it.t0 - sk.t_release) < 0.05), None)
            k_end = rb.ix(idle.t1) if idle else min(k_rel + 40, self.N - 1)
            self.fade_offset(s, k_rel + 1, hh[k_rel, s] - rb.hand_target[min(k_rel + 1, self.N - 1), s], k_end)

    def clear_sack(self, ks: np.ndarray, path: np.ndarray, yaw: np.ndarray, grips: np.ndarray, fixed_ends: bool) -> np.ndarray:
        """Swing the arms forward about the shoulders, at the arms' current length, just enough for
        the held sack to clear the drawn thighs, shanks and torso by SACK_CLEAR_M. Keeping the
        shoulder-to-hand distance keeps the elbows as they are: the load is not curled up to clear
        the legs. The swing angle is smoothed (at most SACK_SWING_RATE_DEG per frame) and is 0 at the
        path's ends. grips (len(ks), 2): handhold heights on the sack per frame and hand."""
        rb, ctx = self.rb, self.ctx
        legs = self.legs
        J = np.zeros((self.N, 18, 3))
        J[:, 0], J[:, 1] = rb.hips[:, 0], rb.hips[:, 1]
        J[:, 2:4], J[:, 4:6] = legs["knee"], legs["ankle"]
        s_, w_ = body_scale(ctx.worker)
        lc, lr, _ = leg_spheres(J[ks], s_, w_)
        tc, tr = torso_spheres(ctx, rb.P[ks], rb.frames[ks])
        cen = np.concatenate([lc, tc], axis=1)
        rad = np.concatenate([lr, tr])
        R = sack_axes(yaw)
        half_up = UP * (grips.mean(axis=1) - SACK_HALF[1])[:, None]
        fwd_dir = fwd(rb.heading[ks])
        sh_mid = rb.shoulders[ks].mean(axis=1)
        angles = np.radians(np.arange(0.0, SACK_SWING_MAX_DEG + 0.1, 1.0))
        need = np.zeros(len(ks))
        for i in range(len(ks)):
            c = swing_forward(sh_mid[i], path[i], fwd_dir[i], angles) - half_up[i]
            depth = (rad[None, :] - box_signed_distance(np.broadcast_to(cen[i], (len(angles),) + cen[i].shape),
                                                         c, np.broadcast_to(R[i], (len(angles), 3, 3)), SACK_HALF)).max(axis=1)
            ok = depth <= -SACK_CLEAR_M
            if ok.any():
                need[i] = angles[int(np.argmax(ok))]
            else:
                need[i] = angles[-1]
                self.log["sack_clear_failed"] += 1
        rate = np.radians(SACK_SWING_RATE_DEG)
        if fixed_ends:
            # Lift-off and touchdown are fixed (the sack rests there): ramp from and to zero.
            need[0] = need[-1] = 0.0
        swing = envelope(need, rate)
        if fixed_ends:
            ramp = np.minimum(np.arange(len(ks)), np.arange(len(ks))[::-1]) * rate
            swing = np.minimum(swing, ramp)
        moved = np.array([swing_forward(sh_mid[i], path[i], fwd_dir[i], swing[i:i + 1])[0] for i in range(len(ks))])
        self.log["sack_shift_max_m"] = round(max(self.log.get("sack_shift_max_m", 0.0),
                                                 float(np.linalg.norm(moved - path, axis=1).max())), 3)
        self.log["sack_swing_max_deg"] = round(max(self.log.get("sack_swing_max_deg", 0.0), float(np.degrees(swing.max()))), 1)
        path = moved
        # Reach: pull the sack toward the shoulders where a handhold is beyond the arm (smoothly).
        excess = np.zeros(len(ks))
        for i, k in enumerate(ks):
            hh = sack_handholds((path[i] - half_up[i])[None], np.array([yaw[i]]), grips[i][None])[0]
            excess[i] = max(float(np.linalg.norm(hh[s] - rb.shoulders[k, s])) - 0.99 * self.arm_len for s in (0, 1))
        pull = envelope(np.maximum(excess, 0.0), 0.004)
        return path + unit(sh_mid - path) * pull[:, None]

    # --- ladder rails ------------------------------------------------------------------------
    def rails(self) -> None:
        rb, scene = self.rb, self.scene
        items = rb.arm_items
        for i, it in self.items("ladder"):
            lad = it.info.get("ladder") or rb._ladder_at(it.t0)
            if lad is None:
                continue
            nxt = self.next_item(i)
            ka = rb.ix(it.t1)
            kb = rb.ix(nxt.t0) if nxt is not None else self.N
            if kb - ka < 5 or not self.active[ka:kb].any():
                continue
            li = next(j for j, l in enumerate(scene.ladders) if l is lad)
            self.rail_plan(lad, li, rb.ix(it.t0), ka, kb, nxt)

    def rail_plan(self, lad, li: int, k_in: int, ka: int, kb: int, nxt) -> None:
        """Hand over hand on the rails over [ka, kb). Each hand is gripping (a fixed height),
        moving (to a new grip, or to / from hanging free) or free (hanging, when no grip is in
        reach, e.g. standing high on a stepladder). A hand starts a move only while the other hand
        is not moving."""
        rb = self.rb
        sh = rb.shoulders
        n_move = int(round(RAIL_MOVE_S / self.dt))
        top_y = lad.base[1] + lad.rungs * 0.3 + 0.9
        low_y = lad.base[1] + 0.3
        rise = np.gradient(sh[ka:kb].mean(axis=1)[:, 1], self.dt) if kb - ka > 1 else np.zeros(1)
        reach = 0.97 * self.arm_len
        state = ["free", "free"]
        grip_y = [0.0, 0.0]
        grip_start = [ka, ka]
        move: list = [None, None]               # (k0, k1, p0, p1, kind) of a moving hand
        target = np.zeros((kb - ka, 2, 3))

        def free_pose(s, k):
            return sh[k, s] + UP * (-0.9 * self.arm_len) + rb.r[k] * (SIDES[s] * 0.08) + rb.f[k] * 0.08

        def wanted(s, k, up):
            y = sh[k, s, 1] + (RAIL_REGRIP_ABOVE if up else RAIL_REGRIP_BELOW)
            return float(np.clip(y, low_y, top_y))

        for s in (0, 1):                         # start gripping where a grip is in reach
            y = float(np.clip(sh[ka, s, 1] + 0.15, low_y, top_y))
            if np.linalg.norm(rail_point(lad, s, y) - sh[ka, s]) <= reach:
                state[s], grip_y[s] = "grip", y
        for k in range(ka, kb):
            up = rise[min(k - ka, len(rise) - 1)] >= 0.0
            for s in (0, 1):
                if state[s] == "move" and k >= move[s][1]:
                    state[s] = "free" if move[s][4] == "release" else "grip"
                    grip_start[s] = k
            for s in (0, 1):
                other_busy = state[1 - s] == "move"
                if state[s] == "grip":
                    # Look ahead by a move's time: re-grip before the grip leaves reach or the band.
                    k_ahead = min(k + n_move, kb - 1)
                    p = rail_point(lad, s, grip_y[s])
                    in_reach = max(np.linalg.norm(p - sh[k, s]), np.linalg.norm(p - sh[k_ahead, s])) <= reach
                    rel = grip_y[s] - sh[min(k + n_move // 2, kb - 1), s, 1]
                    if (not in_reach or rel < RAIL_BAND[0] or rel > RAIL_BAND[1]) and not other_busy and k + 3 < kb:
                        k1 = min(k + n_move, kb - 1)
                        new_y = wanted(s, k1, up)
                        q = rail_point(lad, s, new_y)
                        if abs(new_y - grip_y[s]) > 0.03 and np.linalg.norm(q - sh[k1, s]) <= reach:
                            self.rail_grips.append([s, grip_start[s], k, li])
                            move[s] = (k, k1, p, q, "regrip")
                            state[s], grip_y[s] = "move", new_y
                            self.log["rail_regrips"] += 1
                        elif not in_reach:
                            self.rail_grips.append([s, grip_start[s], k, li])
                            move[s] = (k, k1, p, None, "release")
                            state[s] = "move"
                elif state[s] == "free" and not other_busy and k + 3 < kb:
                    k1 = min(k + n_move, kb - 1)
                    y = wanted(s, k1, up)
                    q = rail_point(lad, s, y)
                    if np.linalg.norm(q - sh[k1, s]) <= reach and np.linalg.norm(q - sh[k, s]) <= reach:
                        move[s] = (k, k1, None, q, "grab")
                        state[s], grip_y[s] = "move", y
            for s in (0, 1):
                if state[s] == "grip":
                    target[k - ka, s] = rail_point(lad, s, grip_y[s])
                    self.note_unreachable(f"rails{li}", s, k, target[k - ka, s])
                elif state[s] == "free":
                    target[k - ka, s] = free_pose(s, k)
                else:
                    k0, k1, p0, p1, kind = move[s]
                    u = (k - k0 + 1) / max(k1 - k0, 1)
                    w = float(smoothstep(u))
                    p0 = free_pose(s, k) if p0 is None else p0
                    p1 = free_pose(s, k) if p1 is None else p1
                    away = fwd(lad.yaw) * (RAIL_AWAY_M * np.sin(np.pi * u)) if kind == "regrip" else 0.0
                    target[k - ka, s] = p0 + (p1 - p0) * w - away
        for s in (0, 1):
            if state[s] == "grip":
                self.rail_grips.append([s, grip_start[s], kb, li])
        self.rail_end[li] = max(self.rail_end.get(li, -np.inf), float(self.t[min(kb, self.N - 1)]))
        self.hand[ka:kb] = target
        # into the window over the rig's blend, out of it over the next item's blend
        for s in (0, 1):
            if ka > k_in:
                self.fade_offset(s, ka - 1, target[0, s] - self.hand_rig_at(ka - 1, s), k_in)
            if nxt is not None and kb < self.N:
                k_out = min(rb.ix(nxt.t1), self.N - 1)
                self.fade_offset(s, kb, target[-1, s] - self.hand_rig_at(kb, s), k_out)

    def hand_rig_at(self, k: int, s: int) -> np.ndarray:
        return self.rb.hand_target[min(k, self.N - 1), s]

    # --- stepladder ----------------------------------------------------------------------------
    def stepladder(self) -> None:
        """The carried stepladder (A-LADDER): it stands at each climb and is carried on edge in the
        right hand in between (scene.stepladder_motion's poses and timing). Here the hand holds one
        point of the upper rail: before a pick-up the hand reaches it on the standing ladder, then
        the ladder turns and folds about it; a set-down is the same backwards. The point is at mid
        length, or lower along the rail if mid length is out of reach of the standing worker."""
        from .scene import LADDER_SWAP_S

        rb, scene = self.rb, self.scene
        if not scene.dyn.get("stepladder"):
            return
        ups = sorted(scene.ladders, key=lambda lad: lad.t_up[0])
        if not ups:
            return
        t, N = self.t, self.N
        height = max(lad.rungs for lad in ups) * 0.3 + 0.7
        sets = np.array([lad.t_up[0] for lad in ups])
        picks = np.array([lad.t_down[1] if lad.t_down else np.inf for lad in ups])
        # The hands leave the rails at the end of the climb down; the pick-up waits for the reach.
        reach_from = picks - REACH_PHASE_S
        for j, lad in enumerate(ups):
            li = next(i for i, l in enumerate(scene.ladders) if l is lad)
            if li in self.rail_end and np.isfinite(picks[j]):
                reach_from[j] = max(reach_from[j], self.rail_end[li])
                picks[j] = max(picks[j], reach_from[j] + REACH_PHASE_S * 0.75)
        # Handhold along the rail per carry (carry j: from pick j to set j + 1).
        along = np.full(len(ups), 0.5)
        for j, lad in enumerate(ups):
            if not np.isfinite(picks[j]):
                continue
            k_pick = rb.ix(picks[j])
            for a in np.arange(0.5, 0.139, -0.02):
                h = base_handhold(lad.base, lad.yaw, stepladder_grip_local(height, a), height)
                if self.reachable(1, k_pick, h, margin=0.95):
                    along[j] = a
                    break
        weight = np.zeros(N)
        stand = np.zeros(N, int)
        carry = np.zeros(N, int)
        for i, ti in enumerate(t):
            k = int(np.searchsorted(sets, ti, side="right")) - 1
            carry[i] = max(k, 0)
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
        yaw0 = np.array([ups[j].yaw for j in stand])
        turn = (heading_wrap(rb.heading - yaw0))
        # The hand first lifts the upright ladder clear, then turns and folds it (set-down: the
        # reverse), so the rails do not sweep through the body.
        rotation = smoothstep(np.clip((weight - LADDER_LIFT_FIRST) / (1.0 - LADDER_LIFT_FIRST), 0.0, 1.0))
        yaw = yaw0 + turn * rotation
        opened = 1.0 - rotation
        base = np.array([ups[j].base for j in stand])
        local = [stepladder_grip_local(height, a) for a in along]
        # right hand: reach to the standing handhold, then carry the ladder by it
        before = self.hand[:, 1].copy()              # the hand path so far (rig, rails)
        hand = before.copy()
        for j in range(len(ups)):
            for edge, sign, jl in ((picks[j], 1, j), (sets[j], -1, j - 1)):
                if not np.isfinite(edge) or jl < 0:
                    continue
                h_stand = base_handhold(ups[j].base, ups[j].yaw, local[jl], height)
                k_e = rb.ix(edge)
                k_r = rb.ix(reach_from[j]) if sign > 0 else rb.ix(edge + REACH_PHASE_S)
                lo, hi = (k_r, k_e) if sign > 0 else (k_e, k_r)
                for k in range(lo, min(hi + 1, N)):
                    u = smoothstep((t[k] - t[k_r]) / (t[k_e] - t[k_r])) if t[k_e] != t[k_r] else 1.0
                    hand[k] = around_shoulder(rb.shoulders[k, 1], before[k], h_stand, float(u))
                    if k <= k_e if sign > 0 else k >= k_e:
                        self.note_unreachable("stepladder", 1, k, hand[k])
        for k in np.flatnonzero((weight > 0.0) & (weight < 1.0)):
            j = stand[k]
            h_stand = base_handhold(ups[j].base, ups[j].yaw, local[carry[k]], height)
            hand[k] = around_shoulder(rb.shoulders[k, 1], h_stand, rb.hand_target[k, 1], float(weight[k]))
            self.note_unreachable("stepladder", 1, k, hand[k])
        held = weight > 0.0
        offset = np.array([handhold_offset(o, y, local[c], height) for o, y, c in zip(opened, yaw, carry, strict=True)])
        hand = self.clear_ladder(hand, held, offset, opened, yaw, height)
        self.ladder_attach = (held, offset)      # the carried ladder hangs from the drawn hand (attach_ladder)
        out = np.zeros((N, 5))
        out[:, :3] = np.where(held[:, None], hand - offset, base)
        out[:, 3], out[:, 4] = np.unwrap(yaw), opened
        self.hand[:, 1] = hand
        self.held["stepladder"] = out
        self.stepladder_along = [[int(rb.ix(picks[j])) if np.isfinite(picks[j]) else N, round(float(along[j]), 3)]
                                 for j in range(len(ups))]
        self.log["stepladder_handhold_along"] = [round(float(a), 2) for a in along]

    # --- hammer --------------------------------------------------------------------------------
    def hammer(self) -> None:
        """Shorten hammer strokes whose hand would move faster than HAND_SPEED_MAX (relative to the
        shoulder): each stroke between two strikes is scaled about the impact point."""
        rb = self.rb
        step_max = HAND_SPEED_MAX * self.dt
        for i, it in self.items("hammer"):
            nxt = self.next_item(i)
            k0, k1 = rb.ix(it.t1), rb.ix(nxt.t0) if nxt is not None else self.N
            strikes = it.info.get("strikes", np.array([]))
            if len(strikes) < 2 or k1 - k0 < 3:
                continue
            impact, _ = rb._hammer_points(slice(k0, k1), bool(it.info.get("overhead")))
            rel = self.hand[k0:k1, 1] - impact
            from_sh = impact - rb.shoulders[k0:k1, 1]           # impact point relative to the shoulder
            ks = sorted({rb.ix(x) - k0 for x in strikes if k0 <= rb.ix(x) < k1})
            for a, b in zip([0] + ks, ks + [k1 - k0 - 1], strict=True):
                if b - a < 2:
                    continue
                for _ in range(3):                      # the stroke [a, b] ends on the strike frame b
                    path = rel[a:b + 1] + from_sh[a:b + 1]
                    peak = float(np.linalg.norm(np.diff(path, axis=0), axis=1).max())
                    if peak <= step_max:
                        break
                    scale = np.full(b - a - 1, step_max / peak)
                    if a == 0:                          # before the first strike: ease in from the rig's pose
                        scale = 1.0 + (scale - 1.0) * smoothstep(np.arange(1, b - a) / (b - a))
                    rel[a + 1:b] *= scale[:, None]
                    self.log["hammer_strokes_shortened"] += 1
            self.hand[k0:k1, 1] = impact + rel

    def clear_ladder(self, hand, held, offset, opened, yaw, height: float) -> np.ndarray:
        """Move the right hand, and the stepladder hanging from it, out to the right and forward just
        enough that the ladder's rails clear the drawn legs and torso (it turns and folds about the
        hand while being picked up or set down), within arm reach; smoothed like the sack."""
        rb, ctx = self.rb, self.ctx
        ks = np.flatnonzero(held & self.active)
        if not len(ks):
            return hand
        J = np.zeros((len(ks), 18, 3))
        J[:, 0], J[:, 1] = rb.hips[ks, 0], rb.hips[ks, 1]
        J[:, 2:4], J[:, 4:6] = self.legs["knee"][ks], self.legs["ankle"][ks]
        s_, w_ = body_scale(ctx.worker)
        lc, lr, _ = leg_spheres(J, s_, w_)
        tc, tr = torso_spheres(ctx, rb.P[ks], rb.frames[ks])
        cen, rad = np.concatenate([lc, tc], axis=1), np.concatenate([lr, tr])
        right_dir, fwd_dir = right(rb.heading[ks]), fwd(rb.heading[ks])
        grid = [(a, b) for a in np.arange(0.0, 0.361, 0.03) for b in np.arange(0.0, 0.241, 0.03)]
        grid.sort(key=lambda ab: np.hypot(*ab))
        need = np.zeros((len(ks), 2))
        for i, k in enumerate(ks):
            rails = ladder_rails(opened[k], yaw[k], height)
            for a, b in grid:
                shift = right_dir[i] * a + fwd_dir[i] * b
                origin = hand[k] + shift - offset[k]
                depth = max(float((RAIL_R + rad - segment_distance(cen[i], origin + p0, origin + p1)).max())
                            for p0, p1 in rails)
                if depth <= -0.005 and (self.reachable(1, k, hand[k] + shift) or (a, b) == grid[0]):
                    need[i] = (a, b)
                    break
            else:
                need[i] = grid[-1]
                self.log["ladder_clear_failed"] = self.log.get("ladder_clear_failed", 0) + 1
        full = np.zeros((self.N, 2))
        full[ks] = need
        a_env, b_env = envelope(full[:, 0], 0.015), envelope(full[:, 1], 0.015)        # Zero where the ladder stands (the hand is then on its standing handhold), ramping in.
        dist = np.full(self.N, np.inf)
        free = np.flatnonzero(~held)
        if len(free):
            idx = np.arange(self.N)
            pos = np.clip(np.searchsorted(free, idx), 0, len(free) - 1)
            prev = np.clip(pos - 1, 0, len(free) - 1)
            dist = np.minimum(np.abs(free[pos] - idx), np.abs(free[prev] - idx)).astype(float)
        cap = dist * 0.015
        a_env, b_env = np.minimum(a_env, cap), np.minimum(b_env, cap)
        out = hand + right(rb.heading) * a_env[:, None] + fwd(rb.heading) * b_env[:, None]
        self.log["ladder_shift_max_m"] = round(float(np.hypot(a_env, b_env).max()), 3)
        return out

    def attach_ladder(self, drawn_hand: np.ndarray) -> None:
        """Pose the held stepladder from the drawn right hand (after IK), so it never lags the hand."""
        if "stepladder" in self.held and getattr(self, "ladder_attach", None) is not None:
            held, offset = self.ladder_attach
            self.held["stepladder"][held, :3] = drawn_hand[held] - offset[held]

    # --- output --------------------------------------------------------------------------------
    def solve(self, weight: np.ndarray) -> dict:
        rb = self.rb
        d = rb.dims
        target = rb.hand_target + (self.hand - rb.hand_target) * weight[:, None, None]
        moved = np.abs(target - rb.hand_target).sum(axis=-1) > 1e-9
        # Edited paths arc around the shoulder as the rig's own do (rig._clear_of_shoulder).
        target = np.where(moved[..., None], rb._clear_of_shoulder(target), target)
        limited = rb.shoulders + speed_limited(target - rb.shoulders, HAND_STEP_MAX)
        # Speed-limit wherever this pass or the rig's own path would pop (active frames only).
        target = np.where((self.active[:, None] | moved)[..., None], limited, target)
        elbow, hand = two_bone(rb.shoulders, target, rb.arm_pole, d.upper_arm, d.forearm, continuous=True)
        return {"elbow": elbow, "hand": hand, "hand_target": target}


RAIL_R = 0.03                  # stepladder rail as a capsule of this radius (world.js: 0.035 x 0.08 box)


def ladder_rails(opened: float, yaw: float, height: float) -> list:
    """The stepladder's two front rails as segments [(p0, p1), ...] relative to its drawn origin."""
    from .contact_model import _front_chain, _ry

    length = height / np.cos(np.arctan2(1, 4))
    R, off = _front_chain(opened, height)
    Y = _ry(yaw)
    return [(Y @ (off + R @ np.array([x, 0.0, 0.0])), Y @ (off + R @ np.array([x, length, 0.0]))) for x in (-0.24, 0.24)]


def segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance (S,) of points (S, 3) from the segment a-b."""
    ab = b - a
    u = np.clip((points - a) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(points - (a + np.outer(u, ab)), axis=1)


def around_shoulder(shoulder: np.ndarray, a: np.ndarray, b: np.ndarray, u: float) -> np.ndarray:
    """Hand path from a to b (fraction u) that swings around the shoulder: its direction from the
    shoulder turns from a's to b's while its distance changes linearly, so it never passes closer
    to the shoulder than its ends (a straight path could cross the folded-elbow zone)."""
    ra, rb_ = a - shoulder, b - shoulder
    la, lb = float(np.linalg.norm(ra)), float(np.linalg.norm(rb_))
    if la < 1e-6 or lb < 1e-6:
        return a + (b - a) * u
    da, db = ra / la, rb_ / lb
    angle = float(np.arccos(np.clip(da @ db, -1.0, 1.0)))
    if angle < 1e-6:
        direction = da
    else:
        direction = (np.sin((1 - u) * angle) * da + np.sin(u * angle) * db) / np.sin(angle)
    return shoulder + direction * (la + (lb - la) * u)


def swing_forward(shoulder: np.ndarray, hand: np.ndarray, forward: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Hand positions (len(angles), 3) after swinging the shoulder-to-hand vector forward by each
    angle (rad) in the vertical plane along `forward` (a straight arm swinging about the shoulder);
    its sideways part and its length stay the same."""
    v = hand - shoulder
    side = unit(np.cross(UP, forward))
    lateral = side * float(v @ side)
    a_fwd, a_up = float(v @ forward), float(v @ UP)
    radius = np.hypot(a_fwd, a_up)
    phi = np.arctan2(a_fwd, -a_up) + np.asarray(angles, float)     # 0 = hanging straight down
    return shoulder + lateral + np.outer(np.sin(phi) * radius, forward) - np.outer(np.cos(phi) * radius, UP)


def heading_wrap(a):
    return (np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi


def envelope(need: np.ndarray, rate: float) -> np.ndarray:
    """Smallest curve >= need whose slope is at most `rate` per frame (both directions)."""
    out = np.asarray(need, float).copy()
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1] - rate)
    for i in range(len(out) - 2, -1, -1):
        out[i] = max(out[i], out[i + 1] - rate)
    return out


# =============================================================================================
# Entry point
# =============================================================================================
def apply_contact(ctx, scene, frames: np.ndarray, rig) -> dict:
    """Contact pass on the rig (in place, render only): re-posed legs and arms in rig.joints,
    rig.hand_l / hand_r, held object poses in rig.held (scene.dynamics uses them) and the IK targets
    in rig.targets. Returns meta.contact."""
    rb = rig.builder
    log = {"early_lifts": 0, "catch_steps": 0, "settle_steps": 0, "kneel_frames": 0, "rail_regrips": 0,
           "hammer_strokes_shortened": 0, "sack_clear_failed": 0, "sacks": 0}
    active = active_frames(ctx, scene, rig, frames)
    weight = edge_weight(active, rb.dt)
    legs = LegPass(ctx, scene, rig, active, log)
    legs.run()
    leg_out = legs.solve(weight)
    arms = ArmPass(ctx, scene, rig, active, leg_out, log)
    arms.sacks()
    arms.rails()
    arms.stepladder()
    arms.hammer()
    arm_out = arms.solve(weight)
    arms.attach_ladder(arm_out["hand"][:, 1])
    J = rig.joints
    J[:, 2:4], J[:, 4:6], J[:, 6:8], J[:, 8:10] = leg_out["knee"], leg_out["ankle"], leg_out["heel"], leg_out["toe"]
    J[:, 12:14], J[:, 14:16] = arm_out["elbow"], arm_out["hand"]
    rig.hand_l, rig.hand_r = J[:, 14].copy(), J[:, 15].copy()
    rig.targets = {"ankle": leg_out["ankle_target"], "hand": arm_out["hand_target"], "planted": leg_out["planted"]}
    rig.held = arms.held
    unreachable = {key: sorted(set(v)) for key, v in arms.unreachable.items()}
    return {"enabled": True, "version": VERSION, "active_frames": int(active.sum()), **log,
            "unreachable": unreachable, "unreachable_frames": int(sum(len(v) for v in unreachable.values())),
            "rail_grips": arms.rail_grips, "sack_channels": arms.sack_channels,
            "stepladder_along": arms.stepladder_along}
