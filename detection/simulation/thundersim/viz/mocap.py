# File: mocap.py
# Purpose: Optional limb motion from real motion capture for the 3D viewer. Clips listed in
#          motionlib/manifest.json replace the procedural limb targets of rig.py for walking,
#          hammering and kneeling floor work. The simulated spine is never touched.
# Contributors: Unassigned
"""Mocap limbs (off by default).

Switch: `THUNDERSIM_MOCAP=1`, the viewer URL `?mocap=1`, or run_payload(mocap=True). The clips come
from motionlib/manifest.json (built by motionlib/tools/manifest.py); this module only reads that
file and the clips it names (numpy arrays, no pickles) and checks each clip's sha256.

How a clip drives the limbs. Every clip is time-warped piecewise-linearly between paired anchors:
simulator events on one side, the same events in the clip on the other.

  walk, carry walk  simulated heel strikes (the extremes of motions.walk's pelvis yaw, which are its
                    bob minima: rig._stride_strikes) <-> the clip's heel strikes. The rig keeps
                    planning and planting the feet (no skate, falls interrupt as before); the clip
                    sets each step's toe-off time, the swing path, the foot roll in swing and stance
                    (heel strike, heel rise), the knee direction and the arm swing. Foot placement
                    and yaw stay the rig's (the clips' foot yaw relative to their smoothed heading
                    is not consistent between clips).
  hammer            the simulator's tool impulses (Builder.impulses times) <-> the clip's right-wrist
                    deceleration peaks. The striking hand reaches the rig's impact point exactly at
                    each impulse; between strikes it follows the clip's stroke.
  kneel_work        the simulator's work impulses <-> the clip's wrist deceleration peaks, at
                    close to real speed (slope 0.85-1.15); arms only.

Where two warp segments do not follow each other in the clip (it loops back), each sampled signal
carries the jump as an offset that fades out over JUNCTION_BLEND_S. Without a listed clip, or for
any other activity, the rig's procedural modes are used unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .geom import UP, smoothstep

MOTIONLIB = Path(__file__).resolve().parents[2] / "motionlib"
MANIFEST = MOTIONLIB / "manifest.json"
ENV_SWITCH = "THUNDERSIM_MOCAP"

JUNCTION_BLEND_S = 0.15        # a jump where the clip loops back fades out over this time [design]
HEADING_SMOOTH_S = 1.1         # heading frame averaged over about one stride, as the clips (cliplib) [design]
WINDOW_FADE_S = 0.3            # clip foot roll and knee direction fade in/out at a walk's first/last strike [design]
LIFT_SCALE_RANGE = (0.5, 1.0)  # a step slower than the clip lifts the foot less (speed ratio, clamped) [design]
STROKE_SCALE_RANGE = (0.6, 1.0)  # a faster hammer rate than the clip's gets a shorter stroke [design]
KNEEL_SLOPE = (0.85, 1.15)     # kneeling work plays at nearly real speed while matching wrist events [design]
KNEEL_MAX_GAP_S = 3.0          # kneeling-work warp segments are at most this long [design]
HAND_ABOVE_FLOOR_M = 0.03      # hands of kneeling work stay this far above the support [design]
KNEEL_ARM_REACH = 0.9          # kneeling-work hands reach down at most this fraction of the arm [design]
SWING_CLEARANCE_M = 0.015      # a swinging foot clears the floor by at least this in mid-swing [src: Winter 1992,
                               # minimum toe clearance about 1.3 cm in normal walking]
CLEARANCE_EASE = 0.1           # ... easing in and out over the first and last 10 % of the swing [design]
HIP_EXTENSION_LIMIT_DEG = -26.0  # clip-shaped steps keep the thigh within this extension [design: 4 deg inside
                               # the curator's -30 deg gate]
ARM_REACH_MAX = 0.995          # clip hand targets stay within this fraction of the rig arm's reach [design]


def requested(flag: bool | None = None) -> bool:
    """Mocap on? An explicit flag wins; otherwise THUNDERSIM_MOCAP=1."""
    if flag is not None:
        return bool(flag)
    return os.environ.get(ENV_SWITCH, "0").strip() == "1"


def manifest_bytes() -> bytes:
    """The manifest as stored (part of the viewer's run cache key); b"" when there is none."""
    return MANIFEST.read_bytes() if MANIFEST.exists() else b""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_index(key: str, n: int) -> int:
    """Deterministic choice in range(n) from a text key (Python's hash() changes per process)."""
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) % max(n, 1)


def heading_frame(heading: np.ndarray) -> np.ndarray:
    """(n,3,3) rotation, columns right(h), up, fwd(h): world = R @ (x right, y up, z forward)."""
    s, c = np.sin(heading), np.cos(heading)
    zero, one = np.zeros_like(heading), np.ones_like(heading)
    return np.stack([np.stack([c, zero, -s], -1), np.stack([zero, one, zero], -1), np.stack([-s, zero, -c], -1)], -1)


def combine_pitch(needed: float, clip: float) -> float:
    """Foot roll on a planted foot: the clip's roll (heel strike toes-up, heel rise before toe-off)
    unless the hip needs more roll the same way to keep the leg in reach (rig._planted), or a roll
    the other way, which wins."""
    if needed == 0.0:
        return clip
    if needed < 0.0 and clip < 0.0:
        return min(needed, clip)
    if needed > 0.0 and clip > 0.0:
        return max(needed, clip)
    return needed


# --- clips -----------------------------------------------------------------------------------------
class ClipData:
    """One stage-S4 clip (motionlib/clips/<clip>.npz) and its manifest entry."""

    def __init__(self, entry: dict, path: Path):
        data = np.load(path, allow_pickle=False)
        self.entry = entry
        self.name = entry["clip"]
        self.fs = float(data["fs"])
        self.arrays = {key: data[key].astype(float) for key in data.files
                       if data[key].dtype.kind == "f" and data[key].ndim >= 1}
        self.n = len(self.arrays["hand_shoulder"])
        self.duration = (self.n - 1) / self.fs

    def at(self, name: str, tau) -> np.ndarray:
        """Field `name` at clip times tau (s, array), linearly interpolated, clamped to the clip."""
        arr = self.arrays[name]
        x = np.clip(np.asarray(tau, float) * self.fs, 0.0, self.n - 1.0)
        i0 = np.floor(x).astype(int)
        i1 = np.minimum(i0 + 1, self.n - 1)
        w = (x - i0).reshape(x.shape + (1,) * (arr.ndim - 1))
        return arr[i0] * (1.0 - w) + arr[i1] * w

    def unit_at(self, name: str, tau) -> np.ndarray:
        v = self.at(name, tau)
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


class Library:
    """The manifest's clips by role. Clips whose file is missing or whose sha256 differs are left out."""

    def __init__(self, manifest: dict):
        self.by_role: dict[str, list] = {}
        self.problems: list[str] = []
        self._cache: dict[str, ClipData] = {}
        for entry in manifest.get("clips", []):
            path = MOTIONLIB / entry["file"]
            if not path.exists():
                self.problems.append(f"{entry['clip']}: file missing")
                continue
            for role in entry.get("roles", {}):
                self.by_role.setdefault(role, []).append(entry)

    @classmethod
    def load(cls) -> Library | None:
        if not MANIFEST.exists():
            return None
        return cls(json.loads(MANIFEST.read_text()))

    def clip(self, entry: dict) -> ClipData | None:
        name = entry["clip"]
        if name not in self._cache:
            path = MOTIONLIB / entry["file"]
            if sha256_file(path) != entry["sha256"]:
                self.problems.append(f"{name}: sha256 differs from the manifest; not used")
                self._cache[name] = None
            else:
                self._cache[name] = ClipData(entry, path)
        return self._cache[name]

    def entries(self, role: str) -> list:
        return self.by_role.get(role, [])


# --- time warp -------------------------------------------------------------------------------------
class Warp:
    """Piecewise-linear map from simulator time t to clip time tau.

    Segment j maps simulator [s[j], s[j+1]] onto clip [a[j], b[j]]. Before s[0] and after s[-1] the
    clip runs on at real speed, clamped to [lo, hi]. Where a segment does not start where the
    previous one ended in the clip, sample() adds the jump as an offset fading over JUNCTION_BLEND_S."""

    def __init__(self, s, a, b, lo: float, hi: float):
        self.s = np.asarray(s, float)
        self.a = np.asarray(a, float)
        self.b = np.asarray(b, float)
        self.lo, self.hi = float(lo), float(hi)
        self.junctions = [j for j in range(1, len(self.a)) if abs(self.a[j] - self.b[j - 1]) > 1e-6]

    def locate(self, t) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(segment index, fraction u in the segment, clip time) for simulator times t. The index is -1
        before s[0] and len(a) after s[-1]."""
        t = np.asarray(t, float)
        k = len(self.a)
        j = np.clip(np.searchsorted(self.s, t, side="right") - 1, -1, k)
        jj = np.clip(j, 0, k - 1)
        span = np.maximum(self.s[jj + 1] - self.s[jj], 1e-9)
        u = np.clip((t - self.s[jj]) / span, 0.0, 1.0)
        tau = self.a[jj] + (self.b[jj] - self.a[jj]) * u
        tau = np.where(j < 0, self.a[0] - (self.s[0] - t), tau)
        tau = np.where(j >= k, self.b[-1] + (t - self.s[-1]), tau)
        return j, u, np.clip(tau, self.lo, self.hi)

    def sample(self, fn, t) -> np.ndarray:
        """fn(clip times) at simulator times t, with the junction jumps faded out."""
        t = np.asarray(t, float)
        _, _, tau = self.locate(t)
        out = fn(tau)
        for j in self.junctions:
            jump = fn(np.array([self.b[j - 1]]))[0] - fn(np.array([self.a[j]]))[0]
            fade = 1.0 - smoothstep((t - self.s[j]) / JUNCTION_BLEND_S)
            fade = np.where(t >= self.s[j], fade, 0.0)
            out = out + jump * fade.reshape(fade.shape + (1,) * (out.ndim - 1))
        return out


# --- walking ---------------------------------------------------------------------------------------
class WalkPlan:
    """One walk call driven by one clip: the warp, and per-frame clip signals over the rig frames
    k0..k1 (ankle relative to the hip and its lift, foot pitch, knee direction)."""

    def __init__(self, rb, call, clip: ClipData, role_name: str, role: dict, times: np.ndarray, sides: np.ndarray,
                 key: str):
        self.call, self.clip, self.role_name, self.role = call, clip, role_name, role
        d = rb.dims
        steps = role["steps"]
        # Simulator segments: a virtual strike one step before the first, so the first planned
        # step (landing on the first strike) has a swing to follow; each segment's landing side.
        s = np.concatenate([[times[0] - (times[1] - times[0])], times])
        landing = list(sides)
        chosen = []
        c = stable_index(key, len(steps))
        for side in landing:
            for _ in range(len(steps)):
                if steps[c % len(steps)][2] == int(side):
                    break
                c += 1
            chosen.append(c % len(steps))
            c = c % len(steps) + 1
        self.steps = [steps[i] for i in chosen]
        self.s = s
        self.sides = np.array(landing, int)
        a = [st[0] for st in self.steps]
        b = [st[1] for st in self.steps]
        lo = max(min(a) - 1.0, 0.0)
        self.warp = Warp(s, a, b, lo, min(max(b) + 1.0, clip.duration))
        self.k0 = rb.ix(s[0] - 0.4)
        self.k1 = min(rb.ix(s[-1] + 0.6) + 1, rb.N)
        t = rb.t[self.k0:self.k1]
        n = len(t)
        self.t = t
        # Heading frame of the simulated body, smoothed like the clips' (one stride).
        self.R = heading_frame(rb.heading_smooth[self.k0:self.k1])
        self.ankle_local = np.zeros((n, 2, 3))
        self.lift = np.zeros((n, 2))
        self.pitch = np.zeros((n, 2))
        self.knee_dir = np.zeros((n, 2, 3))
        for side in (0, 1):
            self.ankle_local[:, side] = self.warp.sample(lambda x: clip.at("ankle_hip", x)[:, side], t) * d.leg
            self.lift[:, side] = self.warp.sample(lambda x: clip.at("ankle_lift", x)[:, side], t) * d.leg
            self.pitch[:, side] = self.warp.sample(lambda x: clip.at("foot_pitch", x)[:, side], t)
            local = self.warp.sample(lambda x: clip.unit_at("knee_pole", x)[:, side], t)
            self.knee_dir[:, side] = np.einsum("nij,nj->ni", self.R, local)
        # Clip foot roll and knee direction apply from the first to the last strike, faded at the ends.
        self.weight = smoothstep((t - (s[1] - WINDOW_FADE_S)) / WINDOW_FADE_S) * \
            (1.0 - smoothstep((t - s[-1]) / WINDOW_FADE_S))
        # Pelvis speed of each clip step at the rig's size, for the lift and arm-swing scales.
        self.clip_speed = np.array([st[4] for st in self.steps]) * d.leg
        self.f_off = np.array([st[3] for st in self.steps])
        speed = np.linalg.norm(rb.vel[self.k0:self.k1], axis=1)
        self.speed_ratio = np.clip(speed / max(float(np.median(self.clip_speed)), 1e-3), 0.0, 1.1)
        # Footsteps (side, lift-off, touchdown) and where each foot stands in the clip: from a
        # touchdown to that foot's next lift-off. The clip's stance roll applies only there.
        self.step_times = []
        for j in range(len(self.steps)):
            t0, t1 = float(s[j]), float(s[j + 1])
            self.step_times.append((int(self.sides[j]), t0 + self.f_off[j] * (t1 - t0), t1))
        self.stance = np.zeros((n, 2), bool)
        for side in (0, 1):
            mine = [(lift, land) for sd, lift, land in self.step_times if sd == side]
            starts = [-np.inf] + [land for _, land in mine]
            ends = [lift for lift, _ in mine] + [np.inf]
            for a, b in zip(starts, ends, strict=True):
                self.stance[(t >= a - 1e-9) & (t <= b + 1e-9), side] = True

    def frame(self, k: int) -> int:
        return int(np.clip(k - self.k0, 0, len(self.t) - 1))

    def raw_ankle(self, rb, k: int, side: int, lift_scale: float) -> np.ndarray:
        """Clip ankle at frame k: horizontal from the simulated hip and the clip's hip-relative
        ankle, vertical from the clip's ankle lift (scaled); rb.hips include the simulated bob."""
        i = self.frame(k)
        horiz = rb.hips[k, side] + self.R[i] @ self.ankle_local[i, side]
        return np.array([horiz[0], self.lift[i, side] * lift_scale, horiz[2]])

    def rig_steps(self, rb, step_cls, place_fraction: float) -> list:
        """rig.Step objects: lift-off at the clip's toe-off, touchdown on each simulated strike."""
        out = []
        for j, (side, t_lift, t1) in enumerate(self.step_times):
            t_stance = (t1 - float(self.s[j])) * (1.0 + self.f_off[j])
            v_sim = float(np.linalg.norm(rb.vel[rb.ix(t1)]))
            scale = float(np.clip(v_sim / max(self.clip_speed[j], 1e-3), *LIFT_SCALE_RANGE))
            shape = SwingShape(self, side, scale)
            out.append(step_cls(t_lift, t1, side, t1 + place_fraction * t_stance, 0.0, shape))
        return out


class SwingShape:
    """The clip's path of one swinging foot, bent so that it leaves from the rig's lift-off pose
    and lands on the rig's planted pose (the endpoint differences blend in over the swing)."""

    def __init__(self, plan: WalkPlan, side: int, lift_scale: float):
        self.plan, self.side, self.lift_scale = plan, side, lift_scale

    def __deepcopy__(self, memo):          # rig._stand_snap deep-copies the feet; the plan is shared
        return self

    def touchdown(self, rb, contact, yaw1: float, k1: int, a1, pitch1: float):
        """Planted pose at touchdown with the clip's heel-strike roll (toes up)."""
        i = self.plan.frame(k1)
        pitch = combine_pitch(pitch1, float(self.plan.pitch[i, self.side] * self.plan.weight[i]))
        if pitch == pitch1:
            return a1, pitch1
        return rb._ankle_at(contact, yaw1, pitch), pitch

    def pose(self, rb, sw, k: int, t: float):
        plan, side = self.plan, self.side
        k0, k1 = rb.ix(sw.t0), rb.ix(sw.t1)
        raw = plan.raw_ankle(rb, k, side, self.lift_scale)
        raw0 = plan.raw_ankle(rb, k0, side, self.lift_scale)
        raw1 = plan.raw_ankle(rb, k1, side, self.lift_scale)
        u = float(np.clip((t - sw.t0) / max(sw.t1 - sw.t0, 1e-6), 0.0, 1.0))
        w = float(smoothstep(u))
        pos = raw + (1.0 - w) * (sw.a0 - raw0) + w * (sw.a1 - raw1) + UP * (sw.lift * np.sin(np.pi * u))
        p, p0, p1 = (float(plan.pitch[plan.frame(x), side]) for x in (k, k0, k1))
        pitch = p + (1.0 - w) * (sw.pitch0 - p0) + w * (sw.pitch1 - p1)
        yaw = sw.yaw0 + float((sw.yaw1 - sw.yaw0 + np.pi) % (2 * np.pi) - np.pi) * w
        pos[1] += self.clearance_lift(rb, sw, pos, yaw, pitch, u)
        pos = self.hip_extension_guard(rb, k, side, pos)
        return pos, yaw, pitch

    @staticmethod
    def hip_extension_guard(rb, k: int, side: int, ankle: np.ndarray) -> np.ndarray:
        """Swing ankle turned forward about the hip (pelvis lateral axis) where the thigh would
        extend past HIP_EXTENSION_LIMIT_DEG relative to the simulated pelvis. Just after toe-off the
        clip's foot rises behind the body while the rig's thigh would still point back."""
        short = HIP_EXTENSION_LIMIT_DEG - thigh_flexion_deg(rb, k, side, ankle)
        if short <= 0.0:
            return ankle
        hip = rb.hips[k, side]
        pelvis = rb.M[k, 0]
        a = np.radians(short)
        turn = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(a), -np.sin(a)], [0.0, np.sin(a), np.cos(a)]])
        return hip + pelvis @ (turn @ (pelvis.T @ (ankle - hip)))


    @staticmethod
    def clearance_lift(rb, sw, ankle, yaw: float, pitch: float, u: float) -> float:
        """Lift (m) that keeps the rig foot's lowest point (heel, ball or toe tip) at least
        SWING_CLEARANCE_M above the landing support in mid-swing, easing to zero at lift-off and
        touchdown. The clips' heel joints are derived points that skim the floor for ~0.1 s before
        heel strike; on the rig that would read as the heel grazing the ground before it lands."""
        if sw.contact is None:
            return 0.0
        need = SWING_CLEARANCE_M * float(smoothstep(u / CLEARANCE_EASE)) * (1.0 - float(smoothstep((u - 1.0 + CLEARANCE_EASE) / CLEARANCE_EASE)))
        if need <= 0.0:
            return 0.0
        d = rb.dims
        f_u, f_n = rb._foot_axes(yaw, pitch)
        sole = ankle - f_n * d.ankle_h
        bend = max(-sw.pitch0, 0.0) * float(1.0 - smoothstep(u / 0.4))       # rig._toe_bend_of in swing
        ball = sole + f_u * d.ball
        toe = ball + (f_u * np.cos(bend) + f_n * np.sin(bend)) * (d.toe - d.ball)
        lowest = min(float((sole - f_u * d.heel)[1]), float(ball[1]), float(toe[1]))
        return max(float(sw.contact[1]) + need - lowest, 0.0)


def thigh_flexion_deg(rb, k: int, side: int, ankle: np.ndarray) -> float:
    """Sagittal thigh flexion (deg, extension < 0) relative to the simulated pelvis segment for an
    ankle at `ankle`, with the knee bent forward as rig.two_bone puts it: the hip-ankle line's
    flexion plus the thigh's angle in front of that line."""
    d = rb.dims
    pelvis = rb.M[k, 0]                                       # columns: lateral, up the spine, posterior
    v = pelvis.T @ (ankle - rb.hips[k, side])
    dist = float(np.linalg.norm(v))
    line = float(np.degrees(np.arctan2(-v[2], -v[1])))
    reach = float(np.clip(dist, abs(d.thigh - d.shank) + 1e-4, d.leg * 0.9995))
    cos_a = (d.thigh ** 2 + reach ** 2 - d.shank ** 2) / (2 * d.thigh * reach)
    return line + float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


# --- the session: one payload ----------------------------------------------------------------------
class Session:
    """Mocap for one run. The rig asks it for walk steps, stance roll, knee poles and arm targets;
    every clip it uses is logged in `uses` (payload meta.mocap)."""

    def __init__(self, library: Library, key: str, only: dict | None = None):
        self.library = library
        self.key = key
        self.only = only or {}
        self.uses: list[dict] = []
        self.walk_plans: dict[int, WalkPlan] = {}
        self.kneel_plans: list = []
        self.rb = None
        self.pitch = None
        self.pole_w = None
        self.pole_dir = None

    @classmethod
    def create(cls, enabled: bool, key: str, only: dict | None = None, manifest: dict | None = None) -> Session | None:
        """manifest: a manifest dict instead of motionlib/manifest.json (the S6 gate tool)."""
        if not enabled:
            return None
        library = Library(manifest) if manifest is not None else Library.load()
        if library is None:
            return None
        return cls(library, key, only)

    def attach(self, rb) -> None:
        """Called once by the rig builder: per-frame arrays for the stance roll and knee poles."""
        self.rb = rb
        n = rb.N
        k = max(int(round(HEADING_SMOOTH_S / rb.dt)), 1)
        h = np.unwrap(rb.heading)
        pad = np.concatenate([np.full(k // 2, h[0]), h, np.full(k - 1 - k // 2, h[-1])])
        rb.heading_smooth = np.convolve(pad, np.ones(k) / k, mode="valid")[:n]
        self.pitch = np.full((n, 2), np.nan)
        self.pole_w = np.zeros(n)
        self.pole_dir = np.zeros((n, 2, 3))

    def pick(self, role: str, want_speed: float | None = None) -> tuple[ClipData | None, dict | None]:
        """The clip for a role: the one named in `only`, else for walking the clip whose speed is
        closest to the simulated walk, else a stable choice from the run key."""
        entries = self.library.entries(role)
        if role in self.only:
            entries = [e for e in entries if e["clip"] == self.only[role]]
        if not entries:
            return None, None
        if want_speed is not None and "speed_leg_per_s" in entries[0]["roles"][role]:
            leg = self.rb.dims.leg
            entry = min(entries, key=lambda e: (abs(np.log(e["roles"][role]["speed_leg_per_s"] * leg / max(want_speed, 0.1))),
                                                e["clip"]))
        else:
            entry = entries[stable_index(f"{self.key}:{role}", len(entries))]
        clip = self.library.clip(entry)
        return (clip, entry["roles"][role]) if clip is not None else (None, None)

    def log(self, role: str, limbs: str, clip: ClipData, t0: float, t1: float) -> None:
        self.uses.append({"role": role, "limbs": limbs, "clip": clip.name, "t0": round(float(t0), 3),
                          "t1": round(float(t1), 3), "licence_class": clip.entry["licence_class"],
                          "credit": clip.entry["credit"]})

    # --- legs --------------------------------------------------------------------------------------
    def walk_steps(self, rb, call, times, sides, step_cls, place_fraction: float) -> list | None:
        """Clip-shaped footsteps for a walk call, or None (the rig plans its own)."""
        if call.name != "walk" or len(times) < 3:
            return None
        carrying = "carry" in rb.ctx.rec.ancestors(call)
        speed = float(np.median(np.linalg.norm(rb.vel[rb.ix(times[0]):rb.ix(times[-1]) + 1], axis=1)))
        clip, role = self.pick("carry_walk", speed) if carrying else (None, None)
        role_name = "carry_walk"
        if clip is None:
            clip, role = self.pick("walk", speed)
            role_name = "walk"
        if clip is None:
            return None
        plan = WalkPlan(rb, call, clip, role_name, role, np.asarray(times, float), np.asarray(sides, int),
                        f"{self.key}:{call.index}")
        self.walk_plans[call.index] = plan
        sl = slice(plan.k0, plan.k1)
        for side in (0, 1):
            self.pitch[sl, side] = np.where(plan.stance[:, side], plan.pitch[:, side] * plan.weight, np.nan)
        self.pole_w[sl] = np.maximum(self.pole_w[sl], plan.weight)
        self.pole_dir[sl] = plan.knee_dir
        steps = plan.rig_steps(rb, step_cls, place_fraction)
        self.log(role_name, "legs", clip, plan.s[1], plan.s[-1])
        return steps

    def stance_pitch(self, k: int, side: int, needed: float) -> float | None:
        """Roll of a planted foot at frame k: the clip's combined with what the hip needs, or None
        outside the clip's stance of that foot."""
        value = self.pitch[k, side]
        return None if np.isnan(value) else combine_pitch(needed, float(value))

    def overextended(self, rb, foot, side: int, k: int) -> bool:
        """A planted foot whose thigh would extend past HIP_EXTENSION_LIMIT_DEG at frame k, inside
        a clip-driven walk: the rig then takes that foot's next (clip-shaped) step early."""
        if self.pole_w[k] <= 0.0:
            return False
        ankle, _ = rb._planted(foot.contact, foot.yaw, rb.hips[k, side], foot.pivot)
        return thigh_flexion_deg(rb, k, side, ankle) < HIP_EXTENSION_LIMIT_DEG

    def knee_poles(self, rb, sl: slice, poles: np.ndarray) -> np.ndarray:
        w = self.pole_w[sl]
        if not (w > 0).any():
            return poles
        clip_poles = rb.hips[sl] + self.pole_dir[sl]
        return poles + (clip_poles - poles) * w[:, None, None]

    # --- arms --------------------------------------------------------------------------------------
    def add_arm_items(self, rb, tl, c) -> bool:
        """Arm items for call c from a clip; False when no clip applies (the rig adds its own)."""
        rec = rb.ctx.rec
        if c.name == "walk":
            plan = self.walk_plans.get(c.index)
            if plan is None or "carry" in rec.ancestors(c):
                return False
            t_first, t_last = float(plan.s[1]), float(plan.s[-1])
            tl.add(c.t0, c.t0 + 0.5, "idle")
            tl.add(t_first - 0.2, t_first + 0.3, "clip", key=("clipwalk", c.index), role="walk", plan=plan, as_mode="idle")
            tl.add(t_last - 0.2, t_last + 0.4, "idle", key=("clipwalk_end", c.index))
            self.log(plan.role_name, "arms", plan.clip, t_first, t_last)
            return True
        if c.name == "hammer":
            # Overhead and on-ladder hammering stay procedural (the hammer clips strike at waist height).
            on_ladder = any(lad.t_up[1] <= c.t0 + 1e-6 and (lad.t_down is None or c.t0 < lad.t_down[0])
                            for lad in rb.scene.ladders if lad.kind in ("ladder", "stepladder"))
            strikes = np.array(sorted(s for x in rec.find("impulses", inside=c) for s in x.args.get("times_abs", [])))
            if c.args.get("overhead") or on_ladder or len(strikes) < 2:
                return False
            clip, role = self.pick("hammer")
            if clip is None:
                return False
            plan = HammerPlan(clip, role, strikes, f"{self.key}:{c.index}")
            tl.add(c.t0, c.t0 + 0.5, "clip", key=("cliphammer", c.index), role="hammer", plan=plan, as_mode="hammer")
            tl.add(c.t1, c.t1 + 0.5, "idle")
            self.log("hammer", "arms", clip, strikes[0], strikes[-1])
            return True
        if c.name == "kneel_work":
            clip, role = self.pick("kneel_work")
            if clip is None:
                return False
            impulses = np.array(sorted(s for x in rec.find("impulses", inside=c) for s in x.args.get("times_abs", [])))
            plan = KneelPlan(clip, role, c.t0 + 0.8, c.t1, impulses, f"{self.key}:{c.index}")
            self.kneel_plans.append(plan)
            tl.add(c.t0 + 0.8, c.t0 + 2.0, "clip", key=("clipfloor", c.index), role="kneel_work", plan=plan, as_mode="floor")
            tl.add(c.t1 - 1.5, c.t1, "idle")
            self.log("kneel_work", "arms", clip, c.t0 + 0.8, c.t1)
            return True
        return False

    def arm_target(self, rb, item, sl: slice):
        role = item.info["role"]
        plan = item.info["plan"]
        if role == "walk":
            return walk_arms(rb, plan, sl)
        if role == "hammer":
            return plan.hands(rb, sl)
        if role == "kneel_work":
            return plan.hands(rb, sl)
        raise ValueError(role)

    def meta(self) -> dict:
        return {"uses": self.uses, "problems": list(self.library.problems)}


def within_reach(rel: np.ndarray) -> np.ndarray:
    """Shoulder-to-hand vectors in arm lengths, shortened to ARM_REACH_MAX where a clip's hand point
    lies a little beyond its own arm length (the hand point is the wrist-to-hand-end midpoint)."""
    length = np.linalg.norm(rel, axis=-1, keepdims=True)
    return rel * np.minimum(1.0, ARM_REACH_MAX / np.maximum(length, 1e-9))


def upper_frame_to_world(rb, sl: slice, local: np.ndarray) -> np.ndarray:
    """Vectors (n,2,3) in the simulated upper-thoracic segment frame (rig M[:, 3]) -> world."""
    return np.einsum("nij,nsj->nsi", rb.M[sl, 3], local)


def walk_arms(rb, plan: WalkPlan, sl: slice):
    """Arm swing of the clip in the simulated upper-thoracic frame, scaled with the walking speed
    around the clip's mean arm pose (smaller swing while speeding up and slowing down)."""
    d = rb.dims
    t = rb.t[sl]
    clip = plan.clip
    rel = plan.warp.sample(lambda x: clip.at("hand_shoulder", x), t)
    span = np.arange(plan.warp.a.min(), plan.warp.b.max(), 0.01)
    mean = clip.at("hand_shoulder", span).mean(axis=0)
    k = np.clip(np.arange(sl.start, sl.stop) - plan.k0, 0, len(plan.t) - 1)
    amp = plan.speed_ratio[k][:, None, None]
    rel = within_reach(mean[None] + (rel - mean[None]) * amp)
    hand = rb.shoulders[sl] + upper_frame_to_world(rb, sl, rel * d.arm)
    pole_local = plan.warp.sample(lambda x: clip.unit_at("elbow_pole", x), t)
    pole = rb.shoulders[sl] + upper_frame_to_world(rb, sl, pole_local) * 0.5
    return hand, pole


# --- hammering -------------------------------------------------------------------------------------
class HammerPlan:
    """Hammer strokes of one clip mapped onto the simulated strikes (impulse times)."""

    def __init__(self, clip: ClipData, role: dict, strikes: np.ndarray, key: str):
        self.clip = clip
        self.strikes = strikes
        clip_strikes = role["strikes"]
        runs = role["runs"]
        # Clip strokes: consecutive strikes inside one run.
        strokes = [(clip_strikes[i], clip_strikes[i + 1]) for r0, r1 in runs for i in range(r0, r1 - 1)]
        start = stable_index(key, len(strokes))
        chosen = [strokes[(start + j) % len(strokes)] for j in range(len(strikes) - 1)]
        self.warp = Warp(strikes, [c[0] for c in chosen], [c[1] for c in chosen], role["window"][0], role["window"][1])
        sim_dt = np.diff(strikes)
        clip_dt = np.array([c[1] - c[0] for c in chosen])
        self.scale = np.clip(sim_dt / np.maximum(clip_dt, 1e-3), *STROKE_SCALE_RANGE)
        window = np.arange(role["window"][0], role["window"][1], 0.01)
        self.left_median = np.median(clip.at("hand_task", window)[:, 0], axis=0)

    def hands(self, rb, sl: slice):
        d = rb.dims
        t = rb.t[sl]
        clip = self.clip
        impact, _ = rb._hammer_points(sl, overhead=False)
        R = heading_frame(rb.heading_smooth[sl])
        j, u, tau = self.warp.locate(t)
        k = len(self.warp.a)
        jj = np.clip(j, 0, k - 1)
        scale = self.scale[jj][:, None] * d.leg
        right_now = np.einsum("nij,nj->ni", R, clip.at("hand_task", tau)[:, 1]) * scale
        at_a = np.einsum("nij,nj->ni", R, clip.at("hand_task", self.warp.a[jj])[:, 1]) * scale
        at_b = np.einsum("nij,nj->ni", R, clip.at("hand_task", self.warp.b[jj])[:, 1]) * scale
        # Before the first and after the last strike the correction holds that strike's value.
        uu = np.where(j < 0, 0.0, np.where(j >= k, 1.0, u))[:, None]
        right_hand = impact + right_now - (at_a * (1.0 - uu) + at_b * uu)
        left_local = self.warp.sample(lambda x: clip.at("hand_task", x)[:, 0], t) - self.left_median[None]
        rr = rb.r[sl]
        fr = rb.f[sl]
        left_base = impact - rr * 0.12 + UP * -0.03 - fr * 0.02
        left_hand = left_base + np.einsum("nij,nj->ni", R, left_local) * d.leg
        hand = np.stack([left_hand, right_hand], axis=1)
        pole_local = self.warp.sample(lambda x: clip.unit_at("elbow_pole", x), t)
        pole = rb.shoulders[sl] + upper_frame_to_world(rb, sl, pole_local) * 0.5
        return hand, pole


# --- kneeling floor work ---------------------------------------------------------------------------
class KneelPlan:
    """Kneeling work arms: the clip at nearly real speed, its wrist events matched to the simulated
    work impulses where one lies within KNEEL_SLOPE; it loops back to the window start at the end."""

    def __init__(self, clip: ClipData, role: dict, t0: float, t1: float, impulses: np.ndarray, key: str):
        self.clip = clip
        w0, w1 = role["window"]
        peaks = np.asarray(role["peaks"], float)
        events = [float(x) for x in impulses if t0 + 0.5 < x < t1 - 0.5]
        # Filler anchors (no event) keep every segment under KNEEL_MAX_GAP_S, so a long gap is not slowed down.
        anchors_t, is_event = [t0], [False]
        for tb in events + [t1]:
            ta = anchors_t[-1]
            pieces = int(np.ceil((tb - ta) / KNEEL_MAX_GAP_S))
            for q in range(1, pieces):
                anchors_t.append(ta + (tb - ta) * q / pieces)
                is_event.append(False)
            anchors_t.append(tb)
            is_event.append(tb != t1)
        tau = w0 + (w1 - w0 - 2.0) * (stable_index(key, 1000) / 1000.0)
        s, a, b = [anchors_t[0]], [], []
        self.matched = []
        for i in range(1, len(anchors_t)):
            ta, tb = anchors_t[i - 1], anchors_t[i]
            dt = tb - ta
            if tau + KNEEL_SLOPE[0] * dt > w1:          # out of clip: loop back to its start
                tau = w0
            lo, hi = tau + KNEEL_SLOPE[0] * dt, min(tau + KNEEL_SLOPE[1] * dt, w1)
            near = peaks[(peaks >= lo) & (peaks <= hi)]
            if is_event[i] and len(near):
                end = float(near[np.argmin(np.abs(near - (tau + dt)))])
                self.matched.append((tb, end))
            else:
                end = min(tau + dt, w1)
            a.append(tau)
            b.append(end)
            s.append(tb)
            tau = end
        self.warp = Warp(s, a, b, w0, w1)
        self.events = len(events)
        self.deck_unreached = 0.0          # share of frames where the clip's hand height was out of reach

    def hands(self, rb, sl: slice):
        """Hands relative to the shoulders in the heading frame (gravity-aligned), not the trunk
        frame: the simulated kneeling trunk is often more upright than the roofers', and hands
        placed in its frame would reach forward into the air instead of down to the work."""
        d = rb.dims
        t = rb.t[sl]
        clip = self.clip
        rel = within_reach(self.warp.sample(lambda x: clip.at("hand_shoulder_hf", x), t))
        R = heading_frame(rb.heading_smooth[sl])
        hand = rb.shoulders[sl] + np.einsum("nij,nsj->nsi", R, rel * d.arm)
        ground = rb.ground_under_hips[sl].mean(axis=1)
        if "hand_deck_h" in clip.arrays:
            # Height above the kneeling surface from the clip (the roofers' hands work on the deck),
            # raised where the simulated shoulder is too high for the arm to reach that low with the
            # elbow still a little bent (KNEEL_ARM_REACH); the clip's horizontal reach is kept.
            want = ground[:, None] + self.warp.sample(lambda x: clip.at("hand_deck_h", x), t) * d.arm
            v = hand - rb.shoulders[sl]
            horiz = np.hypot(v[..., 0], v[..., 2])
            lowest = rb.shoulders[sl][..., 1] - np.sqrt(np.maximum((KNEEL_ARM_REACH * d.arm) ** 2 - horiz ** 2, 0.0))
            self.deck_unreached = float((want < lowest - 0.01).mean())
            hand[..., 1] = np.maximum(want, lowest)
        hand[..., 1] = np.maximum(hand[..., 1], ground[:, None] + HAND_ABOVE_FLOOR_M)
        pole_local = self.warp.sample(lambda x: clip.unit_at("elbow_pole", x), t)
        pole = rb.shoulders[sl] + upper_frame_to_world(rb, sl, pole_local) * 0.5
        return hand, pole
