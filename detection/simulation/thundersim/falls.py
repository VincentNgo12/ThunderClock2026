# File: falls.py
# Purpose: Physics of falls and near-falls: pendulum loss of balance over the stance foot,
#          slips (foot sliding out), ballistic flight about the centre of mass, body-first
#          (Mode A) and feet-first (Mode B) impacts, harness arrest and suspension, safety
#          nets, stairs, slides, collapse and post-fall behaviour.
# Contributors: Unassigned
"""Fall physics, integrated at 1 kHz and committed to the Builder as ordinary segments.

Sources (tags in site.py): free fall t = sqrt(2h/g); Mode A contact with effective mass
22-33 kg (0.35 x body mass), pelvis stiffness 39-55 kN/m in series with the surface;
same-level hip impact speeds near 3.4 +/- 0.5 m/s (hip-protector consensus) emerge from the
pendulum + knee-collapse model and are reported per run; Mode B peak ~(h/d + 1) g over the
stopping distance d; harness free fall <= 1.8 m, stop within 1.07 m, force <= 8 kN (OSHA
1926.502(d)(16)). Rotation, rebound, lying postures and timings follow SIM_SPEC.md 1 [design].

Velocity is kept continuous across every phase change: a step in velocity would become a
fake acceleration spike after differentiation. During flight the body rotates about its
centre of mass, so off-centre IMUs read the omega^2 r term instead of exactly 0 g.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .motions import Builder, G, smooth_pulse
from .site import (BODY_DAMPING_RATIO, BODY_STIFFNESS_N_PER_M, HARNESS_MAX_DECEL_DISTANCE_M,
                   HARNESS_MAX_FORCE_N, SURFACES, available_friction, effective_mass,
                   series_stiffness)
from .spine import PHASE_CODE, root_rotation

DIRECTIONS = {"forward": 0.0, "backward": np.pi, "right": np.pi / 2, "left": -np.pi / 2}
UP = np.array([0.0, 1.0, 0.0])
# Knee-collapse fraction of the pendulum length during same-level falls, U(0, COLLAPSE_MAX).
# Calibrated so hip contact speeds centre near the 3.4 +/- 0.5 m/s consensus (see tests).
COLLAPSE_MAX = 0.25
# Same-level fall descent (realism scorecard, round 2). TRIP_KICK: extra rotation (rad/s) a trip
# adds; PIVOT_MOMENTUM_KEEP: fraction of the walking rotation about the stance foot carried into the
# fall (the rest is absorbed smoothly over the kick time, like a recovery step); PIVOT_RESIST:
# protective rotational resistance (1/s, ramped in over 0.1 s). Neither applies to slips.
# Fitted (realism/fit.py "descent_keep", "descent_resist", "descent2"; fit seeds; Erciyes forward/side/
# recover falls and UNIVRFall walking trips with video-labelled onset/impact): resist 2.0 matches the
# labelled onset->impact time (sim 0.63 s vs real 0.62 s) and brings trip hip speeds to 2.7-2.8 m/s
# (literature 2.1-3.0 for real falls); keep 0.5 beat 1.0 and tied 0.2. Originals: keep 1.0, resist 0.0.
TRIP_KICK = (0.6, 1.4)
PIVOT_MOMENTUM_KEEP = 0.5
PIVOT_RESIST = 2.0
# Upper-trunk curl after the trunk lands (realism scorecard, iteration 1). Lying on the ground,
# the segments conform to the floor: in Erciyes falls (UCI 455, waist + chest, 25 Hz) the
# chest-minus-waist inclination 2-5 s after lying falls has median about -1 deg and a robust SD
# of about 10 deg (forward, backward, side and syncope falls). The descent curl therefore relaxes
# over FLATTEN_S to a residual drawn from N(CURL_REST_DEG) instead of being kept while lying.
CURL_REST_DEG = (-1.0, 10.0)
FLATTEN_S = 0.4
# Body-contact energy loss (realism scorecard, iteration 2). None = the original linear
# spring-damper; a value = Hunt-Crossley damping F = k x (1 + 1.5 alpha v_in), which has no force
# step at touchdown and a restitution of about 1 - alpha v. ROT_CONTACT_ZETA damps the trunk's
# rotational landing (ramped in over ROT_CONTACT_RAMP_S so the chest sees no torque step);
# TRUNK_TOPPLE_GAIN scales the gravity topple of the trunk about the landed pelvis.
# Fitted (realism/fit.py grids "contact".."contact4"; fit seeds; Erciyes forward, backward and side
# falls): zeta 1.2 with a 50 ms ramp-in, topple gain 0.5 (zeta at the edge of its grid; gain 0.25
# fitted the impact features better but left backward falls not lying). Hunt-Crossley alpha 0.6 s/m
# fitted the mats but is NOT used (CONTACT_RESTITUTION below wins): it raised hard-surface site peaks
# 20-50 %. Originals: alpha None, zeta 0.4, ramp 0.0, gain 1.0. See realism/REPORT.md.
CONTACT_HC_ALPHA: float | None = None
# Restitution-based hysteretic contact (iteration 2b). When set, it replaces both damping models:
# loading is a pure spring F = k x (so the loading peak is the undamped v sqrt(k m), as before on
# hard surfaces), unloading follows a stiffer line k/e^2 from the deepest compression, so only e^2
# of the stored energy comes back (no bounce); that line is damped (zeta 0.7) so it cannot ring. Hunt-Crossley's velocity term, fitted on ~3 m/s mat
# impacts, raised hard-surface peaks 20-50 % when extrapolated to 4-10 m/s (see REPORT.md).
CONTACT_RESTITUTION: float | None = 0.25  # fitted ("contact4"; flat for 0.1-0.4)
SET_RECOVERY_S = 1.5  # [design] soft-tissue recovery of the contact's permanent set
ROT_CONTACT_ZETA = 1.2
ROT_CONTACT_RAMP_S = 0.05
TRUNK_TOPPLE_GAIN = 0.5
# Collapse (syncope) descent (realism scorecard, iteration 3). None = the original kinematically
# guided sag (pelvis eased to the floor, no impact). A value = the legs still carry this fraction
# of body weight while the pelvis drops, and it lands through the ordinary body contact.
# SYNC_BUCKLE_S is the knee-buckle duration range before the drop.
# Fitted to Erciyes 919 (acted syncope onto a mat; realism/fit.py "sync"/"sync2", fit seeds): the
# objective is flat for support 0.6-0.8 (weakly identified), 0.7 is its middle; buckle 0.4-0.8 s
# beat 0.2-0.4 s and the original 1-2 s. Originals: None (guided sag) and (1.0, 2.0).
SYNC_SUPPORT: float | None = 0.7
SYNC_BUCKLE_S = (0.4, 0.8)


def side_rng(rng: np.random.Generator) -> np.random.Generator:
    """A generator seeded from rng's current state WITHOUT advancing it, so adding a random draw
    here leaves every later draw of the run unchanged."""
    state = rng.bit_generator.state["state"]
    return np.random.default_rng([int(state["state"]) % 2**63, int(state["inc"]) % 2**63])


def mj(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return 10 * u**3 - 15 * u**4 + 6 * u**5


def bump(k: int) -> np.ndarray:
    """sin^2 bump 0 -> 1 -> 0 with zero slope at both ends."""
    return np.sin(np.pi * np.linspace(0, 1, k)) ** 2


def rot_matrix(heading: float, pitch: float, roll: float) -> np.ndarray:
    """Same convention as spine.root_rotation, for one sample."""
    ch, sh = np.cos(heading), np.sin(heading)
    ry = np.array([[ch, 0, sh], [0, 1, 0], [-sh, 0, ch]])
    cp, sp = np.cos(-pitch), np.sin(-pitch)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    cr, sr = np.cos(-roll), np.sin(-roll)
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return ry @ rx @ rz


@dataclass
class ImpactParams:
    m_eff: float
    k_eff: float
    zeta: float
    friction: float
    rot_omega: float  # rotational contact natural frequency when the trunk lands, rad/s


class Faller:
    """Integrates one fall. Records the centre of mass before contact and the pelvis anchor
    after it, then commits a single Builder segment."""

    def __init__(self, b: Builder, direction: str | float, phase: str = "fall_descent"):
        self.b, self.rng, self.dt = b, b.rng, b.dt
        self.H = b.worker.height_m
        self.M = b.worker.total_mass
        self.alpha = DIRECTIONS[direction] if isinstance(direction, str) else float(direction)
        self.heading = b.pose.heading
        u = np.array([np.cos(self.alpha), np.sin(self.alpha)])
        tilt = np.array([b.pose.pitch, b.pose.roll])
        self.phi = float(tilt @ u)                          # inclination toward the fall direction
        self.perp = float(tilt @ np.array([-u[1], u[0]]))   # perpendicular tilt
        self.dphi = 0.0
        self.dperp = 0.0
        fwd = np.array([-np.sin(self.heading), 0, -np.cos(self.heading)])
        right = np.array([np.cos(self.heading), 0, -np.sin(self.heading)])
        self.d = np.cos(self.alpha) * fwd + np.sin(self.alpha) * right
        self.r_p = b.worker.com_to_pelvis()               # COM -> pelvis anchor, body frame (Worker)
        self.flex0, self.lat0, self.twist0 = b.pose.flex.copy(), b.pose.lat.copy(), b.pose.twist.copy()
        self.upper_gain = float(self.rng.uniform(1.0, 1.4))  # SIM_SPEC 1.3 segment lag [design]
        self.yaw_rate = float(self.rng.uniform(-0.75, 0.75))
        self.com = b.pose.pos - rot_matrix(self.heading, *self.tilt()) @ self.r_p
        self.vcom = b.vel.copy()
        self.pelvis: np.ndarray | None = None
        self.vpel: np.ndarray | None = None
        self.rec: list[tuple] = []
        self.phase = PHASE_CODE[phase]
        self.events: list[tuple[float, str, dict]] = []
        self.contact_speed: float | None = None
        self.flatten: tuple[int, float, float] | None = None  # (record index, curl at landing, rest curl)

    # --- helpers -------------------------------------------------------------------------
    def tilt(self, phi: float | None = None, perp: float | None = None) -> tuple[float, float]:
        phi = self.phi if phi is None else phi
        perp = self.perp if perp is None else perp
        return (phi * np.cos(self.alpha) - perp * np.sin(self.alpha),
                phi * np.sin(self.alpha) + perp * np.cos(self.alpha))

    def t_now(self) -> float:
        return self.b.t + len(self.rec) * self.dt

    def mark(self, name: str, **info) -> None:
        self.events.append((self.t_now(), name, info))

    def pelvis_now(self) -> np.ndarray:
        if self.pelvis is not None:
            return self.pelvis
        return self.com + rot_matrix(self.heading, *self.tilt()) @ self.r_p

    def record(self) -> None:
        curl = float(np.clip((self.upper_gain - 1.0) * abs(self.phi), 0, np.deg2rad(40)))
        if self.flatten is not None:
            i0, c0, rest = self.flatten
            curl = c0 + (rest - c0) * mj((len(self.rec) - i0) * self.dt / FLATTEN_S)
        mode = "com" if self.pelvis is None else "pel"
        xyz = self.com if self.pelvis is None else self.pelvis
        self.rec.append((mode, xyz.copy(), self.heading, self.phi, self.perp, self.phase, curl))

    def start_flatten(self) -> None:
        """The trunk has landed: relax the descent curl toward a resting value (see CURL_REST_DEG)."""
        if self.flatten is not None or not self.rec:
            return
        mean, sd = CURL_REST_DEG
        rest = float(np.deg2rad(np.clip(side_rng(self.rng).normal(mean, sd), -25.0, 25.0)))
        self.flatten = (len(self.rec), float(self.rec[-1][6]), rest)

    def to_pelvis_mode(self) -> None:
        """Switch from centre-of-mass to pelvis-anchor integration with continuous velocity."""
        if self.pelvis is not None:
            return
        if len(self.rec) >= 2 and self.rec[-2][0] == "com":
            _, com_prev, h_prev, phi_prev, perp_prev, _, _ = self.rec[-2]
            p_prev = com_prev + rot_matrix(h_prev, *self.tilt(phi_prev, perp_prev)) @ self.r_p
            self.pelvis = self.pelvis_now()
            self.vpel = (self.pelvis - p_prev) / self.dt
        else:
            self.pelvis = self.pelvis_now()
            self.vpel = self.vcom.copy()

    def to_com_mode(self) -> None:
        if self.pelvis is None:
            return
        self.com = self.pelvis - rot_matrix(self.heading, *self.tilt()) @ self.r_p
        self.vcom = self.vpel.copy()
        self.pelvis, self.vpel = None, None

    def impact_params(self, surface_name: str) -> ImpactParams:
        surface = SURFACES[surface_name]
        k_surface = self.rng.uniform(*surface.stiffness_range)
        k_eff = series_stiffness(self.rng.uniform(*BODY_STIFFNESS_N_PER_M), k_surface)
        zeta = max(self.rng.uniform(*BODY_DAMPING_RATIO), surface.damping_ratio)
        friction = available_friction(surface_name, self.b.site.condition)
        rot_omega = 25.0 * np.sqrt(series_stiffness(47e3, k_surface) / 47e3)  # [design]
        return ImpactParams(effective_mass(self.b.worker.mass_kg), k_eff, zeta, friction, rot_omega)

    # --- phases --------------------------------------------------------------------------
    def pivot(self, max_s: float, release_phi: float | None = None, collapse: float = 0.3, collapse_s: float = 0.5,
              contact_height: float | None = None, kick: float = 0.0, kick_s: float = 0.15,
              slip_du: float = 0.0, slip_s: float = 0.15, friction: float = 0.5, keep: float = 1.0,
              resist: float = 0.0) -> str:
        """Inverted pendulum over the stance foot with knee collapse (l shrinks).

        kick: extra angular velocity (rad/s) delivered smoothly over kick_s (push, stumble from rest).
        slip_du: the foot slides out (against the fall direction) by this extra speed over slip_s,
        then slows by kinetic friction; the pivot acceleration drives the rotation.
        Returns 'release' (support lost at release_phi), 'contact' or 'timeout'.
        """
        # theta: angle of the foot -> COM line (the leg pendulum); phi: trunk inclination, which
        # follows theta with a lag. In walking the legs rotate over the stance foot (theta rate
        # = v / l) while the trunk stays upright, so phi starts with the trunk's own rate (~0).
        theta = self.phi
        height = self.com[1] - self.b.ground_y
        l0 = height / np.cos(theta)
        pivot = self.com - l0 * (np.sin(theta) * self.d + np.cos(theta) * UP)
        v = self.vcom
        v_d = float(v @ self.d)
        drift = v - v_d * self.d - float(v @ UP) * UP   # sideways velocity, decays
        dtheta = v_d / (l0 * np.cos(theta))
        kick = kick + (keep - 1.0) * dtheta              # momentum absorbed smoothly (no velocity step)
        offset = 0.0                                    # trunk lean relative to the leg line
        track_w, track_z = 12.0, 0.7                    # trunk follows the leg line (~0.2 s) [design]
        v_pivot = 0.0                                   # pivot speed along d (a slipping foot is negative)
        prev_l = l0
        for i in range(int(max_s / self.dt)):
            t = i * self.dt
            length = l0 * (1 - collapse * mj(t / collapse_s))
            dl = (length - prev_l) / self.dt
            prev_l = length
            if t < slip_s and slip_du > 0:
                a_pivot = -slip_du * np.pi / (2 * slip_s) * np.sin(np.pi * t / slip_s)
            elif v_pivot < 0:
                a_pivot = friction * G
            else:
                a_pivot, v_pivot = 0.0, 0.0
            v_pivot += a_pivot * self.dt
            kick_acc = kick * np.pi / (2 * kick_s) * np.sin(np.pi * t / kick_s) if t < kick_s else 0.0
            ddtheta = (G * np.sin(theta) - a_pivot * np.cos(theta)) / length - 2 * dl * dtheta / length + kick_acc
            if resist > 0:
                ddtheta -= resist * min(t / 0.1, 1.0) * dtheta
            dtheta += ddtheta * self.dt
            theta += dtheta * self.dt
            ddphi = track_w**2 * (theta + offset - self.phi) - 2 * track_z * track_w * self.dphi
            self.dphi += ddphi * self.dt
            self.phi += self.dphi * self.dt
            self._perp_step(0.0)
            self.heading += self.yaw_rate * 0.3 * self.dt
            pivot = pivot + (v_pivot * self.d + drift) * self.dt
            drift = drift * (1 - self.dt / 0.4)
            new_com = pivot + length * (np.sin(theta) * self.d + np.cos(theta) * UP)
            self.vcom = (new_com - self.com) / self.dt
            self.com = new_com
            self.record()
            if release_phi is not None and abs(theta) >= release_phi:
                return "release"
            if contact_height is not None and self.pelvis_now()[1] <= self.b.ground_y + contact_height:
                return "contact"
            if abs(theta) > np.deg2rad(80):
                return "contact"
        return "timeout"

    def flight(self, landing_y: float, max_s: float = 4.0, tumble_acc: float = 0.0, harness: dict | None = None,
               pelvis_stop: float | None = None) -> str:
        """Ballistic centre of mass; spin changes smoothly by tumble_acc (rad/s^2 over 0.2 s).

        Ends when the lowest trunk point reaches landing_y + 0.12 m (body-first), when the pelvis
        reaches pelvis_stop (feet-first touchdown), or when a harness lanyard goes taut."""
        self.to_com_mode()
        start_y = self.com[1]
        self.mark("free_fall_start")
        for i in range(int(max_s / self.dt)):
            t = i * self.dt
            self.vcom = self.vcom + np.array([0, -G, 0]) * self.dt
            self.com = self.com + self.vcom * self.dt
            if t < 0.2:
                self.dphi += tumble_acc * self.dt
            self.dphi *= 1 - 0.1 * self.dt
            self.phi += self.dphi * self.dt
            self._perp_step(self.perp, omega=0.0)
            self.heading += self.yaw_rate * self.dt
            self.record()
            if harness is not None and start_y - self.com[1] >= harness["free_fall_m"]:
                self.mark("free_fall_end", reason="lanyard_taut")
                return "harness"
            pelvis_y = self.pelvis_now()[1]
            if pelvis_stop is not None:
                if pelvis_y <= pelvis_stop:
                    self.mark("free_fall_end", reason="feet")
                    return "feet"
                continue
            upper_y = (self.com + rot_matrix(self.heading, *self.tilt()) @ np.array([0.0, 0.30 * self.H, 0.10]))[1]
            pelvis_floor = landing_y + lying_rest_height(self.alpha, self.phi)
            if pelvis_y <= pelvis_floor + 0.02 or upper_y <= landing_y + 0.12:
                self.mark("free_fall_end", reason="ground")
                return "ground"
        return "timeout"

    def body_contact(self, surface: str, rest_height: float, phi_rest: float, duration: float,
                     soften: float = 1.0, guided_s: float | None = None, support: float = 0.0) -> float:
        """Mode A: pelvis spring-damper on the surface plus trunk rotation about the pelvis down
        to phi_rest, which ends in a stiff damped rotational contact (back/side/front landing).

        soften < 1 models a hands-first arrest (lower stiffness and damping energy).
        guided_s: lower the pelvis on a smooth path instead of letting it drop (collapse).
        support: fraction of gravity still carried by the legs until the pelvis touches down.
        Returns the peak pelvis specific force from the contact, in g.
        """
        self.to_pelvis_mode()
        params = self.impact_params(surface)
        k_eff = params.k_eff * soften**2  # peak ~ v * sqrt(k/m): soften scales the peak linearly
        c = 2 * params.zeta * np.sqrt(k_eff * params.m_eff)
        floor = self.b.ground_y + rest_height
        rod = 0.30 * self.H
        target = phi_rest
        direction = np.sign(target - self.phi) or 1.0
        peak = 0.0
        y_start = self.pelvis[1]
        self.phase = PHASE_CODE["impact"]
        touched = False
        landed_at = None
        max_comp, t_max = 0.0, 0.0
        for i in range(int(duration / self.dt)):
            t = (i + 1) * self.dt
            compression = floor - self.pelvis[1]
            if guided_s is not None and t < guided_s:
                new_y = y_start + (floor - y_start) * mj(t / guided_s)
                self.vpel = np.array([self.vpel[0] * 0.995, (new_y - self.pelvis[1]) / self.dt, self.vpel[2] * 0.995])
            else:
                if compression <= 0:
                    force = 0.0
                elif CONTACT_RESTITUTION is not None:
                    if compression > max_comp:
                        max_comp, t_max = compression, t
                    if self.vpel[1] <= 0 and compression >= max_comp - 1e-9:
                        force = k_eff * compression                       # loading
                    else:
                        e2 = max(CONTACT_RESTITUTION, 0.02) ** 2
                        # unloading line k/e^2; its permanent set recovers like soft tissue
                        # (SET_RECOVERY_S), so a lying body does not stay sunk into the support
                        rest = max_comp * (1.0 - e2) * np.exp(-(t - t_max) / SET_RECOVERY_S)
                        k_u = k_eff / e2
                        c_u = 2 * 0.7 * np.sqrt(k_u * params.m_eff)  # damped unloading line: no ringing
                        force = min(max(k_u * (compression - rest) - c_u * self.vpel[1], 0.0), k_eff * compression)
                elif CONTACT_HC_ALPHA is None:
                    force = max(k_eff * compression - c * self.vpel[1], 0.0)
                else:
                    force = max(k_eff * compression * (1.0 - 1.5 * CONTACT_HC_ALPHA * self.vpel[1]), 0.0)
                if force > 0 and not touched:
                    touched = True
                    self.start_flatten()  # the trunk conforms to the floor from pelvis touchdown on
                    self.contact_speed = float(-self.vpel[1])
                    self.mark("impact", surface=surface, speed_ms=round(self.contact_speed, 2))
                peak = max(peak, force / params.m_eff / G)
                horizontal = self.vpel[[0, 2]]
                speed = np.linalg.norm(horizontal)
                if compression > -0.01 and speed > 1e-6:
                    horizontal = horizontal - horizontal / speed * min(params.friction * G * self.dt, speed)
                carried = support * G if not touched else 0.0
                self.vpel = np.array([horizontal[0], self.vpel[1] + (force / params.m_eff - G + carried) * self.dt, horizontal[1]])
            self.pelvis = self.pelvis + self.vpel * self.dt
            # Trunk: rotates about the pelvis under gravity until it lands, then a damped contact.
            if (self.phi - target) * direction < 0 and landed_at is None:
                ddphi = direction * TRUNK_TOPPLE_GAIN * (1.5 * G / rod * abs(np.sin(self.phi)) + 2.0)
            else:
                if landed_at is None:
                    landed_at = t
                self.start_flatten()
                w = params.rot_omega * soften
                ramp = min((t - landed_at) / ROT_CONTACT_RAMP_S, 1.0) if ROT_CONTACT_RAMP_S > 0 else 1.0
                zeta_rot = 0.4 + (ROT_CONTACT_ZETA - 0.4) * ramp
                ddphi = -w * w * (self.phi - target) - 2 * zeta_rot * w * self.dphi
            self.dphi += ddphi * self.dt
            self.phi += self.dphi * self.dt
            self._perp_step(0.0, omega=6.0)
            if t > 0.4:
                self.phase = PHASE_CODE["lying"]
            self.record()
        if guided_s is not None and not touched:
            self.mark("impact", surface=surface, guided=True)
        return peak

    def pelvis_push(self, dv: np.ndarray, duration: float = 0.12) -> None:
        """Smoothly change the pelvis velocity by dv (e.g. rolling on to the next stair)."""
        self.to_pelvis_mode()
        k = int(duration / self.dt)
        weights = np.sin(np.pi * np.arange(k) / k)
        weights /= weights.sum()
        for i in range(k):
            self.vpel = self.vpel + dv * weights[i] + np.array([0, -G, 0]) * self.dt
            self.pelvis = self.pelvis + self.vpel * self.dt
            self.record()

    def feet_first(self, stop_distance: float) -> float:
        """Mode B: constant deceleration over the knee/hip stopping distance. Returns peak g."""
        self.to_pelvis_mode()
        v0 = abs(min(self.vpel[1], -1e-3))
        decel = v0 * v0 / (2 * stop_distance)
        horizontal0 = self.vpel[[0, 2]].copy()
        self.phase = PHASE_CODE["impact"]
        self.mark("impact", mode="feet_first", speed_ms=round(v0, 2))
        ramp_k, i = int(0.01 / self.dt), 0
        while self.vpel[1] < 0:
            ramp = min((i + 1) / ramp_k, 1.0)  # 10 ms force rise
            vy = self.vpel[1] + (decel * ramp) * self.dt
            frac = max(-vy / v0, 0.0)
            self.vpel = np.array([horizontal0[0] * frac, vy, horizontal0[1] * frac])
            self.pelvis = self.pelvis + self.vpel * self.dt
            self._relax_tilt(0.0, 0.0, 6.0)
            self.record()
            i += 1
        return (decel + G) / G

    def harness_arrest(self, force_n: float, lanyard_m: float, hang_s: float = 30.0) -> float:
        """Shock-absorber arrest, then pendulum swing near upright. Returns peak specific force (g)."""
        self.to_com_mode()
        self.phase = PHASE_CODE["impact"]
        v0 = abs(self.vcom[1])
        # Absorber sized to stop within 1.07 m including the ~30 ms force rise (plus 10 % margin).
        usable = max(HARNESS_MAX_DECEL_DISTANCE_M - v0 * 0.03, 0.3)
        needed = 1.1 * self.M * (G + v0 * v0 / (2 * usable))
        force = float(np.clip(max(force_n, needed), 0, HARNESS_MAX_FORCE_N))
        self.mark("arrest_start", force_n=round(force))
        hang_phi = np.deg2rad(self.rng.uniform(15, 35))  # dorsal D-ring: trunk 15-35 deg forward [design]
        target_phi, target_perp = hang_phi * np.cos(self.alpha), -hang_phi * np.sin(self.alpha)
        start_y = self.com[1]
        i = 0
        while self.vcom[1] < 0:
            ramp = min((i + 1) * self.dt / 0.03, 1.0)  # ~30 ms force rise
            self.vcom = self.vcom + np.array([0, force * ramp / self.M - G, 0]) * self.dt
            self.vcom[[0, 2]] *= 0.995
            self.com = self.com + self.vcom * self.dt
            self._relax_tilt(target_phi, target_perp, 5.0)
            self.record()
            i += 1
        decel = start_y - self.com[1]
        self.mark("arrest_end", decel_distance_m=round(decel, 3), force_n=round(force))
        # Suspended: pendulum swing with period 2*pi*sqrt(L/g), lanyard bounce, slow spin.
        length = lanyard_m + decel
        omega = np.sqrt(G / length)
        v_h = self.vcom[[0, 2]].copy()
        bounce_amp, bounce_hz = self.rng.uniform(0.03, 0.06), self.rng.uniform(1.0, 2.0)
        tau = self.rng.uniform(5, 15)
        base = self.com.copy()
        self.phase = PHASE_CODE["suspended"]
        self.mark("suspended")
        for i in range(int(hang_s / self.dt)):
            t = (i + 1) * self.dt  # first hang sample is one step after the arrest (no duplicate)
            decay = np.exp(-t / tau)
            swing = (v_h / omega) * np.sin(omega * t) * decay
            # Bounce fades in with a minimum-jerk envelope: no position or velocity step.
            onset = mj(t / 0.3)
            y = base[1] + bounce_amp * onset * np.sin(2 * np.pi * bounce_hz * t) * np.exp(-t / 0.8) + float(swing @ swing) / (2 * length)
            self.com = np.array([base[0] + swing[0], y, base[2] + swing[1]])
            self.heading += 0.2 * np.sin(0.5 * t) * decay * self.dt
            self._relax_tilt(target_phi, target_perp, 4.0)
            self.record()
        self.vcom = np.zeros(3)
        return force / self.M / G

    def net_catch(self, net_y: float, duration: float = 4.0) -> float:
        """Whole body caught by a safety net: soft, bouncy contact, then lying in the sag."""
        self.to_com_mode()
        surface = SURFACES["safety_net"]
        k = self.rng.uniform(*surface.stiffness_range)
        c = 2 * surface.damping_ratio * np.sqrt(k * self.M)
        lie = np.deg2rad(self.rng.uniform(70, 100)) * (1 if self.phi >= 0 else -1)
        peak = 0.0
        self.phase = PHASE_CODE["impact"]
        touched = False
        for i in range(int(duration / self.dt)):
            compression = net_y + 0.15 - self.com[1]
            force = max(k * compression - c * self.vcom[1], 0.0) if compression > 0 else 0.0
            if force > 0 and not touched:
                touched = True
                self.mark("impact", surface="safety_net", speed_ms=round(float(-self.vcom[1]), 2))
            peak = max(peak, force / self.M / G)
            self.vcom = self.vcom + np.array([0, force / self.M - G, 0]) * self.dt
            self.vcom[[0, 2]] *= 1 - 2 * self.dt
            self.com = self.com + self.vcom * self.dt
            self._relax_tilt(lie, 0.0, 3.0)
            if i == int(1.5 / self.dt):
                self.phase = PHASE_CODE["lying"]
            self.record()
        self.b.ground_y = self.com[1] - 0.2
        return peak

    def slide(self, slope_deg: float, duration: float, friction: float) -> None:
        """Sliding down a roof or bank on the back: |specific force| = g cos(slope) sqrt(1 + mu^2)."""
        self.to_com_mode()
        slope = np.deg2rad(slope_deg)
        a = max(G * (np.sin(slope) - friction * np.cos(slope)), 0.3)
        down = self.d * np.cos(slope) + np.array([0, -np.sin(slope), 0])
        self.phase = PHASE_CODE["fall_descent"]
        self.mark("slide_start", slope_deg=round(slope_deg, 1))
        lie = np.deg2rad(self.rng.uniform(60, 80))
        for i in range(int(duration / self.dt)):
            ramp = min((i + 1) * self.dt / 0.2, 1.0)
            self.vcom = self.vcom + down * a * ramp * self.dt
            self.com = self.com + self.vcom * self.dt
            self._relax_tilt(lie, 0.0, 4.0)
            self.record()

    def _relax_tilt(self, target_phi: float, target_perp: float, omega: float) -> None:
        self.dphi += (-omega * omega * (self.phi - target_phi) - 2 * 0.5 * omega * self.dphi) * self.dt
        self.phi += self.dphi * self.dt
        self._perp_step(target_perp, omega)

    def _perp_step(self, target: float, omega: float = 4.0, zeta: float = 0.7) -> None:
        """Second-order (rate-continuous) dynamics for the perpendicular tilt in every phase."""
        self.dperp += (-omega * omega * (self.perp - target) - 2 * zeta * omega * self.dperp) * self.dt
        self.perp += self.dperp * self.dt

    # --- commit --------------------------------------------------------------------------
    def commit(self) -> None:
        """Convert the record to Builder arrays (pelvis anchor, angles, joints) and append."""
        b = self.b
        k = len(self.rec)
        s = b.seg(k * self.dt, "fall_descent", fidget=0.0)
        s.k = k
        for name in ("worn", "breath_amp", "breath_hz", "fidget", "sway"):
            arr = getattr(s, name)
            setattr(s, name, np.resize(arr, k))
        com_mode = np.array([r[0] == "com" for r in self.rec])
        xyz = np.array([r[1] for r in self.rec])
        heading = np.array([r[2] for r in self.rec])
        phi = np.array([r[3] for r in self.rec])
        perp = np.array([r[4] for r in self.rec])
        pitch = phi * np.cos(self.alpha) - perp * np.sin(self.alpha)
        roll = phi * np.sin(self.alpha) + perp * np.cos(self.alpha)
        pos = xyz.copy()
        if com_mode.any():
            pos[com_mode] = xyz[com_mode] + root_rotation(heading[com_mode], pitch[com_mode], roll[com_mode]).apply(self.r_p)
        curl = np.array([r[6] for r in self.rec])
        share = np.array([0.4, 0.35, 0.25])
        s.pos, s.heading, s.pitch, s.roll = pos, heading, pitch, roll
        s.phase = np.array([r[5] for r in self.rec], np.int16)
        s.flex = self.flex0[None] + np.outer(curl * np.cos(self.alpha) * np.sign(phi + 1e-9), share)
        s.lat = self.lat0[None] + np.outer(curl * np.sin(self.alpha) * np.sign(phi + 1e-9), share)
        s.twist = np.tile(self.twist0, (k, 1))
        b.commit(s)
        b.vel = np.zeros(3)
        b.events.extend(self.events)


# --- scenario-level fall routines ------------------------------------------------------------
def lying_rest_height(alpha: float, phi: float) -> float:
    """Pelvis-sensor height when lying: supine ~0.04 m, side ~0.17 m, prone ~0.20 m [design]."""
    face = np.cos(alpha) * np.sign(phi if phi != 0 else 1.0)
    if face < -0.5:
        return 0.04
    if face > 0.5:
        return 0.20
    return 0.17


def pick_lying_angle(rng: np.random.Generator, sign: float) -> float:
    """SIM_SPEC 1.3: inclination >= 60 deg with p = 0.85, else curled 45-70 deg."""
    deg = rng.uniform(82, 95) if rng.random() < 0.85 else rng.uniform(45, 70)
    return sign * np.deg2rad(deg)


def same_level_fall(b: Builder, direction: str, surface: str | None = None, hands: bool = False,
                    slip: bool = False, kick: float | None = None) -> dict:
    """Trip (forward, over the stance foot), slip (foot slides out, backward fall), side fall,
    or knock-down. Hip contact speed emerges from the pendulum + knee-collapse model."""
    surface = surface or b.site.surface
    rng = b.rng
    f = Faller(b, direction)
    f.mark("fall_onset", kind="slip" if slip else ("trip" if direction == "forward" else "fall"))
    friction = available_friction(surface, b.site.condition)
    if kick is None:
        kick = 0.0 if slip else float(rng.uniform(*TRIP_KICK))
    rest = lying_rest_height(f.alpha, 1.0)
    f.pivot(max_s=3.0, collapse=float(rng.uniform(0.0, COLLAPSE_MAX)), collapse_s=float(rng.uniform(0.35, 0.6)),
            contact_height=rest + 0.15, kick=kick, slip_du=float(rng.uniform(1.5, 2.5)) if slip else 0.0,
            keep=1.0 if slip else PIVOT_MOMENTUM_KEEP, resist=0.0 if slip else PIVOT_RESIST,
            friction=max(friction, 0.05))
    soften = float(rng.uniform(0.5, 0.7)) if hands else 1.0  # hands-first arrest: 30-50 % softer
    peak = f.body_contact(surface, rest, pick_lying_angle(rng, 1.0), duration=1.6, soften=soften)
    f.commit()
    return {"peak_pelvis_g": peak, "hip_speed_ms": f.contact_speed, "hands_first": hands}


def height_fall(b: Builder, height: float, start: str = "edge", protection: str = "none",
                surface: str | None = None) -> dict:
    """Fall from height. start: edge | ladder_back | ladder_slip | hole | roof | excavation.
    protection: none | harness | net."""
    surface = surface or b.site.surface
    rng = b.rng
    info: dict = {"height_m": height, "start": start, "protection": protection}
    direction = {"ladder_back": "backward", "ladder_slip": "backward", "hole": "forward"}.get(start, str(rng.choice(list(DIRECTIONS))))
    f = Faller(b, direction)
    f.mark("fall_onset", kind=f"height_{start}")
    landing_y = b.ground_y - height
    feet_first = start in ("hole", "ladder_slip")
    if start in ("roof", "excavation"):
        mu = available_friction(surface, b.site.condition) * 0.5
        f.slide(float(rng.uniform(20, 35)), float(rng.uniform(0.5, 1.5) if start == "roof" else rng.uniform(0.05, 1.0)), mu)
    elif start in ("edge", "ladder_back"):
        f.pivot(max_s=1.2, release_phi=np.deg2rad(rng.uniform(20, 45)), collapse=0.1,
                kick=float(rng.uniform(0.6, 1.5)))
    tumble_acc = 0.0
    if start == "edge":
        target = float(rng.uniform(4.0, 10.0)) * (0.5 if height < 3 else 1.0)  # SIM_SPEC H-SCAF tumble
        if protection == "harness":
            target *= 0.4  # a tethered worker stepping off an edge tumbles far less [design]
        tumble_acc = (np.sign(f.phi or 1.0) * target - f.dphi) / 0.2
    elif feet_first:
        tumble_acc = float(rng.uniform(-1.0, 1.0)) / 0.2  # little rotation (< 2 rad/s)
    harness = None
    if protection == "harness":
        harness = {"free_fall_m": float(rng.uniform(0.3, 1.8))}
        info["free_fall_m"] = round(harness["free_fall_m"], 2)
    net_y = None
    if protection == "net":
        net_y = b.ground_y - min(height, 9.1)  # nets <= 9.1 m below (OSHA 1926.502(c)(1))
        landing_y = net_y
    pelvis_stop = landing_y + b.worker.pelvis_height - 0.05 if feet_first and net_y is None else None
    result = f.flight(landing_y, tumble_acc=tumble_acc, harness=harness, pelvis_stop=pelvis_stop)
    if result == "harness":
        info["arrest_peak_g"] = round(f.harness_arrest(float(rng.uniform(3000, 6000) if rng.random() < 0.9 else rng.uniform(6000, 8000)),
                                                       float(rng.uniform(1.2, 1.8))), 2)
        f.commit()
        return info
    if net_y is not None:
        info["peak_g"] = round(f.net_catch(net_y), 2)
        f.commit()
        return info
    b.ground_y = landing_y
    if result == "feet":
        info["mode"] = "B"
        info["peak_pelvis_g"] = round(f.feet_first(float(rng.uniform(0.1, 0.2))), 2)  # unplanned: stiff legs
        f.commit()
        if height >= 1.0 and rng.random() < 0.7:  # SIM_SPEC 1.2: secondary topple after Mode B
            b.hold(float(rng.uniform(0.2, 0.6)), "impact", sway=0.0, fidget=0.0)
            info["secondary"] = same_level_fall(b, str(rng.choice(list(DIRECTIONS))), surface, kick=float(rng.uniform(1.0, 2.0)))
            info["ends"] = "lying"
        else:
            b.stand_up(1.2, "recovery")
            info["ends"] = "standing"
        return info
    info["mode"] = "A"
    sign = 1.0 if f.phi >= 0 else -1.0
    lie = pick_lying_angle(rng, sign)
    if start == "excavation":
        lie = sign * np.deg2rad(rng.uniform(45, 70))  # curled against the trench wall
    lie += np.round((f.phi - lie) / (2 * np.pi)) * 2 * np.pi  # settle to the nearest lying orientation
    info["peak_pelvis_g"] = round(f.body_contact(surface, lying_rest_height(f.alpha, f.phi), lie, duration=1.8), 2)
    info["ends"] = "lying"
    f.commit()
    return info


def stairs_fall(b: Builder, steps: int, surface: str = "concrete") -> dict:
    """Tumble down a flight: 2-6 impacts 0.2-0.5 s apart, ending on the stairs or the landing."""
    rng = b.rng
    f = Faller(b, "forward" if rng.random() < 0.6 else "backward")
    f.mark("fall_onset", kind="stairs")
    rest = 0.12
    f.pivot(max_s=1.2, collapse=0.4, collapse_s=0.4, contact_height=rest + 0.15, kick=float(rng.uniform(1.0, 2.0)))
    impacts = int(rng.integers(2, 7))
    sign = 1.0 if f.phi >= 0 else -1.0
    peaks = []
    drop_each = 0.18 * steps / impacts
    for i in range(impacts):
        last = i == impacts - 1
        on_landing = rng.random() < 0.5
        lie = sign * np.deg2rad(rng.uniform(82, 95) if (last and on_landing) else rng.uniform(45, 70))
        peaks.append(f.body_contact(surface, rest, lie, duration=1.6 if last else float(rng.uniform(0.2, 0.5))))
        if not last:
            b.ground_y -= drop_each
            f.pelvis_push(f.d * float(rng.uniform(0.8, 1.5)) + UP * 0.6)
    f.commit()
    return {"impacts": impacts, "peak_pelvis_g": round(max(peaks), 2)}


def slow_collapse(b: Builder, from_posture: str = "standing") -> dict:
    """Syncope / heat collapse: knees buckle over 1-2 s, trunk folds, gravity topple onto the
    ground. The pelvis is lowered smoothly (no free fall below 0.8 g); the trunk lands at
    ~1.5-4 g at the upper sensors (SIM_SPEC SYNC)."""
    rng = b.rng
    b.event("fall_onset", kind="collapse")
    b.conscious = False
    buckle = float(rng.uniform(*SYNC_BUCKLE_S)) if (from_posture == "standing" or SYNC_SUPPORT is not None) else 1.0
    if from_posture == "standing":
        drop_to = 0.30 if SYNC_SUPPORT is None else 0.42  # a supported drop starts from a half-buckled stance
        b.move(buckle, "collapse", voluntary=False, y=b.ground_y + drop_to * b.worker.height_m,
               pitch=np.deg2rad(rng.uniform(15, 30)), flex=np.deg2rad([8, 8, 10]))
    else:
        b.move(1.0 if SYNC_SUPPORT is None else min(buckle, 1.0), "collapse", voluntary=False, flex=np.deg2rad([10, 10, 12]))
    f = Faller(b, str(rng.choice(["forward", "left", "right"])), phase="collapse")
    rest = lying_rest_height(f.alpha, 1.0)
    if SYNC_SUPPORT is None:
        peak = f.body_contact(b.site.surface, rest, pick_lying_angle(rng, 1.0), duration=3.0,
                              guided_s=float(rng.uniform(0.8, 1.4)))
    else:
        peak = f.body_contact(b.site.surface, rest, pick_lying_angle(rng, 1.0), duration=3.0, support=SYNC_SUPPORT)
    f.commit()
    return {"peak_pelvis_g": round(peak, 2)}


def feet_first_landing(b: Builder, height: float, forward_speed: float, stop_distance: float,
                       takeoff_up: float = 0.0, phase: str = "activity", label: str = "jump") -> float:
    """Upright ballistic drop and Mode B landing on the feet (planned jumps). Returns peak g."""
    takeoff = float(takeoff_up)
    y = b.pose.pos[1]
    fwd = np.array([-np.sin(b.pose.heading), 0, -np.cos(b.pose.heading)])
    pos = b.pose.pos.copy()
    touch_y = b.ground_y - height + b.worker.pelvis_height - 0.05
    ps = []
    # Take-off: smooth push (150 ms) from rest to (takeoff up, forward_speed ahead).
    k = int(0.15 * b.fs)
    weights = np.sin(np.pi * np.arange(k) / k)
    weights /= weights.sum()
    vy, vh = 0.0, 0.0
    for i in range(k):
        vy += takeoff * weights[i]  # legs push: net upward velocity gain (gravity balanced by the legs)
        vh += forward_speed * weights[i]
        y += vy * b.dt
        pos = pos + fwd * vh * b.dt
        ps.append(np.array([pos[0], y, pos[2]]))
    events = [(b.t + len(ps) * b.dt, "free_fall_start", {"kind": label})]
    while not (y <= touch_y and vy < 0):
        vy -= G * b.dt
        y += vy * b.dt
        pos = pos + fwd * vh * b.dt
        ps.append(np.array([pos[0], y, pos[2]]))
    events.append((b.t + len(ps) * b.dt, "free_fall_end", {"kind": label}))
    v0 = -vy
    decel = v0 * v0 / (2 * stop_distance)
    events.append((b.t + len(ps) * b.dt, "impact", {"mode": "feet_first", "kind": label, "speed_ms": round(v0, 2)}))
    b.events.extend(events)
    i, ramp_k = 0, int(0.01 * b.fs)
    while vy < 0:
        vy += decel * min((i + 1) / ramp_k, 1.0) * b.dt
        y += vy * b.dt
        pos = pos + fwd * vh * max(-vy / v0, 0.0) * b.dt
        ps.append(np.array([pos[0], y, pos[2]]))
        i += 1
    s = b.seg(len(ps) * b.dt, phase)
    s.k = len(ps)
    for name in ("heading", "pitch", "roll", "flex", "lat", "twist", "phase", "worn", "breath_amp", "breath_hz", "fidget", "sway"):
        setattr(s, name, np.resize(getattr(s, name), (s.k,) + getattr(s, name).shape[1:]))
    s.pos = np.array(ps)
    b.commit(s)
    b.vel = np.zeros(3)
    b.ground_y -= height
    return (decel + G) / G


def near_fall(b: Builder, kind: str = "stumble") -> dict:
    """Stumble-and-catch (NF-STUMBLE) or slip with a rail grab (NF-RAIL): dip, fast trunk
    rotation, recovery steps. Starts from the current velocity (no velocity step)."""
    rng = b.rng
    b.event("near_fall_onset", kind=kind)
    p0 = b.pose.copy()
    v0 = b.vel.copy()
    if kind == "stumble":
        rate = float(rng.uniform(1.5, 3.5))
        delta = float(rng.uniform(np.deg2rad(15), np.deg2rad(35)))
        dip, dip_s = float(rng.uniform(0.5, 0.8)), float(rng.uniform(0.05, 0.2))
    else:
        rate = float(rng.uniform(1.5, 3.0))
        delta = -float(rng.uniform(np.deg2rad(10), np.deg2rad(20)))
        dip, dip_s = float(rng.uniform(0.4, 0.7)), float(rng.uniform(0.2, 0.4))
    tilt_s = abs(delta) / rate * 1.875
    total = tilt_s + 0.8
    s = b.seg(total, "near_fall")
    k = s.k
    t = np.arange(k) * b.dt
    up = np.clip(t / tilt_s, 0, 1)
    down = np.clip((t - tilt_s) / 0.8, 0, 1)
    shape = (10 * up**3 - 15 * up**4 + 6 * up**5) - (10 * down**3 - 15 * down**4 + 6 * down**5)
    s.pitch = p0.pitch + delta * shape
    s.flex = p0.flex + np.outer(shape * delta * 0.3, [0.4, 0.35, 0.25])
    vy, y, ys = float(v0[1]), p0.pos[1], []
    for ti in t:
        if ti < dip_s:
            a = -(1 - dip) * G
        elif vy < 0:
            a = 1.5 * G
        else:
            a, vy = 0.0, 0.0
        vy += a * b.dt
        y += vy * b.dt
        ys.append(y)
    fwd = np.array([-np.sin(p0.heading), 0, -np.cos(p0.heading)])
    decay = 1 - np.clip(t / total, 0, 1)
    decay = decay * decay * (3 - 2 * decay)  # smoothstep from 1 to 0
    lurch = 0.6 * np.sin(np.pi * np.clip(t / total, 0, 1)) ** 2 * np.sign(delta)
    vh = np.outer(decay, v0 * np.array([1, 0, 1])) + np.outer(lurch, fwd)
    s.pos = p0.pos + np.cumsum(vh, axis=0) * b.dt
    s.pos[:, 1] = np.array(ys)
    if kind == "rail":
        b.impulses(s, np.array([dip_s]), (2.0, 4.0), pelvis_g=1.5)  # arm arrest on the rail
    b.commit(s)
    b.vel = np.zeros(3)
    b.walk(float(rng.uniform(1.0, 3.0)), speed=1.0, phase="recovery", stop=True)
    return {"kind": kind, "pitch_rate": round(rate, 2)}


def struck_by(b: Builder, knockdown: bool) -> dict:
    """Blow from a swinging load: 1-3 g trunk jolt over 50-150 ms, then a stumble or a fall."""
    rng = b.rng
    direction = str(rng.choice(["left", "right", "forward"]))  # a posterior blow pushes forward
    jolt_g = float(rng.uniform(1.0, 3.0))
    width = float(rng.uniform(0.05, 0.15))
    s = b.seg(width, "near_fall")
    t = np.arange(s.k) * b.dt
    acc = jolt_g * G * np.sin(np.pi * t / width) ** 2  # sin^2 pulse peaking at jolt_g, smooth edges
    v = np.cumsum(acc) * b.dt
    alpha = DIRECTIONS[direction]
    fwd = np.array([-np.sin(b.pose.heading), 0, -np.cos(b.pose.heading)])
    right = np.array([np.cos(b.pose.heading), 0, -np.sin(b.pose.heading)])
    dvec = np.cos(alpha) * fwd + np.sin(alpha) * right
    s.pos = b.pose.pos + np.outer(np.cumsum(v) * b.dt, dvec)
    b.event("struck", jolt_g=round(jolt_g, 2), direction=direction)
    b.commit(s)
    b.vel = dvec * float(v[-1])
    if knockdown:
        return {"jolt_g": jolt_g, **same_level_fall(b, direction, kick=float(rng.uniform(0.8, 1.5)))}
    near_fall(b, "stumble")
    return {"jolt_g": jolt_g}


def post_fall(b: Builder, behaviour: str, duration: float, phase: str = "lying") -> None:
    """unconscious | still_then_move | struggle | get_up. Breathing continues while worn."""
    rng = b.rng
    if behaviour == "unconscious":
        b.conscious = False
        b.hold(duration, phase, sway=0.0, fidget=0.0)
    elif behaviour == "still_then_move":
        still = min(float(rng.uniform(10, 60)), duration)
        b.conscious = False
        b.hold(still, phase, sway=0.0, fidget=0.0)
        b.conscious = True
        _movement_bursts(b, duration - still, phase, amp_deg=8.0)
    elif behaviour == "struggle":
        b.conscious = True
        _movement_bursts(b, duration, phase, amp_deg=12.0)
    elif behaviour == "get_up":
        b.conscious = True
        b.hold(min(duration, float(rng.uniform(3, 15))), phase, sway=0.0)
        b.get_up("recovery")
        b.walk(float(rng.uniform(5, 15)), phase="recovery")
    b.conscious = True


def _movement_bursts(b: Builder, duration: float, phase: str, amp_deg: float) -> None:
    rng = b.rng
    remaining = duration
    while remaining > 0.5:
        seg = b.seg(min(float(rng.uniform(1.5, 5)), remaining), phase, fidget=3.0)
        shape = bump(seg.k)
        seg.flex = seg.flex + np.outer(shape, np.deg2rad(rng.uniform(-amp_deg, amp_deg, 3)))
        seg.lat = seg.lat + np.outer(shape, np.deg2rad(rng.uniform(-amp_deg, amp_deg, 3)) * 0.6)
        seg.pos[:, 1] += 0.03 * shape * rng.uniform(0.3, 1.0)
        b.commit(seg)
        remaining -= seg.k * b.dt
        if remaining > 0.5:
            pause = b.seg(min(float(rng.uniform(1, 4)), remaining), phase, fidget=1.0)
            b.commit(pause)
            remaining -= pause.k * b.dt


__all__ = ["Faller", "same_level_fall", "height_fall", "stairs_fall", "slow_collapse", "feet_first_landing",
           "near_fall", "struck_by", "post_fall", "smooth_pulse"]
