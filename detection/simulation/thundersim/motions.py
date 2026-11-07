# File: motions.py
# Purpose: Builder that strings voluntary motion primitives (stand, walk, run, stoop, kneel,
#          crawl, ladder, sit, lie down, tools, equipment...) into one continuous PoseTrack.
# Contributors: Unassigned
"""Voluntary motion primitives. Physics-driven falls live in falls.py and use the same Builder.

Postures are kinematic keyframes joined by minimum-jerk transitions whose duration is
stretched so the trunk never exceeds MAX_VOLUNTARY_RATE. Oscillations (gait, vibration,
breathing, sway, fidget) are added on top. All amplitudes are [design] values unless a
source is named; they were chosen to land in the ranges SIM_SPEC.md section 4 describes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .site import SURFACES, Site
from .spine import (MAX_VOLUNTARY_RATE, PHASE_CODE, PoseTrack, Worker, clamp_joints)

G = 9.80665


def minjerk(k: int) -> np.ndarray:
    tau = np.linspace(0.0, 1.0, max(k, 2))[:k]
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def forward_vec(heading: float) -> np.ndarray:
    return np.array([-np.sin(heading), 0.0, -np.cos(heading)])


def right_vec(heading: float) -> np.ndarray:
    return np.array([np.cos(heading), 0.0, -np.sin(heading)])


def smooth_pulse(fs: float, f_hz: float, sigma_s: float) -> np.ndarray:
    """Displacement shape (Gaussian-windowed sine) whose 2nd derivative peaks at 1 m/s^2 per m.

    It starts and ends with zero displacement AND zero velocity, so adding it never creates a
    velocity step (which finite differences would turn into a fake acceleration spike).
    """
    centre = 3 * sigma_s
    tau = np.arange(int(2 * centre * fs) + 1) / fs - centre
    shape = np.exp(-(tau / sigma_s) ** 2) * np.sin(2 * np.pi * f_hz * tau)
    accel = np.gradient(np.gradient(shape, 1 / fs), 1 / fs)
    return shape / np.max(np.abs(accel))


def heel_pulse(fs: float, sigma_s: float) -> np.ndarray:
    """Displacement dip -exp(-tau^2/sigma^2): its 2nd derivative is an upward spike (peak 1)
    with small negative side lobes (~0.45), like a trunk heel-strike transient. Zero velocity
    at both ends."""
    tau = np.arange(int(6 * sigma_s * fs) + 1) / fs - 3 * sigma_s
    shape = -np.exp(-(tau / sigma_s) ** 2)
    accel = np.gradient(np.gradient(shape, 1 / fs), 1 / fs)
    return shape / np.max(accel)


def taper(k: int, fs: float, seconds: float = 0.2) -> np.ndarray:
    """0 -> 1 -> 0 minimum-jerk window, used to fade overlays in and out without steps."""
    w = np.ones(k)
    r = min(int(seconds * fs), k // 2)
    if r >= 2:
        w[:r] = minjerk(r)
        w[k - r:] = minjerk(r)[::-1]
    return w


def band_noise(rng: np.random.Generator, k: int, fs: float, lo: float | None, hi: float) -> np.ndarray:
    """Unit-RMS noise limited to [lo, hi] Hz (low-pass when lo is None)."""
    # 4th-order, zero-phase (effective 8th order): steep enough that the noise stays smooth
    # after double differentiation. A 2nd-order roll-off leaves white-noise *acceleration*.
    x = rng.normal(size=k + 4000)
    if lo is None:
        sos = signal.butter(4, hi, fs=fs, output="sos")
    else:
        sos = signal.butter(4, [lo, min(hi, fs / 2 - 1)], btype="band", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, x)[2000:2000 + k]
    return y / max(np.std(y), 1e-12)


@dataclass
class Pose:
    pos: np.ndarray
    heading: float
    pitch: float
    roll: float
    flex: np.ndarray
    lat: np.ndarray
    twist: np.ndarray
    worn: bool = True

    def copy(self) -> "Pose":
        return Pose(self.pos.copy(), self.heading, self.pitch, self.roll, self.flex.copy(), self.lat.copy(), self.twist.copy(), self.worn)


class Seg:
    """Arrays for one segment, initialised to a constant pose."""

    def __init__(self, pose: Pose, k: int, phase: int, breath_amp: float, breath_hz: float, fidget: float, sway: float):
        self.k = k
        self.pos = np.tile(pose.pos, (k, 1))
        self.heading = np.full(k, pose.heading)
        self.pitch = np.full(k, pose.pitch)
        self.roll = np.full(k, pose.roll)
        self.flex = np.tile(pose.flex, (k, 1))
        self.lat = np.tile(pose.lat, (k, 1))
        self.twist = np.tile(pose.twist, (k, 1))
        self.phase = np.full(k, phase, np.int16)
        self.worn = np.full(k, pose.worn)
        on = 1.0 if pose.worn else 0.0
        self.breath_amp = np.full(k, breath_amp * on)
        self.breath_hz = np.full(k, breath_hz)
        self.fidget = np.full(k, fidget * on)
        self.sway = np.full(k, sway * on)


class Builder:
    """Accumulates segments at fs = 1 kHz and records ground-truth events."""

    def __init__(self, worker: Worker, site: Site, rng: np.random.Generator, fs: float = 1000.0):
        self.worker, self.site, self.rng, self.fs = worker, site, rng, fs
        self.dt = 1.0 / fs
        self.segs: list[Seg] = []
        self.events: list[tuple[float, str, dict]] = []
        self.mount_steps: list[tuple[float, int, np.ndarray]] = []
        self.work_modes: list[tuple[float, str]] = [(0.0, "normal")]
        self.n = 0
        h = worker.height_m
        self.pose = Pose(np.array([0.0, worker.pelvis_height, 0.0]), float(rng.uniform(-np.pi, np.pi)), 0.0, 0.0,
                         np.zeros(3), np.zeros(3), np.zeros(3))
        self.vel = np.zeros(3)
        self.ground_y = 0.0  # current support-surface height under the worker
        self.breath_hz = float(rng.uniform(0.2, 0.33))   # SIM_SPEC.md 1.3 breathing range
        self.breath_amp = np.deg2rad(rng.uniform(0.4, 0.8))  # ~ +/-0.01 g gravity projection
        self.conscious = True
        self.height = h

    # --- bookkeeping ---------------------------------------------------------------------
    @property
    def t(self) -> float:
        return self.n / self.fs

    def event(self, name: str, **info) -> None:
        self.events.append((self.t, name, info))

    def set_work_mode(self, mode: str) -> None:
        self.work_modes.append((self.t, mode))

    def seg(self, duration: float, phase: str = "activity", fidget: float | None = None, sway: float = 0.0) -> Seg:
        k = max(int(round(duration * self.fs)), 2)
        if fidget is None:
            fidget = 1.0 if self.conscious else 0.0
        return Seg(self.pose, k, PHASE_CODE[phase], self.breath_amp, self.breath_hz, fidget, sway)

    CHANNELS = ("pos", "heading", "pitch", "roll", "flex", "lat", "twist")

    def _match_join(self, s: Seg) -> None:
        """Make every channel C1-continuous with the previous segment.

        The new segment is shifted so its first sample continues the previous one, and any
        rate mismatch dv is removed with c(t) = dv * tau * (1 - exp(-t / tau)), tau = 50 ms:
        a brief, finite acceleration instead of a velocity step (which differentiation would
        turn into a fake impact). Intentional rapid changes inside a segment are untouched.
        """
        if not self.segs or s.k < 3:
            return
        prev = self.segs[-1]
        tau = 0.05
        t = (np.arange(s.k) + 1) * self.dt
        corr = tau * (1 - np.exp(-t / tau))
        for name in self.CHANNELS:
            cur = getattr(s, name)
            old = getattr(prev, name)
            prev_rate = (old[-1] - old[-2]) * self.fs if prev.k >= 2 else 0.0 * old[-1]
            first = old[-1] + prev_rate * self.dt
            new_rate = (cur[1] - cur[0]) * self.fs
            shifted = cur + (first - cur[0])
            dv = prev_rate - new_rate
            if cur.ndim == 1:
                shifted = shifted + dv * corr
            else:
                shifted = shifted + corr[:, None] * dv[None, :]
            setattr(s, name, shifted)

    def commit(self, s: Seg) -> None:
        self._match_join(s)
        self.segs.append(s)
        self.n += s.k
        last = s.k - 1
        if s.k >= 2:
            self.vel = (s.pos[last] - s.pos[last - 1]) * self.fs
        self.pose = Pose(s.pos[last].copy(), float(s.heading[last]), float(s.pitch[last]), float(s.roll[last]),
                         s.flex[last].copy(), s.lat[last].copy(), s.twist[last].copy(), bool(s.worn[last]))

    # --- generic voluntary transitions ---------------------------------------------------
    def move(self, duration: float, phase: str = "activity", voluntary: bool = True, **target) -> Seg:
        """Minimum-jerk move of any of: pos, y, heading, pitch, roll, flex, lat, twist."""
        p = self.pose
        goal = {"pos": p.pos.copy(), "heading": p.heading, "pitch": p.pitch, "roll": p.roll,
                "flex": p.flex.copy(), "lat": p.lat.copy(), "twist": p.twist.copy()}
        if "y" in target:
            goal["pos"][1] = target.pop("y")
        if "forward" in target:
            goal["pos"] = goal["pos"] + forward_vec(p.heading) * target.pop("forward")
        for key, value in target.items():
            goal[key] = np.array(value, float) if key in ("pos", "flex", "lat", "twist") else float(value)
        if voluntary:
            change = max(abs(goal["pitch"] - p.pitch), abs(goal["roll"] - p.roll),
                         float(np.max(np.abs(goal["flex"] - p.flex))), float(np.max(np.abs(goal["lat"] - p.lat))))
            duration = max(duration, change / MAX_VOLUNTARY_RATE * 1.9)  # min-jerk peak rate = 1.875x mean
        s = self.seg(duration, phase)
        w = minjerk(s.k)
        s.pos = p.pos + (goal["pos"] - p.pos) * w[:, None]
        s.heading = p.heading + (goal["heading"] - p.heading) * w
        s.pitch = p.pitch + (goal["pitch"] - p.pitch) * w
        s.roll = p.roll + (goal["roll"] - p.roll) * w
        s.flex = p.flex + (goal["flex"] - p.flex) * w[:, None]
        s.lat = p.lat + (goal["lat"] - p.lat) * w[:, None]
        s.twist = p.twist + (goal["twist"] - p.twist) * w[:, None]
        self.commit(s)
        return s

    def hold(self, duration: float, phase: str = "activity", sway: float = 1.0, fidget: float | None = None) -> Seg:
        s = self.seg(duration, phase, fidget=fidget, sway=sway)
        self.commit(s)
        return s

    # --- postures (pelvis-sensor heights are [design] fractions of body height) ----------
    def stand_up(self, duration: float = 1.2, phase: str = "activity") -> None:
        self.move(duration, phase, y=self.ground_y + self.worker.pelvis_height, pitch=0.0, roll=0.0,
                  flex=np.zeros(3), lat=np.zeros(3), twist=np.zeros(3))

    def stand(self, duration: float, phase: str = "activity") -> None:
        self.hold(duration, phase, sway=1.0)

    def stoop(self, depth_deg: float, down_s: float = 1.2, hold_s: float = 1.0, up_s: float = 1.2, phase: str = "activity") -> None:
        """Trunk flexion with lumbo-pelvic rhythm (~42 % pelvis, Paloschi 2021 [src via spine note])."""
        hip = np.deg2rad(depth_deg) * 0.45
        spine = np.deg2rad(depth_deg) - hip
        self.move(down_s, phase, y=self.ground_y + self.worker.pelvis_height - 0.03, pitch=hip,
                  flex=spine * np.array([0.45, 0.35, 0.20]))
        self.hold(hold_s, phase, sway=0.5)
        self.stand_up(up_s, phase)

    def squat(self, hold_s: float, phase: str = "activity") -> None:
        self.move(1.0, phase, y=self.ground_y + 0.30 * self.height, pitch=np.deg2rad(30), flex=np.deg2rad([8, 6, 4]))
        self.hold(hold_s, phase, sway=0.4)
        self.stand_up(1.0, phase)

    def kneel(self, trunk_deg: float = 10.0, duration: float = 1.8, phase: str = "activity") -> None:
        hip = np.deg2rad(trunk_deg) * 0.6
        self.move(duration, phase, y=self.ground_y + 0.30 * self.height, pitch=hip,
                  flex=(np.deg2rad(trunk_deg) - hip) * np.array([0.45, 0.35, 0.20]))

    def quadruped(self, duration: float = 1.5, phase: str = "activity") -> None:
        self.move(duration, phase, y=self.ground_y + 0.28 * self.height, pitch=np.deg2rad(82), flex=np.deg2rad([2, 2, 4]))

    def sit_ground(self, duration: float = 1.8, phase: str = "activity") -> None:
        self.move(duration, phase, y=self.ground_y + 0.10, pitch=np.deg2rad(-8), flex=np.deg2rad([10, 8, 6]))

    def lie(self, posture: str, duration: float = 2.0, phase: str = "activity") -> None:
        """Controlled lie-down from sitting/kneeling into supine, prone or side posture."""
        if posture == "supine":
            if self.pose.pos[1] > self.ground_y + 0.25:
                self.kneel(10, 1.5, phase)
                self.sit_ground(1.5, phase)
            self.move(duration, phase, y=self.ground_y + 0.03, pitch=np.deg2rad(-88), roll=0.0, flex=np.zeros(3))
        elif posture == "prone":
            if self.pose.pos[1] > self.ground_y + 0.45 * self.height:
                self.kneel(10, 1.5, phase)
            self.quadruped(1.2, phase)
            self.move(duration, phase, y=self.ground_y + 0.20, pitch=np.deg2rad(88), flex=np.zeros(3))
        else:  # side
            if self.pose.pos[1] > self.ground_y + 0.25:
                self.kneel(10, 1.5, phase)
                self.sit_ground(1.5, phase)
            sign = 1.0 if posture == "right" else -1.0
            self.move(duration, phase, y=self.ground_y + 0.17, pitch=0.0, roll=sign * np.deg2rad(85), flex=np.deg2rad([8, 8, 10]))

    def get_up(self, phase: str = "recovery") -> None:
        """From any lying posture: sit or kneel, then stand."""
        if self.pose.pitch > np.deg2rad(45):  # prone-ish: push up to hands and knees
            self.quadruped(1.5, phase)
            self.kneel(10, 1.2, phase)
        else:
            self.sit_ground(1.8, phase)
            self.kneel(10, 1.2, phase)
        self.stand_up(1.3, phase)

    # --- locomotion ----------------------------------------------------------------------
    def walk(self, duration: float, speed: float | None = None, phase: str = "activity", load_kg: float = 0.0,
             turn_rate: float = 0.0, stop: bool = True, lean_deg: float = 3.0) -> None:
        surface = SURFACES[self.site.surface]
        if speed is None:
            speed = float(self.rng.uniform(1.1, 1.45))
        speed *= 0.75 if surface.name in ("mud", "gravel", "rebar_mat") else 1.0
        speed *= 0.85 if load_kg > 10 else 1.0
        s = self.seg(duration, phase)
        k, t = s.k, np.arange(s.k) * self.dt
        ramp = np.ones(k)
        r = min(int(0.6 * self.fs), k // 2)
        if self.vel[[0, 2]].dot(self.vel[[0, 2]]) < 0.04:  # starting from rest
            ramp[:r] = minjerk(r)
        if stop:
            ramp[k - r:] = minjerk(r)[::-1]
        f_step = float(np.clip(1.8 * np.sqrt(speed / 1.3) * self.rng.uniform(0.95, 1.05), 1.4, 2.3))
        jitter = np.cumsum(band_noise(self.rng, k, self.fs, None, 0.5)) * self.dt * 0.15  # step timing variability
        theta = np.pi * f_step * t + jitter + self.rng.uniform(0, 2 * np.pi)  # stride phase
        path_heading = self.pose.heading + turn_rate * t
        heading = path_heading + np.deg2rad(4) * np.sin(theta) * ramp  # pelvis rotation within the stride
        fwd = np.stack([-np.sin(path_heading), np.zeros(k), -np.cos(path_heading)], axis=1)
        right = np.stack([np.cos(path_heading), np.zeros(k), -np.sin(path_heading)], axis=1)
        s.pos = self.pose.pos + np.cumsum(fwd * (speed * ramp)[:, None], axis=0) * self.dt
        bob = 0.025 * (speed / 1.3) * ramp * (np.cos(2 * theta) - 1) / 2
        lateral = 0.02 * ramp * np.sin(theta)
        s.pos[:, 1] += bob
        s.pos += lateral[:, None] * right
        # Uneven ground: random step-height offsets, smoothed within each step.
        if surface.irregularity_m > 0:
            steps = np.floor((theta - theta[0]) / np.pi).astype(int)
            offsets = self.rng.normal(0, surface.irregularity_m, steps.max() + 2)
            raw = offsets[steps]
            s.pos[:, 1] += signal.sosfiltfilt(signal.butter(2, 3.0, fs=self.fs, output="sos"), raw) * ramp
        # Heel-strike transients: damped oscillation at each foot contact, amplitude by surface.
        if surface.heel_strike_g > 0:
            contacts = np.where(np.diff(np.floor((2 * theta - np.pi) / (2 * np.pi))) > 0)[0]
            pulse = surface.heel_strike_g * G * heel_pulse(self.fs, 0.25 / surface.heel_strike_hz)
            for c in contacts:
                end = min(c + len(pulse), k)
                s.pos[c:end, 1] += pulse[: end - c] * ramp[c] * self.rng.uniform(0.7, 1.3)
        lean = np.deg2rad(lean_deg + (6.0 if load_kg > 10 else 0.0))
        s.heading = heading
        s.pitch = self.pose.pitch + (lean * 0.5 + np.deg2rad(2) * np.sin(2 * theta + 0.5)) * ramp
        s.roll = self.pose.roll + np.deg2rad(3) * np.sin(theta) * ramp
        s.flex = self.pose.flex + (lean * 0.5 * np.array([0.4, 0.35, 0.25]))[None] * ramp[:, None]
        s.twist = self.pose.twist + np.outer(-np.deg2rad(2) * np.sin(theta) * ramp, [0.2, 0.4, 0.4])
        s.breath_hz[:] = 0.4
        self.commit(s)
        if stop:
            self.vel = np.zeros(3)

    def run(self, duration: float, speed: float = 3.0, phase: str = "activity") -> None:
        """Running: stance pulses (~2-4 g) separated by flight phases (~0 g) at 2.5-3 Hz."""
        s = self.seg(duration, phase)
        k = s.k
        cadence = float(self.rng.uniform(2.5, 3.0))
        t_step = 1.0 / cadence
        t_stance = t_step * self.rng.uniform(0.6, 0.7)
        t = np.arange(k) * self.dt
        tin = np.mod(t, t_step)
        # Specific force: half-sine during stance, zero in flight; mean equals g over a step.
        peak = G * t_step * np.pi / (2 * t_stance)
        f = np.where(tin < t_stance, peak * np.sin(np.pi * tin / t_stance), 0.0)
        ramp = np.clip(t / 0.8, 0, 1) * np.clip((t[-1] - t) / 0.8, 0, 1)
        acc_y = (f - G) * ramp
        vy = np.cumsum(acc_y) * self.dt
        vy -= np.linspace(0, vy[-1], k)
        y = np.cumsum(vy) * self.dt
        y -= np.linspace(0, y[-1], k)
        fwd = forward_vec(self.pose.heading)
        s.pos = self.pose.pos + np.outer(np.cumsum(speed * ramp) * self.dt, fwd)
        s.pos[:, 1] += y
        s.pitch = self.pose.pitch + np.deg2rad(8) * ramp
        s.roll = self.pose.roll + np.deg2rad(4) * np.sin(np.pi * cadence * t) * ramp
        s.breath_hz[:] = 0.6
        self.commit(s)
        self.vel = np.zeros(3)

    def crawl(self, duration: float, phase: str = "activity") -> None:
        s = self.seg(duration, phase)
        t = np.arange(s.k) * self.dt
        f = float(self.rng.uniform(0.8, 1.2))
        fwd = forward_vec(self.pose.heading)
        ramp = taper(s.k, self.fs, 0.8)
        s.pos = self.pose.pos + np.outer(np.cumsum(0.3 * ramp) * self.dt, fwd)
        s.pos[:, 1] += 0.01 * (np.cos(2 * np.pi * f * t) - 1) * ramp
        s.roll = self.pose.roll + np.deg2rad(4) * np.sin(np.pi * f * t) * ramp
        s.heading = self.pose.heading + np.deg2rad(3) * np.sin(np.pi * f * t) * ramp
        self.commit(s)

    def ladder(self, height: float, up: bool = True, phase: str = "activity") -> None:
        """Climb at ~0.6-1 rung/s, rungs 0.3 m apart, ladder at 1:4 (29 CFR 1926.1053(b)(5)(i))."""
        rungs = max(int(round(height / 0.3)), 1)
        rate = float(self.rng.uniform(0.6, 1.0))
        self.move(0.8, phase, pitch=np.deg2rad(8), flex=np.deg2rad([2, 2, 3]))
        sign = 1.0 if up else -1.0
        for _ in range(rungs):
            s = self.seg(1.0 / rate, phase)
            w = minjerk(s.k)
            rise = sign * 0.3 * w
            s.pos[:, 1] += rise
            s.pos += np.outer(sign * 0.3 / 4.0 * w, forward_vec(self.pose.heading))
            s.roll = self.pose.roll + np.deg2rad(3) * np.sin(np.pi * w)
            s.pos += np.outer(0.015 * np.sin(np.pi * w), right_vec(self.pose.heading))
            self.commit(s)
        self.ground_y += sign * rungs * 0.3

    # --- work activities -----------------------------------------------------------------
    def impulses(self, s: Seg, times: np.ndarray, upper_g: tuple[float, float], pelvis_g: float = 0.4) -> None:
        """Tool strikes: short damped jerks of the upper joints (and a smaller pelvis jolt)."""
        shape = smooth_pulse(self.fs, 22.0, 0.012)
        spacing = self.worker.spacings()
        for c in (np.asarray(times) * self.fs).astype(int):
            if c < 0 or c >= s.k:
                continue
            end = min(c + len(shape), s.k)
            angle = self.rng.uniform(*upper_g) * G / spacing[2]  # tangential accel = L * angular accel
            s.flex[c:end, 2] += angle * shape[: end - c]
            s.flex[c:end, 1] += 0.4 * angle * shape[: end - c]
            s.pos[c:end, 1] += pelvis_g * G * self.rng.uniform(0.3, 1.0) * shape[: end - c]

    def hammer(self, duration: float, overhead: bool = False, phase: str = "activity") -> None:
        if overhead:
            self.move(1.0, phase, flex=np.deg2rad([-4, -6, -8]))
        s = self.seg(duration, phase, sway=0.5)
        rate = self.rng.uniform(2.0, 6.0)  # SIM_SPEC A-HAMMER: 2-6 Hz strikes
        base = np.arange(0.5, duration, 1.0 / rate)
        self.impulses(s, base + self.rng.normal(0, 0.02, len(base)), (1.5, 3.0), pelvis_g=0.5)
        self.commit(s)
        if overhead:
            self.stand_up(1.0, phase)

    def vibrate(self, duration: float, band: tuple[float, float], peak_g: float, slots: str = "trunk", phase: str = "activity") -> Seg:
        """Band-limited vibration (displacement from double-integrated acceleration noise)."""
        s = self.seg(duration, phase, sway=0.3)
        for axis, weight in ((1, 1.0), (0, 0.5), (2, 0.5)):
            a = band_noise(self.rng, s.k, self.fs, band[0], band[1]) * peak_g * G / 3 * weight
            spec = np.fft.rfft(a)
            freqs = np.fft.rfftfreq(s.k, self.dt)
            freqs[0] = 1.0
            disp = np.fft.irfft(-spec / (2 * np.pi * freqs) ** 2, n=s.k)
            disp = (disp - np.linspace(disp[0], disp[-1], s.k)) * taper(s.k, self.fs, 0.3)
            if slots == "upper":
                spacing = self.worker.spacings()[2]
                s.flex[:, 2] += disp / spacing
            else:
                s.pos[:, axis] += disp
        self.commit(s)
        return s

    def jackhammer(self, duration: float, phase: str = "activity") -> None:
        self.move(1.0, phase, pitch=np.deg2rad(18), flex=np.deg2rad([5, 4, 3]))
        self.vibrate(duration, (20.0, 40.0), float(self.rng.uniform(1.0, 5.0)), "trunk", phase)  # SIM_SPEC A-JACK [design]
        self.stand_up(1.0, phase)

    def drill_overhead(self, duration: float, phase: str = "activity") -> None:
        ext = self.rng.uniform(10, 30)
        self.move(1.2, phase, flex=-np.deg2rad(ext) * np.array([0.3, 0.35, 0.35]))
        self.vibrate(duration, (20.0, 50.0), float(self.rng.uniform(0.2, 1.0)), "upper", phase)
        self.stand_up(1.0, phase)

    def carry(self, duration: float, load_kg: float, phase: str = "activity") -> None:
        self.stoop(self.rng.uniform(60, 90), hold_s=self.rng.uniform(1, 3))
        self.walk(duration, load_kg=load_kg, phase=phase)
        self.stoop(self.rng.uniform(60, 90), hold_s=self.rng.uniform(1, 3))

    def rebar_tie(self, duration: float, phase: str = "activity") -> None:
        """Sustained stoop at 70-100 deg with small tying jerks and short shuffles."""
        depth = np.deg2rad(self.rng.uniform(70, 100))
        hip = depth * 0.5
        self.move(1.5, phase, y=self.ground_y + self.worker.pelvis_height - 0.05, pitch=hip,
                  flex=(depth - hip) * np.array([0.45, 0.35, 0.2]))
        elapsed = 0.0
        while elapsed < duration:
            chunk = float(self.rng.uniform(20, 40))
            s = self.seg(chunk, phase, sway=0.4)
            self.impulses(s, np.arange(1.0, chunk, self.rng.uniform(3, 5)), (0.1, 0.3), pelvis_g=0.05)
            self.commit(s)
            s = self.seg(1.5, phase)
            s.pos += np.outer(0.4 * minjerk(s.k), forward_vec(self.pose.heading))
            self.commit(s)
            elapsed += chunk + 1.5
        self.stand_up(1.5, phase)

    def kneel_work(self, duration: float, phase: str = "activity") -> None:
        self.kneel(float(self.rng.uniform(45, 90)), 2.0, phase)
        s = self.seg(duration, phase, sway=0.3)
        self.impulses(s, np.arange(2.0, duration, self.rng.uniform(4, 8)), (0.1, 0.4), pelvis_g=0.05)
        self.commit(s)
        self.stand_up(1.5, phase)

    def sit(self, seat_height: float = 0.45, hard: bool = False, phase: str = "activity") -> None:
        """Sit on a bucket/bench. hard=True: last 0.1-0.3 s near free fall, then a 2-4 g seat impact."""
        seated_y = self.ground_y + seat_height + 0.08
        if not hard:
            self.move(1.5, phase, y=seated_y, pitch=np.deg2rad(-5), flex=np.deg2rad([6, 4, 2]))
            return
        drop = float(self.rng.uniform(0.04, 0.15))
        self.move(0.8, phase, y=seated_y + drop, pitch=np.deg2rad(5), flex=np.deg2rad([4, 3, 2]))
        dip = float(self.rng.uniform(0.5, 0.8))
        y, v, ys = seated_y + drop, 0.0, []
        k_seat, m = float(self.rng.uniform(2.5e4, 5e4)), 30.0
        c = 2 * 0.4 * np.sqrt(k_seat * m)
        for _ in range(int(0.8 * self.fs)):
            if y > seated_y:
                a = -(1 - dip) * G  # specific force = dip * g while dropping onto the seat
            else:
                force = max(k_seat * (seated_y - y) - c * v, 0.0)  # the seat only pushes
                a = force / m - G
            v += a * self.dt
            y += v * self.dt
            ys.append(y)
        s = self.seg(len(ys) / self.fs, phase)
        s.pos[:, 1] = np.array(ys)[: s.k]
        self.commit(s)
        self.move(0.5, phase, y=seated_y, pitch=np.deg2rad(-5))

    def cab_ride(self, duration: float, phase: str = "activity") -> None:
        """Equipment seat: whole-body vibration 1-8 Hz, jolts 1-3 g, dips over bumps, vehicle motion."""
        self.ladder(1.0, up=True, phase=phase)
        self.sit(0.45, phase=phase)
        s = self.seg(duration, phase, sway=0.0)
        t = np.arange(s.k) * self.dt
        wbv = band_noise(self.rng, s.k, self.fs, 1.0, 8.0) * self.rng.uniform(0.3, 1.0)  # m/s^2 rms
        bumps = np.zeros(s.k)
        for c in self.rng.choice(s.k, size=max(int(duration / 8), 1), replace=False):
            width = int(self.rng.uniform(0.08, 0.2) * self.fs)
            g_bump = self.rng.uniform(1.0, 3.0) * G
            end = min(c + width, s.k)
            bumps[c:end] += g_bump * np.sin(np.pi * np.arange(end - c) / width)
            dip_end = min(end + width, s.k)
            bumps[end:dip_end] -= self.rng.uniform(0.1, 0.4) * G * np.sin(np.pi * np.arange(dip_end - end) / width)
        a = wbv + bumps
        a -= a.mean()
        spec = np.fft.rfft(a)
        freqs = np.fft.rfftfreq(s.k, self.dt)
        freqs[0] = 1.0
        disp = np.fft.irfft(-spec / (2 * np.pi * freqs) ** 2, n=s.k)
        disp = signal.sosfiltfilt(signal.butter(2, 0.3, "high", fs=self.fs, output="sos"), disp)
        s.pos[:, 1] += disp * taper(s.k, self.fs, 1.0)
        # Vehicle travel: slow speed changes and turns, starting and ending at rest.
        speed = np.clip(np.cumsum(band_noise(self.rng, s.k, self.fs, None, 0.05)) * self.dt * 0.5 + 2.0, 0, 4)
        speed *= taper(s.k, self.fs, 4.0)
        heading = self.pose.heading + np.cumsum(band_noise(self.rng, s.k, self.fs, None, 0.05)) * self.dt * 0.2
        step = np.stack([-np.sin(heading), np.zeros(s.k), -np.cos(heading)], axis=1)
        s.pos += np.cumsum(step * speed[:, None], axis=0) * self.dt
        s.heading = heading
        self.commit(s)
        self.vel = np.zeros(3)
        self.stand_up(1.2, phase)
        self.ladder(1.0, up=False, phase=phase)

    def jump_down(self, height: float, phase: str = "activity") -> None:
        """Planned jump from a step/truck bed: crouch, take off, ballistic drop, feet-first landing."""
        from .falls import feet_first_landing

        self.move(0.4, phase, y=self.pose.pos[1] - 0.10, pitch=np.deg2rad(15), flex=np.deg2rad([5, 4, 3]))
        stop = float(self.rng.uniform(0.25, 0.4))  # planned landing: deep knee flexion
        feet_first_landing(self, height, forward_speed=float(self.rng.uniform(0.5, 1.0)), stop_distance=stop,
                           takeoff_up=float(self.rng.uniform(0.3, 0.8)), phase=phase, label="jump")
        self.stand_up(0.8, phase)

    def lying_activity(self, duration: float, phase: str = "lying_work", amp_deg: float = 5.0) -> None:
        """Conscious work while lying (mechanic under a vehicle, crawl-space work): arm-driven
        trunk movements every few seconds with short pauses (sin^2 bumps, no velocity steps)."""
        remaining = duration
        while remaining > 0.5:
            seg = self.seg(min(float(self.rng.uniform(1.5, 4.0)), remaining), phase, fidget=2.0)
            shape = np.sin(np.pi * np.linspace(0, 1, seg.k)) ** 2
            seg.flex = seg.flex + np.outer(shape, np.deg2rad(self.rng.uniform(-amp_deg, amp_deg, 3)))
            seg.lat = seg.lat + np.outer(shape, np.deg2rad(self.rng.uniform(-amp_deg, amp_deg, 3)) * 0.6)
            seg.pos[:, 1] += 0.01 * shape * self.rng.uniform(0.2, 1.0)
            self.commit(seg)
            remaining -= seg.k * self.dt
            if remaining > 0.5:
                pause = self.seg(min(float(self.rng.uniform(1.0, 6.0)), remaining), phase, fidget=1.5)
                self.commit(pause)
                remaining -= pause.k * self.dt

    def remove_strip(self, table_height: float = 0.75) -> None:
        """End of shift: take the strip off and leave it on a table (no breathing, no motion)."""
        self.move(2.0, "activity", twist=np.deg2rad([5, 8, 10]), lat=np.deg2rad([2, 3, 4]))
        self.event("strip_removed")
        s = self.seg(3.0, "device_off", fidget=0.0)
        s.worn[:] = False
        s.breath_amp[:] = 0
        s.sway[:] = 0
        w = minjerk(s.k)
        target_pitch = np.deg2rad(-90 if self.rng.random() < 0.5 else 90)
        s.pitch = self.pose.pitch + (target_pitch - self.pose.pitch) * w
        s.flex = self.pose.flex * (1 - w)[:, None]
        s.lat = self.pose.lat * (1 - w)[:, None]
        s.twist = self.pose.twist * (1 - w)[:, None]
        s.pos[:, 1] = self.pose.pos[1] + (self.ground_y + table_height - self.pose.pos[1]) * w + 0.15 * np.sin(np.pi * w)
        s.pos += np.outer(0.4 * w, forward_vec(self.pose.heading))
        self.commit(s)

    def strap_slip(self, slot: int, degrees: float) -> None:
        """The strap shifts on one anchor: a step change in that sensor's mounting rotation."""
        axis = self.rng.normal(size=3)
        axis[1] *= 0.3
        axis /= np.linalg.norm(axis)
        self.mount_steps.append((self.t, slot, axis * np.deg2rad(degrees)))
        self.event("strap_slip", slot=slot, degrees=degrees)

    def hat_off(self, phase: str = "activity") -> None:
        """Hard hat knocked off: cable tug on the upper-thoracic anchor (1-5 g, <50 ms)."""
        s = self.seg(1.0, phase, sway=1.0)
        g_peak = self.rng.uniform(1.0, 5.0)
        pulse = smooth_pulse(self.fs, 25.0, 0.008) * g_peak * G / self.worker.spacings()[2]
        s.flex[100:100 + len(pulse), 2] += pulse
        rot = self.rng.uniform(0.5, 2.0)  # rad/s peak, brief twist of the upper segment
        width = int(0.3 * self.fs)
        # sin^2 bump: zero rate at both ends, peak rate = rot.
        s.twist[100:100 + width, 2] += rot * 0.3 / np.pi * np.sin(np.pi * np.arange(width) / width) ** 2
        self.commit(s)
        self.event("hat_off", peak_g=g_peak)

    # --- posture primitives (used by posture.py; existing primitives are unchanged) -------
    def posture_hold(self, trunk_deg: float, hold_s: float, lat_deg: float = 0.0, twist_deg: float = 0.0,
                     pelvis_share: float = 0.45, down_s: float = 1.5, drift_deg: float = 3.0, sway: float = 0.4,
                     phase: str = "activity", return_up: bool = True) -> None:
        """Move into a trunk posture and hold it, with slow natural drift, then (optionally) stand.

        trunk_deg is the upper-thoracic inclination from upright (forward +): pelvis_share of it at
        the hip (lumbo-pelvic rhythm), the rest as spinal flexion. lat_deg / twist_deg are side bend
        (right +) and axial twist, mostly in the spine. Drift is +/-drift_deg (RMS) at < 0.2 Hz.
        """
        trunk = np.deg2rad(trunk_deg)
        hip = trunk * pelvis_share
        lat, twist = np.deg2rad(lat_deg), np.deg2rad(twist_deg)
        drop = 0.03 + 0.05 * max(np.sin(hip), 0.0)
        self.move(down_s, phase, y=self.ground_y + self.worker.pelvis_height - drop, pitch=hip, roll=0.3 * lat,
                  flex=(trunk - hip) * np.array([0.45, 0.35, 0.20]), lat=0.7 * lat * np.array([0.3, 0.35, 0.35]),
                  twist=twist * np.array([0.2, 0.4, 0.4]))
        s = self.seg(hold_s, phase, sway=sway)
        if drift_deg > 0:
            env = taper(s.k, self.fs, 1.0)
            for arr, share in ((s.flex, np.array([0.4, 0.35, 0.25])), (s.lat, np.array([0.3, 0.35, 0.35]))):
                arr += np.outer(np.deg2rad(drift_deg) * band_noise(self.rng, s.k, self.fs, None, 0.2) * env, share)
        self.commit(s)
        if return_up:
            self.stand_up(1.5, phase)

    def functional_calibration(self, bends: int = 3, depth_deg: float = 40.0) -> None:
        """Three slow forward bends after the neutral stand (spine-model note, Q2 step 3): they
        reveal each sensor's sagittal plane so mounting rotation about the spine can be removed."""
        self.event("functional_cal_start")
        for _ in range(bends):
            self.posture_hold(depth_deg, 1.0, down_s=1.8, drift_deg=0.0, sway=0.2)
            self.hold(1.0, sway=0.5)
        self.event("functional_cal_end")

    def sit_posture(self, duration: float, slouch_deg: float = 0.0, seat_height: float = 0.45,
                    phase: str = "activity") -> None:
        """Sit on a bench or seat; slouch_deg > 0 tilts the pelvis back and flexes the lumbar and
        thoracic spine (a slumped spinal curve with a near-upright trunk)."""
        self.sit(seat_height, phase=phase)
        slouch = np.deg2rad(slouch_deg)
        self.move(2.0, phase, pitch=-0.35 * slouch, flex=slouch * np.array([0.55, 0.45, 0.35]))
        s = self.seg(duration, phase, sway=0.2)
        env = taper(s.k, self.fs, 1.0)
        s.flex += np.outer(np.deg2rad(2.0) * band_noise(self.rng, s.k, self.fs, None, 0.15) * env, [0.4, 0.35, 0.25])
        self.commit(s)
        self.move(1.5, phase, pitch=np.deg2rad(-5), flex=np.deg2rad([6, 4, 2]))
        self.stand_up(1.5, phase)

    # --- finish --------------------------------------------------------------------------
    def build(self) -> PoseTrack:
        cat = lambda name: np.concatenate([getattr(s, name) for s in self.segs])
        pos, heading, pitch, roll = cat("pos"), cat("heading"), cat("pitch"), cat("roll")
        flex, lat, twist = cat("flex"), cat("lat"), cat("twist")
        phase, worn = cat("phase"), cat("worn")
        breath_amp, breath_hz, fidget, sway = cat("breath_amp"), cat("breath_hz"), cat("fidget"), cat("sway")
        n = len(pos)
        # Overlay envelopes change per segment; smooth them (1 Hz) so switching sway, fidget or
        # breathing on/off never creates a position step (which would read as a fake impact).
        env_sos = signal.butter(2, 1.0, fs=self.fs, output="sos")
        if n > 100:
            breath_amp = np.clip(signal.sosfiltfilt(env_sos, breath_amp), 0, None)
            fidget = np.clip(signal.sosfiltfilt(env_sos, fidget), 0, None)
            sway = np.clip(signal.sosfiltfilt(env_sos, sway), 0, None)
        # Breathing: thoracic/upper flexion oscillation (~ +/-0.01 g projection) and 1 mm bob.
        breath = np.sin(2 * np.pi * np.cumsum(breath_hz) * self.dt)
        flex[:, 1] += 0.5 * breath_amp * breath
        flex[:, 2] += breath_amp * breath
        pos[:, 1] += 0.001 * breath * (breath_amp > 0)
        # Postural sway while standing: slow < 0.5 Hz drift of pelvis position and lean.
        for axis in (0, 2):
            pos[:, axis] += 0.004 * sway * band_noise(self.rng, n, self.fs, None, 0.4)
        pitch += np.deg2rad(0.4) * sway * band_noise(self.rng, n, self.fs, None, 0.4)
        roll += np.deg2rad(0.3) * sway * band_noise(self.rng, n, self.fs, None, 0.4)
        # Fidget: small involuntary joint motion of a conscious wearer.
        for arr in (flex, lat, twist):
            for j in range(3):
                arr[:, j] += np.deg2rad(0.12) * fidget * band_noise(self.rng, n, self.fs, None, 2.0)
        # Light zero-phase smoothing removes numerical kinks at segment joins (keeps impacts < 120 Hz).
        sos = signal.butter(4, 120.0, fs=self.fs, output="sos")
        smooth = lambda x: signal.sosfiltfilt(sos, x, axis=0) if len(x) > 60 else x
        pos, heading, pitch, roll = smooth(pos), smooth(heading), smooth(pitch), smooth(roll)
        flex, lat, twist = smooth(flex), smooth(lat), smooth(twist)
        mount_delta = np.zeros((n, 4, 3))
        for t0, slot, rotvec in self.mount_steps:
            start = int(t0 * self.fs)
            ramp_len = int(0.3 * self.fs)
            w = np.ones(n - start)
            w[: min(ramp_len, n - start)] = minjerk(min(ramp_len, n - start))
            mount_delta[start:, slot] += np.outer(w, rotvec)
        track = PoseTrack(self.fs, pos, heading, pitch, roll, flex, lat, twist, phase, worn, mount_delta)
        clamp_joints(track)
        return track
