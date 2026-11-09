# File: spine.py
# Purpose: Kinematic spine model (4 rigid links: pelvis, lumbar, thoracic, upper thoracic),
#          Hermite spine curve and virtual IMU readings at the 4 strip anchors.
# Contributors: Unassigned
"""Spine kinematics from research_notes/.../spine_model_from_4_imus.md.

Frame: right-handed, +X wearer-right, +Y up, +Z posterior (protocol/ORIENTATION.md).
Pelvis orientation R_root = Ry(heading) * Rx(-pitch) * Rz(-roll): pitch > 0 leans the top
forward (towards -Z), roll > 0 leans it to the wearer's right. Joint rotations use the
ISB spine order mapped to this frame, intrinsic X-Z'-Y'': flexion (forward +), lateral
bend (right +), axial twist. Anchor i+1 = anchor i + L_i * (segment i long axis), the
note's chain recipe; the drawn curve between anchors is visual interpolation only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation

from .model_api import GRAVITY

GRAVITY_WORLD = np.array([0.0, -GRAVITY, 0.0])
UP = np.array([0.0, 1.0, 0.0])

# Physiological range of motion per inter-segment joint (rad). Whole-lumbar flexion is
# shared by two joints in this model; these per-joint limits are DESIGN ESTIMATES.
JOINT_FLEX_RANGE = (np.deg2rad(-15.0), np.deg2rad(32.0))
JOINT_LAT_RANGE = np.deg2rad(15.0)
JOINT_TWIST_RANGE = np.deg2rad(15.0)
MAX_VOLUNTARY_RATE = np.deg2rad(180.0)  # trunk rate cap for voluntary moves (design estimate)


@dataclass(frozen=True)
class Worker:
    """Body and strip geometry. Spacings should be tape-measured at setup when possible."""

    height_m: float = 1.75
    mass_kg: float = 80.0
    load_kg: float = 8.0  # tool belt, harness and PPE
    spacing_m: tuple[float, float, float] | None = None  # pelvis->lumbar->thoracic->upper thoracic
    # Sensor pitch at neutral from spinal curvature (lordosis/kyphosis); calibration zeroes it.
    neutral_curvature_deg: tuple[float, float, float, float] = (-12.0, -6.0, 10.0, 18.0)
    mount_error_deg: float = 6.0  # random strap misalignment per sensor (SIM_SPEC 2: +/-5-10 deg) [design]
    sensor_offset_m: float = 0.04  # IMU sits 3-5 cm posterior to the spine line (SIM_SPEC 0) [design]
    # Optional body geometry (added for a body-model adapter, e.g. Anny; None = the old formulas).
    # pelvis_height_m: pelvis-anchor height standing (default 0.53 x height).
    # com_to_pelvis_m: whole-body centre of mass -> pelvis anchor in the segment frame (+X right, +Y up,
    #   +Z posterior), used by the fall model (default (0, -0.03 x height, 0.10)).
    pelvis_height_m: float | None = None
    com_to_pelvis_m: tuple[float, float, float] | None = None

    def spacings(self) -> np.ndarray:
        if self.spacing_m is not None:
            return np.asarray(self.spacing_m, float)
        # Fractions of body height: DESIGN ESTIMATES (S1-L3, L3-T10, T10-T2 surface distances).
        return np.array([0.07, 0.09, 0.11]) * self.height_m

    @property
    def pelvis_height(self) -> float:
        if self.pelvis_height_m is not None:
            return float(self.pelvis_height_m)
        return 0.53 * self.height_m  # sacral sensor height standing (design estimate)

    def com_to_pelvis(self) -> np.ndarray:
        """Centre of mass -> pelvis anchor, segment frame, metres (fall model)."""
        if self.com_to_pelvis_m is not None:
            return np.asarray(self.com_to_pelvis_m, float)
        return np.array([0.0, -0.03 * self.height_m, 0.10])  # [design]

    @property
    def total_mass(self) -> float:
        return self.mass_kg + self.load_kg


@dataclass
class PoseTrack:
    """Body pose sampled at fs (1 kHz). Angles in radians, positions in metres."""

    fs: float
    root_pos: np.ndarray        # (N,3) pelvis anchor position
    heading: np.ndarray         # (N,)
    pitch: np.ndarray           # (N,) whole-pelvis forward lean
    roll: np.ndarray            # (N,) whole-pelvis right lean
    flex: np.ndarray            # (N,3) joint flexion: pelvis-lumbar, lumbar-thoracic, thoracic-upper
    lat: np.ndarray             # (N,3) joint lateral bend
    twist: np.ndarray           # (N,3) joint axial twist
    phase: np.ndarray           # (N,) int ground-truth phase code, see PHASES
    worn: np.ndarray            # (N,) bool, False once the strip is taken off
    mount_delta: np.ndarray = field(default=None)  # (N,4,3) extra mount rotation vectors (strap slip)

    def __len__(self) -> int:
        return int(self.root_pos.shape[0])

    @property
    def t(self) -> np.ndarray:
        return np.arange(len(self)) / self.fs


# Ground-truth phase codes stored per sample (also exported for model developers).
PHASES = {
    0: "activity", 1: "calibration_stand", 2: "fall_descent", 3: "impact", 4: "lying",
    5: "suspended", 6: "recovery", 7: "near_fall", 8: "lying_work", 9: "device_off",
    10: "free_fall", 11: "collapse",
}
PHASE_CODE = {name: code for code, name in PHASES.items()}


def root_rotation(heading: np.ndarray, pitch: np.ndarray, roll: np.ndarray) -> Rotation:
    n = len(heading)
    zeros = np.zeros(n)
    ry = Rotation.from_rotvec(np.stack([zeros, heading, zeros], axis=1))
    rx = Rotation.from_rotvec(np.stack([-pitch, zeros, zeros], axis=1))
    rz = Rotation.from_rotvec(np.stack([zeros, zeros, -roll], axis=1))
    return ry * rx * rz


def joint_rotation(flex: np.ndarray, lat: np.ndarray, twist: np.ndarray) -> Rotation:
    return Rotation.from_euler("XZY", np.stack([-flex, -lat, twist], axis=1))


def soft_limit(x: np.ndarray, lo: float, hi: float, knee: float = np.deg2rad(3.0)) -> np.ndarray:
    """Identity inside [lo+knee, hi-knee], exponential approach to the limit outside.

    C1-continuous, so a joint reaching its range of motion decelerates instead of stopping
    dead (a hard clip would create a velocity step, i.e. a fake acceleration spike).
    """
    y = np.array(x, float, copy=True)
    top = hi - knee
    above = y > top
    y[above] = hi - knee * np.exp(-(y[above] - top) / knee)
    bottom = lo + knee
    below = y < bottom
    y[below] = lo + knee * np.exp((y[below] - bottom) / knee)
    return y


def clamp_joints(track: PoseTrack) -> None:
    """Enforce physiological joint ranges in place (smoothly)."""
    track.flex[:] = soft_limit(track.flex, *JOINT_FLEX_RANGE)
    track.lat[:] = soft_limit(track.lat, -JOINT_LAT_RANGE, JOINT_LAT_RANGE)
    track.twist[:] = soft_limit(track.twist, -JOINT_TWIST_RANGE, JOINT_TWIST_RANGE)


@dataclass
class StripKinematics:
    """Segment orientations and anchor positions for every sample."""

    seg_rot: list[Rotation]        # 4 Rotations of length N (segment -> world)
    anchors: np.ndarray            # (N,4,3) anchor positions
    chip_rot: list[Rotation]       # 4 Rotations of length N (chip -> world)


def strip_kinematics(track: PoseTrack, worker: Worker, mounts: list[Rotation]) -> StripKinematics:
    n = len(track)
    spacing = worker.spacings()
    seg = [root_rotation(track.heading, track.pitch, track.roll)]
    for joint in range(3):
        seg.append(seg[-1] * joint_rotation(track.flex[:, joint], track.lat[:, joint], track.twist[:, joint]))
    spine_line = np.empty((n, 4, 3))
    spine_line[:, 0] = track.root_pos
    for i in range(1, 4):
        spine_line[:, i] = spine_line[:, i - 1] + spacing[i - 1] * seg[i - 1].apply(UP)
    # Sensors sit on the skin, posterior to the spine line, so rotation gives them lever arms.
    anchors = spine_line + np.stack([seg[i].apply([0.0, 0.0, worker.sensor_offset_m]) for i in range(4)], axis=1)
    chips = []
    for i in range(4):
        curvature = Rotation.from_rotvec([-np.deg2rad(worker.neutral_curvature_deg[i]), 0, 0])
        chip = seg[i] * curvature * mounts[i]
        if track.mount_delta is not None:
            chip = chip * Rotation.from_rotvec(track.mount_delta[:, i])
        chips.append(chip)
    return StripKinematics(seg, anchors, chips)


def random_mounts(worker: Worker, rng: np.random.Generator) -> list[Rotation]:
    mounts = []
    for _ in range(4):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        mounts.append(Rotation.from_rotvec(axis * np.deg2rad(rng.normal(0, worker.mount_error_deg))))
    return mounts


def virtual_imu(kin: StripKinematics, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """True chip-frame specific force (m/s^2) and angular rate (rad/s), each (N,4,3)."""
    n = kin.anchors.shape[0]
    dt = 1.0 / fs
    accel_world = np.gradient(np.gradient(kin.anchors, dt, axis=0), dt, axis=0)
    acc = np.empty((n, 4, 3))
    gyro = np.zeros((n, 4, 3))
    for i, chip in enumerate(kin.chip_rot):
        specific_force = accel_world[:, i] - GRAVITY_WORLD  # what an accelerometer measures
        acc[:, i] = chip.inv().apply(specific_force)
        if n >= 3:
            # Body-frame rate from the central difference R(t-dt)^-1 R(t+dt).
            rel = chip[:-2].inv() * chip[2:]
            gyro[1:-1, i] = rel.as_rotvec() / (2 * dt)
            gyro[0, i], gyro[-1, i] = gyro[1, i], gyro[-2, i]
    return acc, gyro


def apply_soft_tissue(acc: np.ndarray, gyro: np.ndarray, fs: float, fn_hz: np.ndarray, zeta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """IMU on a 2nd-order mount over the bone (SIM_SPEC 1.5): base-excited absolute response
    H(s) = (2 zeta w s + w^2) / (s^2 + 2 zeta w s + w^2), per anchor and axis. DC gain 1, so
    gravity and slow posture pass unchanged; sharp impacts are smoothed and can ring."""
    from scipy import signal

    acc_out, gyro_out = np.empty_like(acc), np.empty_like(gyro)
    for slot in range(acc.shape[1]):
        w = 2 * np.pi * fn_hz[slot]
        b, a = signal.bilinear([2 * zeta[slot] * w, w * w], [1.0, 2 * zeta[slot] * w, w * w], fs)
        zi = signal.lfilter_zi(b, a)
        for data, out in ((acc, acc_out), (gyro, gyro_out)):
            x = data[:, slot]
            out[:, slot] = signal.lfilter(b, a, x, axis=0, zi=zi[:, None] * x[0][None])[0]
    return acc_out, gyro_out


def spine_curve(anchors: np.ndarray, seg_axes: np.ndarray, points: int = 40) -> np.ndarray:
    """Cubic Hermite curve through the 4 anchors of ONE frame (visual interpolation only).

    anchors (4,3); seg_axes (4,3) unit segment long axes. Hermite fits because the IMUs
    measure orientation (the curve's tangent), not shape between anchors.
    """
    chord = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(anchors, axis=0), axis=1))])
    spline = CubicHermiteSpline(chord, anchors, seg_axes, axis=0)  # unit tangents per unit chord length
    return spline(np.linspace(0, chord[-1], points))
