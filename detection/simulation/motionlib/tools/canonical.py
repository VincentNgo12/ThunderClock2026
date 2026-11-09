# File: canonical.py
# Purpose: Stage S2 of the motion library: convert downloaded clips (ASF/AMC, BVH, CML JSON,
#          NIOSH CSV) into one canonical format, motionlib/canonical/<clip>.npz, with a data card
#          in motionlib/cards/<clip>.json. Adds the four trunk segment orientations in the
#          simulator's segment convention so the simulated trunk can be compared with real data.
# Contributors: Unassigned
"""Run from detection/simulation (numpy/scipy only; downloads are parsed, never executed):

    .venv/bin/python -I motionlib/tools/canonical.py            # every clip
    .venv/bin/python -I motionlib/tools/canonical.py cmu_115_01 # clips whose id starts with this

Canonical clip (all float32 unless noted), 100 Hz, metres, +Y up, right-handed:
  joints (N,20,3)        CANON_JOINTS: rig.JOINTS (18) + pelvis (hip-centre midpoint) + head_top
  valid (20,) bool       False where the source has no such joint (the value is then NaN)
  trunk_anchors (N,4,3)  S1, L3, T10 and T2 points on the source spine line (simulator anchor order)
  seg_q (N,4,4)          segment -> world quaternions (x,y,z,w), simulator convention, IMU-like:
                         the rigid source bone under each anchor, neutral-calibrated so that every
                         segment equals Ry(heading) during the clip's neutral stand (as the
                         simulator's calibration stand has pitch = flex = 0)
  seg_q_chord (N,4,4)    geometric chord frames, not calibrated: +Y from anchor i to anchor i+1
                         (T2 -> neck base for the last), +X wearer-right, +Z posterior. Rebuilds
                         the anchors with the simulator's recipe p(i+1) = p(i) + L_i * Y_i.
  trunk_angles (N,12)    from seg_q, decomposed exactly like spine.root_rotation / joint_rotation:
                         heading, pitch, roll, flex x3, lat x3, twist x3 (rad)
  trunk_angles_chord (N,12)  the same from the chord frames after the same neutral calibration
  neutral (2,) int       frame window [k0, k1) used for the neutral calibration
  fs, source_fps, source_frames, height_est_m; clip, joint_map (json str)
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent))   # our own tools only, never raw/
from readers import asf_amc_motion, read_bvh, read_cml_json, read_csv_table, resample_motion  # noqa: E402

MOTIONLIB = Path(__file__).resolve().parents[1]
RAW = MOTIONLIB / "raw"
OUT_DIR = MOTIONLIB / "canonical"
CARD_DIR = MOTIONLIB / "cards"
FS = 100.0
UP = np.array([0.0, 1.0, 0.0])

RIG_JOINTS = ("hip_l", "hip_r", "knee_l", "knee_r", "ankle_l", "ankle_r", "heel_l", "heel_r", "toe_l", "toe_r",
              "shoulder_l", "shoulder_r", "elbow_l", "elbow_r", "hand_l", "hand_r", "neck", "head")   # = rig.JOINTS
CANON_JOINTS = RIG_JOINTS + ("pelvis", "head_top")
TRUNK_ANGLE_NAMES = ("heading", "pitch", "roll", "flex_pl", "flex_lt", "flex_tu", "lat_pl", "lat_lt", "lat_tu",
                     "twist_pl", "twist_lt", "twist_tu")

# Where the simulator's anchors sit along the spine line from the hip-centre midpoint (0) to the
# neck base (1). From the simulator's design geometry, in units of body height H: hip joints
# 0.02 H below S1 (rig.Dims.hip), S1-L3-T10-T2 spacings 0.07/0.09/0.11 H (spine.Worker.spacings),
# neck base 0.03 H above T2 (rig.Dims.neck). [design]
_CHAIN = np.array([0.02, 0.07, 0.09, 0.11, 0.03])
ANCHOR_FRACTIONS = np.cumsum(_CHAIN)[:4] / _CHAIN.sum()

# Foot and hand geometry for joints the sources lack, as fractions of body height (rig.Dims). [design]
ANKLE_H, HEEL_BACK, TOE_FWD, HAND_EXTRA = 0.039, 0.055 / 1.75, 0.2 / 1.75, 0.045 / 1.75
LEG_FRACTION = 0.475          # (thigh + shank) / H in rig.Dims; used to estimate the source height

NEUTRAL_WINDOW_S = 0.5
NEUTRAL_MAX_SPEED = 0.10      # m/s pelvis speed for a "standing" window [design]
NEUTRAL_MIN_KNEE_DEG = 145.0  # knee inner angle for a "standing" window (relaxed stances read 145-160) [design]
NEUTRAL_MAX_TILT_DEG = 15.0   # pelvis -> neck base tilt from vertical for a "standing" window [design]
REF_MAX_PIECE_DIFF_M = 0.01   # a subject reference must have the same spine pieces within 1 cm [design]


@dataclass
class Landmarks:
    """Source joints in metres, +Y up, at the source rate. Missing optional joints are None."""

    fps: float
    points: dict[str, np.ndarray]                    # name -> (N,3)
    spine: list[np.ndarray]                          # polyline hip_mid ... neck_base, each (N,3)
    spine_rot: list[np.ndarray] | None = None        # rigid rotation (N,3,3) of each polyline piece
    joint_map: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    root_relative: bool = False
    source_fps: float | None = None                  # set when the adapter already resampled to FS
    source_frames: int | None = None
    subject: str | None = None                       # same performer and skeleton (neutral reference)


# =============================================================================================
# Source adapters: each returns Landmarks
# =============================================================================================
Y_UP_FROM_Z_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])   # (x,y,z) -> (x,z,-y), a rotation


def cmu_landmarks(asf_path: Path, amc_path: Path, fps: float, subject: str | None = None) -> Landmarks:
    source, unit = asf_amc_motion(asf_path.read_text(errors="replace"), amc_path.read_text(errors="replace"), fps)
    motion = resample_motion(source, FS)     # rigid resampling: bone lengths stay exact
    scale = (1.0 / unit) * 0.0254            # CMU FAQ: lengths are inches / 0.45
    pos = lambda name, which="start": getattr(motion, which)[:, motion.index(name)] * scale  # noqa: E731
    rot = lambda name: motion.rot[:, motion.index(name)]                                       # noqa: E731
    points = {}
    for side, s in (("l", "l"), ("r", "r")):
        points[f"hip_{side}"] = pos(f"{s}femur")
        points[f"knee_{side}"] = pos(f"{s}tibia")
        points[f"ankle_{side}"] = pos(f"{s}foot")
        points[f"ball_{side}"] = pos(f"{s}toes")
        points[f"toe_{side}"] = pos(f"{s}toes", "end")
        points[f"shoulder_{side}"] = pos(f"{s}humerus")
        points[f"elbow_{side}"] = pos(f"{s}radius")
        points[f"wrist_{side}"] = pos(f"{s}wrist")
        points[f"hand_end_{side}"] = pos(f"{s}hand", "end")
    points["neck_base"] = pos("lowerneck")
    points["head"] = 0.5 * (pos("head") + pos("head", "end"))
    points["head_top"] = pos("head", "end")
    hip_mid = 0.5 * (points["hip_l"] + points["hip_r"])
    spine = [hip_mid, pos("root"), pos("lowerback", "end"), pos("upperback", "end"), pos("thorax", "end")]
    spine_rot = [rot("root"), rot("lowerback"), rot("upperback"), rot("thorax")]
    joint_map = {"hip": "femur start", "knee": "tibia start", "ankle": "foot start", "toe": "toes end (tip)",
                 "heel": "derived from ankle, toe and knee (rig.Dims proportions)", "shoulder": "humerus start",
                 "elbow": "radius start", "hand": "midpoint of wrist and hand end", "neck": "lowerneck start (neck base)",
                 "head": "midpoint of head bone", "head_top": "head end", "pelvis": "midpoint of femur starts",
                 "spine": "hip mid -> root -> lowerback -> upperback -> thorax end; bones root/lowerback/upperback/thorax"}
    return Landmarks(FS, points, spine, spine_rot, joint_map, source_fps=fps, source_frames=len(source.start),
                     subject=subject)


def _bvh_name(names: list[str], *candidates: str) -> str:
    for name in candidates:
        if name in names:
            return name
    raise KeyError(f"none of {candidates} in the BVH")


def bvh_landmarks(text: str, scale: float, subject: str | None = None) -> Landmarks:
    """ACCAD (Hips/ToSpine/Spine/Spine1, LeftArm...) and 100STYLE (Hips/Chest..Chest4, LeftShoulder...)."""
    source = read_bvh(text)
    motion = resample_motion(source, FS)     # rigid resampling: bone lengths stay exact
    names = motion.names
    pos = lambda name, which="start": getattr(motion, which)[:, motion.index(name)] * scale  # noqa: E731
    arm_style = "LeftArm" in names   # ACCAD: LeftShoulder is the clavicle, LeftArm the humerus
    points = {}
    for side, s in (("l", "Left"), ("r", "Right")):
        points[f"hip_{side}"] = pos(_bvh_name(names, f"{s}UpLeg", f"{s}Hip"))
        points[f"knee_{side}"] = pos(_bvh_name(names, f"{s}Leg", f"{s}Knee"))
        points[f"ankle_{side}"] = pos(_bvh_name(names, f"{s}Foot", f"{s}Ankle"))
        toe = _bvh_name(names, f"{s}ToeBase", f"{s}Toe")
        points[f"ball_{side}"] = pos(toe)
        points[f"toe_{side}"] = pos(toe, "end")
        if arm_style:
            shoulder, elbow, wrist = f"{s}Arm", f"{s}ForeArm", f"{s}Hand"
        else:
            shoulder, elbow, wrist = f"{s}Shoulder", f"{s}Elbow", f"{s}Wrist"
        points[f"shoulder_{side}"] = pos(shoulder)
        points[f"elbow_{side}"] = pos(elbow)
        points[f"wrist_{side}"] = pos(wrist)
        points[f"hand_end_{side}"] = pos(wrist, "end")
    neck = names.index("Neck")
    points["neck_base"] = pos("Neck")
    points["head"] = 0.5 * (pos("Head") + pos("Head", "end"))
    points["head_top"] = pos("Head", "end")
    # Spine joints from the root up to (not including) the neck.
    chain = []
    j = motion.parent[neck]
    while j >= 0:
        chain.append(j)
        j = motion.parent[j]
    chain.reverse()
    hip_mid = 0.5 * (points["hip_l"] + points["hip_r"])
    spine = [hip_mid] + [motion.start[:, j] * scale for j in chain] + [points["neck_base"]]
    # The piece from joint j to its child is rigid with j; hip_mid -> root is rigid with the root.
    spine_rot = [motion.rot[:, chain[0]]] + [motion.rot[:, j] for j in chain]
    joint_map = {"hip": "UpLeg/Hip", "knee": "Leg/Knee", "ankle": "Foot/Ankle", "toe": "ToeBase/Toe end site",
                 "heel": "derived from ankle, toe and knee (rig.Dims proportions)",
                 "shoulder": "Arm" if arm_style else "Shoulder", "elbow": "ForeArm" if arm_style else "Elbow",
                 "hand": "midpoint of wrist joint and its end site", "neck": "Neck joint (neck base)",
                 "head": "midpoint of Head joint and end site", "head_top": "Head end site",
                 "pelvis": "midpoint of hip joints", "spine": "hip mid -> " + " -> ".join(names[j] for j in chain) + " -> Neck"}
    return Landmarks(FS, points, spine, spine_rot, joint_map, source_fps=source.fps, source_frames=len(source.start),
                     subject=subject)


CML_NAMES = {"hip": "hip", "knee": "knee", "ankle": "ankle", "toe": "foot", "shoulder": "shoulder", "elbow": "elbow",
             "wrist": "wrist", "hand_end": "hand"}


def cml_landmarks(text: str) -> tuple[Landmarks, dict]:
    """CML 20-joint Kinect-style skeleton, Z up in the file, 30 Hz [CML paper]."""
    joints, meta = read_cml_json(text)
    convert = lambda name: joints[name] @ Y_UP_FROM_Z_UP.T  # noqa: E731
    points = {}
    for side, s in (("l", "Left"), ("r", "Right")):
        for key, part in CML_NAMES.items():
            points[f"{key}_{side}"] = convert(s + part)
    points["neck_base"] = convert("Shouldercentre")
    points["head"] = convert("Head")
    hip_mid = 0.5 * (points["hip_l"] + points["hip_r"])
    spine = [hip_mid, convert("Hipcentre"), convert("Spine"), points["neck_base"]]
    joint_map = {"toe": "Foot (Kinect foot point, used as toe)", "heel": "derived (rig.Dims proportions)",
                 "hand": "midpoint of Wrist and Hand", "neck": "Shouldercentre", "head": "Head",
                 "head_top": "not available", "spine": "hip mid -> Hipcentre -> Spine -> Shouldercentre (3 points)"}
    # CML's skeleton alignment leaves lengths non-metric (in-lab thighs read ~0.86 m, CMU-derived
    # ones ~0.35 m), so the clip is scaled to a nominal 1.75 m body. Angles are unaffected. [design]
    leg = np.median(np.linalg.norm(points["hip_l"] - points["knee_l"], axis=1)) + \
        np.median(np.linalg.norm(points["knee_l"] - points["ankle_l"], axis=1))
    scale = LEG_FRACTION * 1.75 / leg
    points = {k: v * scale for k, v in points.items()}
    spine = [p * scale for p in spine]
    lm = Landmarks(30.0, points, spine, None, joint_map,
                   ["CML 20-joint skeleton at 30 Hz: trunk segments come from a 3-point spine (coarse).",
                    f"CML lengths are not metric; scaled by {scale:.3f} to a nominal 1.75 m body (positions are "
                    "not true metres, angles are)."])
    return lm, meta


NIOSH_JOINTS = {"hip_r": "r_hip", "hip_l": "l_hip", "knee_r": "r_knee", "knee_l": "l_knee", "ankle_r": "r_foot",
                "ankle_l": "l_foot", "shoulder_r": "r_shoulder", "shoulder_l": "l_shoulder", "elbow_r": "r_elbow",
                "elbow_l": "l_elbow", "wrist_r": "r_wrist", "wrist_l": "l_wrist", "neck_base": "thorax",
                "head_top": "head", "hip_centre": "hip", "spine_mid": "spine"}


def niosh_trials(text: str) -> dict[tuple[int, int], Landmarks]:
    """NIOSH RD-1042 3D joint positions (OpenSim from Vicon markers), 'Ground Truth' rows only.
    Human3.6M-style joints, mm, Z up, pelvis-relative, 100 Hz [NIOSH data dictionary]."""
    header, table = read_csv_table(text)
    source_col = header.index("Source")
    # read_csv_table turns the text 'Source' column into NaN; re-read it from the raw lines.
    sources = [line.split(",")[source_col].strip() for line in text.splitlines()[1:] if line.strip()]
    rows = np.array([s == "Ground Truth" for s in sources[:len(table)]])
    col = {h: i for i, h in enumerate(header)}
    trials = {}
    for subject in np.unique(table[rows, col["Subject"]]):
        for trial in np.unique(table[rows, col["Trial"]]):
            mask = rows & (table[:, col["Subject"]] == subject) & (table[:, col["Trial"]] == trial)
            if mask.sum() < 50:
                continue
            block = table[mask]
            block = block[np.argsort(block[:, col["Frame"]])]

            def joint(name: str) -> np.ndarray:
                xyz = np.stack([block[:, col[f"{name} {a} (mm)"]] for a in "xyz"], axis=1) / 1000.0
                return xyz @ Y_UP_FROM_Z_UP.T

            points = {key: joint(name) for key, name in NIOSH_JOINTS.items()}
            neck, top = points["neck_base"], points["head_top"]
            points["head"] = neck + 0.6 * (top - neck)          # head centre between neck base and head top [design]
            hip_mid = 0.5 * (points["hip_l"] + points["hip_r"])
            spine = [hip_mid, points.pop("hip_centre"), points.pop("spine_mid"), neck]
            joint_map = {"ankle": "r_foot/l_foot (Human3.6M foot = ankle)", "toe": "not available",
                         "heel": "not available", "hand": "wrist extended along the forearm by 0.045 m x s [design]",
                         "neck": "thorax", "head_top": "head", "head": "60 % from thorax to head [design]",
                         "spine": "hip mid -> hip -> spine -> thorax (3 points)"}
            trials[(int(subject), int(trial))] = Landmarks(
                100.0, points, spine, None, joint_map,
                ["NIOSH roofing: shingle installation on a roof mock-up; positions are pelvis-relative "
                 "(no global translation) and the roof slope is unknown, so feet and floor are not usable."],
                root_relative=True)
    return trials


# =============================================================================================
# Common conversion
# =============================================================================================
def resample_points(x: np.ndarray, fps: float, t_out: np.ndarray) -> np.ndarray:
    t_in = np.arange(len(x)) / fps
    return CubicSpline(t_in, x, axis=0)(t_out)


def resample_rot(r: np.ndarray, fps: float, t_out: np.ndarray) -> np.ndarray:
    t_in = np.arange(len(r)) / fps
    return Slerp(t_in, Rotation.from_matrix(r))(t_out).as_matrix()


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def frame_from_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(N,3,3) rotation with +Y along y and +X as close to x as possible; Z = X x Y (posterior)."""
    y = unit(y)
    x = unit(x - y * np.sum(x * y, axis=-1, keepdims=True))
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=-1)


def knee_angle(hip, knee, ankle) -> np.ndarray:
    a, b = unit(hip - knee), unit(ankle - knee)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1, 1)))


def polyline_points(spine: list[np.ndarray], fractions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Points at arc-length fractions of a per-frame polyline. Returns (N,F,3) and the piece index (F,)
    each fraction falls on in the median frame (so a point's owner bone never switches)."""
    pts = np.stack(spine, axis=1)                                    # (N,P,3)
    seg_len = np.linalg.norm(np.diff(pts, axis=1), axis=-1)          # (N,P-1)
    cum = np.concatenate([np.zeros((len(pts), 1)), np.cumsum(seg_len, axis=1)], axis=1)
    out = np.empty((len(pts), len(fractions), 3))
    for f, frac in enumerate(fractions):
        s = frac * cum[:, -1]
        piece = np.clip((cum[:, 1:] < s[:, None]).sum(axis=1), 0, pts.shape[1] - 2)
        rows = np.arange(len(pts))
        w = (s - cum[rows, piece]) / np.maximum(seg_len[rows, piece], 1e-9)
        out[:, f] = pts[rows, piece] + w[:, None] * (pts[rows, piece + 1] - pts[rows, piece])
    median_cum = np.median(cum, axis=0)
    owners = np.array([int(np.clip(np.searchsorted(median_cum, frac * median_cum[-1]) - 1, 0, pts.shape[1] - 2))
                       for frac in fractions])
    return out, owners


def find_neutral(points: dict, n: int) -> tuple[int, int, str, dict]:
    """The clip's most upright, stillest 0.5 s: standing still if possible, else a fallback."""
    win = int(NEUTRAL_WINDOW_S * FS)
    hip_mid = 0.5 * (points["hip_l"] + points["hip_r"])
    tilt = np.degrees(np.arccos(np.clip(unit(points["neck_base"] - hip_mid)[:, 1], -1, 1)))
    vel = np.gradient(hip_mid, 1.0 / FS, axis=0)
    speed = np.linalg.norm(vel[:, [0, 2]], axis=1)
    knees = np.minimum(knee_angle(points["hip_l"], points["knee_l"], points["ankle_l"]),
                       knee_angle(points["hip_r"], points["knee_r"], points["ankle_r"]))
    standing = (speed < NEUTRAL_MAX_SPEED) & (knees > NEUTRAL_MIN_KNEE_DEG) & (tilt < NEUTRAL_MAX_TILT_DEG)
    score = tilt + 10.0 * speed       # fallback: prefer upright (walking) frames over slow, bent ones
    kernel = np.ones(win) / win
    if n < win:
        return 0, n, "whole clip (shorter than the window)", {}
    mean_score = np.convolve(score, kernel, mode="valid")
    all_standing = np.convolve(standing.astype(float), kernel, mode="valid") > 0.999
    method = "standing still"
    if all_standing.any():
        mean_score = np.where(all_standing, mean_score, np.inf)
    else:
        method = "fallback: no still standing window, most upright and slowest 0.5 s"
    k0 = int(np.argmin(mean_score))
    info = {"tilt_deg": round(float(tilt[k0:k0 + win].mean()), 2), "speed_mps": round(float(speed[k0:k0 + win].mean()), 3),
            "knee_deg": round(float(knees[k0:k0 + win].mean()), 1)}
    return k0, k0 + win, method, info


def heading_of_right(right: np.ndarray) -> float:
    """Simulator heading whose Ry(heading) maps +X onto this horizontal right vector."""
    return float(np.arctan2(-right[2], right[0]))


def sim_angles(seg: list[Rotation]) -> np.ndarray:
    """Inverse of spine.root_rotation (Ry(h) Rx(-pitch) Rz(-roll)) and spine.joint_rotation
    (intrinsic XZY of -flex, -lat, twist). Returns (N,12) in TRUNK_ANGLE_NAMES order."""
    root = seg[0].as_euler("YXZ")
    out = np.zeros((len(root), 12))
    out[:, 0], out[:, 1], out[:, 2] = root[:, 0], -root[:, 1], -root[:, 2]
    for j in range(3):
        rel = (seg[j].inv() * seg[j + 1]).as_euler("XZY")
        out[:, 3 + j], out[:, 6 + j], out[:, 9 + j] = -rel[:, 0], -rel[:, 1], rel[:, 2]
    return out


def window_means(frames: list[np.ndarray], k0: int, k1: int) -> list[Rotation]:
    return [Rotation.from_matrix(f[k0:k1]).mean() for f in frames]


def calibrate(frames: list[np.ndarray], neutral: list[Rotation], heading: float) -> list[Rotation]:
    """R'(t) = R(t) R(neutral)^-1 Ry(heading): every segment reads Ry(heading) in the neutral stand."""
    ry = Rotation.from_rotvec([0.0, heading, 0.0])
    return [Rotation.from_matrix(f) * mean.inv() * ry for f, mean in zip(frames, neutral)]


def build_clip(lm: Landmarks, reference: dict | None = None) -> dict:
    """reference: another clip's neutral stand of the same subject and skeleton (see main), used
    when this clip has no still standing window of its own."""
    n_in = len(lm.spine[0])
    duration = (n_in - 1) / lm.fps
    n_src = lm.source_frames if lm.source_frames is not None else n_in
    source_fps = lm.source_fps if lm.source_fps is not None else lm.fps
    t_out = np.minimum(np.arange(int(np.floor(duration * FS + 1e-3)) + 1) / FS, duration)
    n = len(t_out)
    pts = {k: resample_points(v, lm.fps, t_out) for k, v in lm.points.items()}
    spine = [resample_points(p, lm.fps, t_out) for p in lm.spine]
    spine_rot = [resample_rot(r, lm.fps, t_out) for r in lm.spine_rot] if lm.spine_rot is not None else None

    leg = np.median(np.linalg.norm(pts["hip_l"] - pts["knee_l"], axis=1)) + \
        np.median(np.linalg.norm(pts["knee_l"] - pts["ankle_l"], axis=1))
    height = float(leg / LEG_FRACTION)

    joints = np.full((n, len(CANON_JOINTS), 3), np.nan)
    valid = np.zeros(len(CANON_JOINTS), bool)

    def put(name: str, value: np.ndarray | None) -> None:
        if value is not None:
            joints[:, CANON_JOINTS.index(name)] = value
            valid[CANON_JOINTS.index(name)] = True

    for side in ("l", "r"):
        hip, knee, ankle = pts[f"hip_{side}"], pts[f"knee_{side}"], pts[f"ankle_{side}"]
        put(f"hip_{side}", hip)
        put(f"knee_{side}", knee)
        put(f"ankle_{side}", ankle)
        toe = pts.get(f"toe_{side}")
        if toe is not None:
            # Foot frame: u heel->toe along the foot, n the sole normal in the shank-foot plane.
            u = unit(toe - ankle)
            shank = knee - ankle
            normal = unit(shank - u * np.sum(shank * u, axis=1, keepdims=True))
            u_flat = unit(np.cross(np.cross(normal, u), normal))
            sole = ankle - normal * (ANKLE_H * height)
            put(f"heel_{side}", sole - u_flat * (HEEL_BACK * height))
            put(f"toe_{side}", toe)
        put(f"shoulder_{side}", pts[f"shoulder_{side}"])
        put(f"elbow_{side}", pts[f"elbow_{side}"])
        wrist = pts[f"wrist_{side}"]
        hand_end = pts.get(f"hand_end_{side}")
        if hand_end is not None:
            put(f"hand_{side}", 0.5 * (wrist + hand_end))
        else:
            put(f"hand_{side}", wrist + unit(wrist - pts[f"elbow_{side}"]) * (HAND_EXTRA * height))
    put("neck", pts["neck_base"])
    put("head", pts["head"])
    put("pelvis", 0.5 * (pts["hip_l"] + pts["hip_r"]))
    put("head_top", pts.get("head_top"))

    # --- trunk: anchors, neutral window, segment frames ----------------------------------------
    anchors, owners = polyline_points(spine, ANCHOR_FRACTIONS)
    owners[0] = 0                                    # S1 is always the pelvis
    pieces = np.median(np.linalg.norm(np.diff(np.stack(spine, axis=1), axis=1), axis=-1), axis=0)
    k0, k1, neutral_method, neutral_info = find_neutral(pts, n)
    right_neutral = (pts["hip_r"] - pts["hip_l"])[k0:k1].mean(axis=0)
    right_neutral[1] = 0.0
    heading = heading_of_right(unit(right_neutral))
    ends = [anchors[:, 1], anchors[:, 2], anchors[:, 3], pts["neck_base"]]
    chord_y = [ends[i] - anchors[:, i] for i in range(4)]

    use_ref = (reference is not None and neutral_method != "standing still" and spine_rot is not None
               and reference["owners"] == [int(o) for o in owners]
               and np.abs(np.array(reference["pieces"]) - pieces).max() < REF_MAX_PIECE_DIFF_M)
    if use_ref:
        neutral_method = f"subject reference: standing still in {reference['clip']}"
        neutral_info = dict(reference["info"])
        heading = reference["heading"]
    if spine_rot is not None:
        bone = [spine_rot[owners[i]] for i in range(4)]
        bone_neutral = ([Rotation.from_quat(q) for q in reference["bone_q"]] if use_ref else window_means(bone, k0, k1))
        seg = calibrate(bone, bone_neutral, heading)
        lateral = [s.apply([1.0, 0.0, 0.0]) for s in seg]
        trunk_method = "rigid source bone under each anchor (" + ", ".join(str(o) for o in owners) + " = spine piece index)"
    else:
        lat_pelvis = pts["hip_r"] - pts["hip_l"]
        lat_top = pts["shoulder_r"] - pts["shoulder_l"]
        lateral = [unit((1 - w) * unit(lat_pelvis) + w * unit(lat_top)) for w in (0.0, 1 / 3, 2 / 3, 1.0)]
        frames = [frame_from_xy(lateral[i], chord_y[i]) for i in range(4)]
        seg = calibrate(frames, window_means(frames, k0, k1), heading)
        trunk_method = "positions: chord directions with hip-line to shoulder-line lateral axes blended [design]"
    chord = [frame_from_xy(lateral[i], chord_y[i]) for i in range(4)]
    chord_neutral = ([Rotation.from_quat(q) for q in reference["chord_q"]] if use_ref else window_means(chord, k0, k1))
    chord_cal = calibrate(chord, chord_neutral, heading)
    # What another clip of this subject needs to borrow this clip's neutral stand.
    own_reference = None
    if neutral_method == "standing still" and spine_rot is not None:
        own_reference = {"owners": [int(o) for o in owners], "pieces": [float(x) for x in pieces], "heading": heading,
                         "info": neutral_info, "bone_q": [m.as_quat().tolist() for m in window_means(bone, k0, k1)],
                         "chord_q": [m.as_quat().tolist() for m in window_means(chord, k0, k1)]}

    # Handedness: the toes (or, without toes, the bent knees) must point anterior, i.e. along -Z of
    # the body frame (X = hip line, Y = pelvis -> neck base), which also holds lying or crawling.
    hip_mid = 0.5 * (pts["hip_l"] + pts["hip_r"])
    posterior = unit(np.cross(unit(pts["hip_r"] - pts["hip_l"]), unit(pts["neck_base"] - hip_mid)))
    if pts.get("toe_l") is not None:
        forward = 0.5 * (pts["toe_l"] + pts["toe_r"]) - 0.5 * (pts["ankle_l"] + pts["ankle_r"])
    else:
        forward = 0.5 * (pts["knee_l"] + pts["knee_r"]) - 0.25 * (pts["hip_l"] + pts["hip_r"] + pts["ankle_l"] + pts["ankle_r"])
    anterior_share = float(np.mean(np.sum(forward * posterior, axis=1) < 0))

    return {
        "joints": joints.astype(np.float32), "valid": valid, "trunk_anchors": anchors.astype(np.float32),
        "seg_q": np.stack([s.as_quat() for s in seg], axis=1).astype(np.float32),
        "seg_q_chord": np.stack([Rotation.from_matrix(c).as_quat() for c in chord], axis=1).astype(np.float32),
        "trunk_angles": sim_angles(seg).astype(np.float32),
        "trunk_angles_chord": sim_angles(chord_cal).astype(np.float32),
        "neutral": np.array([k0, k1]), "neutral_from_reference": np.bool_(use_ref), "fs": np.float32(FS), "source_fps": np.float32(source_fps),
        "source_frames": np.int64(n_src), "height_est_m": np.float32(height),
        "joint_names": np.array(CANON_JOINTS), "trunk_angle_names": np.array(TRUNK_ANGLE_NAMES),
        "_meta": {"neutral_method": neutral_method, "neutral_info": neutral_info, "trunk_method": trunk_method,
                  "anterior_share": round(anterior_share, 3), "spine_points": len(lm.spine),
                  "seconds": round(float(duration), 3), "frames": int(n), "own_reference": own_reference},
    }


# =============================================================================================
# Clip list and driver
# =============================================================================================
ACCAD_MEMBERS = {
    "Male1_A1_Stand.bvh": ["idle"], "Male1_A2_Sway.bvh": ["idle"], "Male1_A4_LookAround.bvh": ["idle"],
    "Male1_A5_PickUpBox.bvh": ["pickup"], "Male1_A6_LiftBox.bvh": ["pickup", "carry"], "Male1_A7_Crouch.bvh": ["squat"],
    "Male1_A8_CrouchToLie.bvh": ["lie"], "Male1_A9_LieDown.bvh": ["lie"], "Male1_A10_LieToCrouch.bvh": ["get_up"],
    "Male1_A11_CrawlForward.bvh": ["crawl"], "Male1_B3_Walk.bvh": ["walk"],
    "Male1_B19_WalkToPickUpBox_Working.bvh": ["walk", "pickup"], "Male1_B20_WalkWithBox.bvh": ["walk", "carry", "load"],
    "Male1_B21_PutDownBoxToWalk.bvh": ["putdown", "walk"], "Male1_B24_WalkToCrouch.bvh": ["walk", "squat"],
    "Male1_B25_CrouchToWalk.bvh": ["squat", "walk"],
    "Female1_A01_Stand.bvh": ["idle"], "Female1_A05_PickUpBox.bvh": ["pickup"], "Female1_A06_LiftBox.bvh": ["pickup", "carry"],
    "Female1_A07_Crouch.bvh": ["squat"], "Female1_B03_Walk1.bvh": ["walk"],
    "Female1_B19_WalkToPickUpBox.bvh": ["walk", "pickup"], "Female1_B20_WalkWithBox.bvh": ["walk", "carry", "load"],
    "Female1_B21_PutDownBoxToWalk.bvh": ["putdown", "walk"],
}
CMU_60HZ_SUBJECTS = {62, 79}     # mocap.cs.cmu.edu motion index lists these subjects at 60 fps


def sources() -> dict:
    return json.loads((MOTIONLIB / "sources.json").read_text())["sources"]


def checked_bytes(path: Path, sha256: str) -> bytes:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"{path.name}: sha256 does not match sources.json")
    return data


def clip_jobs():
    """Yield (clip id, source key, file entry, extra card fields, landmark builder)."""
    src = sources()
    cmu_files = {Path(e["path"]).name: e for e in src["cmu"]["files"]}
    for name, entry in cmu_files.items():
        if not name.endswith(".amc") or entry.get("duplicate_of"):
            continue          # skeletons, and trials the CMU site lists twice under two subjects
        subject, trial = name[:-4].split("_")
        asf = cmu_files[f"{subject}.asf"]
        fps = 60.0 if int(subject) in CMU_60HZ_SUBJECTS else 120.0

        def build(entry=entry, asf=asf, fps=fps):
            checked_bytes(MOTIONLIB / asf["path"], asf["sha256"])
            checked_bytes(MOTIONLIB / entry["path"], entry["sha256"])
            return cmu_landmarks(MOTIONLIB / asf["path"], MOTIONLIB / entry["path"], fps, subject="cmu:" + asf["sha256"][:16])

        yield f"cmu_{int(subject):02d}_{trial}", "cmu", entry, {"tags": entry.get("tags", []), "asf_sha256": asf["sha256"]}, build

    for entry in src["accad"]["files"]:
        archive = MOTIONLIB / entry["path"]
        for member, tags in ACCAD_MEMBERS.items():
            if not member.startswith(Path(entry["path"]).name.split("_")[0] + "_"):
                continue

            def build(entry=entry, member=member):
                data = checked_bytes(MOTIONLIB / entry["path"], entry["sha256"])
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    text = zf.read(member).decode("latin-1")
                return bvh_landmarks(text, 0.01, subject="accad:" + member.split("_")[0])

            clip = "accad_" + re.sub(r"[^a-z0-9]+", "_", member[:-4].lower())
            yield clip, "accad", entry, {"tags": tags, "member": member}, build

    for entry in src["100style"]["files"]:
        if not entry["path"].endswith(".bvh"):
            continue

        def build(entry=entry):
            text = checked_bytes(MOTIONLIB / entry["path"], entry["sha256"]).decode("latin-1")
            return bvh_landmarks(text, 0.01, subject="100style")

        clip = "100style_" + Path(entry["path"]).stem.lower()
        yield clip, "100style", entry, {"tags": entry.get("tags", []), "member": entry.get("member")}, build

    for entry in src["cml"]["files"]:
        text = checked_bytes(MOTIONLIB / entry["path"], entry["sha256"]).decode("utf-8")
        head = json.loads(text)
        if head.get("data source") != "In-lab experiment data":
            continue   # CMU-derived CML samples duplicate originals we already convert from ASF/AMC

        def build(text=text):
            return cml_landmarks(text)

        clip = "cml_" + Path(entry["path"]).stem.lower()
        yield clip, "cml", entry, {"tags": entry.get("tags", []), "member": entry.get("member"),
                                   "upstream": head.get("data source"), "original_label": head.get("original label")}, build

    for entry in src["niosh_roofing"]["files"]:
        if "rd-1042" not in entry["path"]:
            continue     # rd-1062 holds hip joint angles only: reference data for step 1, no clip
        data = checked_bytes(MOTIONLIB / entry["path"], entry["sha256"])
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            member = next(m for m in zf.namelist() if m.endswith("Dataset-3DJointPositions_roofing.csv"))
            trials = niosh_trials(zf.read(member).decode("utf-8", "replace"))
        for (subject, trial), lm in sorted(trials.items()):
            yield (f"niosh_s{subject}_t{trial}", "niosh_roofing", entry,
                   {"tags": ["kneel", "kneel_work"], "member": member, "subject": subject, "trial": trial},
                   lambda lm=lm: lm)


def write_clip(clip: str, source_key: str, entry: dict, extra: dict, lm, cml_meta: dict | None,
               reference: dict | None = None) -> tuple[dict, dict | None]:
    data = build_clip(lm, reference)
    meta = data.pop("_meta")
    joint_map = dict(lm.joint_map)
    np.savez_compressed(OUT_DIR / f"{clip}.npz", clip=np.array(clip), joint_map=np.array(json.dumps(joint_map)), **data)
    src = sources()[source_key]
    problems = []
    if meta["anterior_share"] < 0.7:
        problems.append(f"handedness: feet/knees point anterior in only {meta['anterior_share']:.0%} of frames")
    if bool(data["neutral_from_reference"]):
        lm.notes.append(f"No still standing window in this clip: seg_q and trunk_angles are calibrated to the "
                        f"neutral stand of {reference['clip']} (same performer and skeleton), so 'neutral' frames "
                        "are not zero-angle frames of this clip.")
    elif meta["neutral_info"].get("tilt_deg", 0.0) > 15.0:
        lm.notes.append("No upright neutral window in this clip (trunk tilt > 15 deg): seg_q and trunk_angles are "
                        "calibrated to a non-neutral posture; use seg_q_chord or another clip of the subject.")
    if not 1.4 <= float(data["height_est_m"]) <= 2.1:
        problems.append(f"estimated height {float(data['height_est_m']):.2f} m is implausible")
    card = {
        "clip": clip, "source": source_key, "source_name": src["name"],
        "file": entry["path"], "url": entry["url"], "sha256": entry["sha256"],
        "licence": {"class": src["class"], "name": src["licence"], "url": src["licence_url"], "credit": src["credit"]},
        "tags": extra.get("tags", []),
        "source_fps": float(data["source_fps"]), "source_frames": int(data["source_frames"]), "seconds": meta["seconds"],
        "fs": FS, "frames": meta["frames"], "height_est_m": round(float(data["height_est_m"]), 3),
        "valid_joints": [n for n, v in zip(CANON_JOINTS, data["valid"]) if v],
        "joint_map": joint_map, "root_relative": lm.root_relative,
        "trunk": {"method": meta["trunk_method"], "spine_points": meta["spine_points"],
                  "neutral_frames": [int(x) for x in data["neutral"]], "neutral_method": meta["neutral_method"],
                  "neutral_ok": meta["neutral_info"].get("tilt_deg", 0.0) <= 15.0,
                  "neutral_reference": reference["clip"] if bool(data["neutral_from_reference"]) else None,
                  **meta["neutral_info"]},
        "anterior_share": meta["anterior_share"], "notes": lm.notes, "problems": problems,
        "status": "canonical" if not problems else "rejected", "gates": {}, "reviewed_by": None,
    }
    for key in ("member", "asf_sha256", "subject", "trial", "upstream", "original_label"):
        if extra.get(key) is not None:
            card[key] = extra[key]
    if cml_meta:
        card["cml"] = cml_meta
    if lm.subject:
        card["subject"] = lm.subject
    (CARD_DIR / f"{clip}.json").write_text(json.dumps(card, indent=2) + "\n")
    own = meta["own_reference"]
    if own is not None:
        own = dict(own, clip=clip)
    return card, own


def main() -> None:
    """Pass 1 converts every clip and collects each subject's best still standing window. Pass 2
    re-converts the clips without a standing window of their own against that subject reference."""
    prefix = sys.argv[1] if len(sys.argv) > 1 else ""
    OUT_DIR.mkdir(exist_ok=True)
    CARD_DIR.mkdir(exist_ok=True)
    index_path = OUT_DIR / "index.json"
    references_path = OUT_DIR / "references.json"
    index = json.loads(index_path.read_text()) if prefix and index_path.exists() else {}
    references = json.loads(references_path.read_text()) if references_path.exists() else {}
    failures = 0
    needs_reference = []

    def convert(clip, source_key, entry, extra, build, reference=None):
        result = build()
        lm, cml_meta = result if isinstance(result, tuple) else (result, None)
        card, own = write_clip(clip, source_key, entry, extra, lm, cml_meta, reference)
        index[clip] = {"source": source_key, "tags": card["tags"], "seconds": card["seconds"], "status": card["status"],
                       "licence_class": card["licence"]["class"], "neutral": card["trunk"]["neutral_method"]}
        return lm, card, own

    for job in clip_jobs():
        clip = job[0]
        if not clip.startswith(prefix):
            continue
        try:
            lm, card, own = convert(*job)
        except Exception as exc:  # report and continue with the next clip
            print(f"{clip}: FAILED ({type(exc).__name__}: {exc})")
            failures += 1
            continue
        if own is not None and lm.subject:
            best = references.get(lm.subject)
            if best is None or best["clip"] == clip or own["info"]["tilt_deg"] < best["info"]["tilt_deg"]:
                references[lm.subject] = own
        elif lm.subject:
            needs_reference.append((job, lm.subject))
        report(clip, card)

    for job, subject in needs_reference:
        if subject not in references:
            continue
        try:
            _, card, _ = convert(*job, reference=references[subject])
        except Exception as exc:
            print(f"{job[0]}: FAILED with reference ({type(exc).__name__}: {exc})")
            failures += 1
            continue
        report(job[0], card)

    references_path.write_text(json.dumps(dict(sorted(references.items())), indent=1) + "\n")
    index_path.write_text(json.dumps(dict(sorted(index.items())), indent=1) + "\n")
    print(f"{len(index)} clips in index, {failures} failure(s)")
    sys.exit(1 if failures else 0)


def report(clip: str, card: dict) -> None:
    flag = "" if card["status"] == "canonical" else "  REJECTED: " + "; ".join(card["problems"])
    print(f"{clip}: {card['seconds']:.1f} s @ {card['source_fps']:.0f} Hz, H~{card['height_est_m']:.2f} m, "
          f"neutral {card['trunk']['neutral_method'][:40]} tilt {card['trunk'].get('tilt_deg', '-')}{flag}")


if __name__ == "__main__":
    main()
