# File: cliplib.py
# Purpose: Small helpers shared by stages S3 (annotate.py), S4 (features.py) and S5 (gates.py):
#          load a canonical clip, read joints by name, the smoothed heading frame, joint angles
#          and boolean run detection. numpy/scipy only; reads our own canonical/*.npz, never raw/.
# Contributors: Unassigned

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

MOTIONLIB = Path(__file__).resolve().parents[1]
CANONICAL = MOTIONLIB / "canonical"
ANNOTATIONS = MOTIONLIB / "annotations"
CLIPS = MOTIONLIB / "clips"
CARDS = MOTIONLIB / "cards"
SIDES = ("l", "r")
HEADING_WINDOW_S = 1.1      # heading frame = pelvis heading averaged over about one stride [design]


class Clip:
    """One canonical clip (stage S2 output) with joints as float64 arrays."""

    def __init__(self, path: Path):
        data = np.load(path)
        self.name = str(data["clip"])
        self.fs = float(data["fs"])
        self.names = [str(n) for n in data["joint_names"]]
        self.joints = data["joints"].astype(float)
        self.valid = data["valid"].astype(bool)
        self.seg_q = data["seg_q"].astype(float)
        self.angles = data["trunk_angles"].astype(float)
        self.height = float(data["height_est_m"])
        self.n = len(self.joints)
        self.t = np.arange(self.n) / self.fs

    def j(self, name: str) -> np.ndarray:
        return self.joints[:, self.names.index(name)]

    def has(self, *names: str) -> bool:
        return all(self.valid[self.names.index(n)] for n in names)


def load_clip(name: str) -> Clip:
    return Clip(CANONICAL / f"{name}.npz")


def load_card(name: str) -> dict:
    return json.loads((CARDS / f"{name}.json").read_text())


def save_card(card: dict) -> None:
    (CARDS / f"{card['clip']}.json").write_text(json.dumps(card, indent=2) + "\n")


def clip_names(prefix: str = "") -> list[str]:
    return [p.stem for p in sorted(CANONICAL.glob("*.npz")) if p.stem.startswith(prefix)]


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) frame ranges where a boolean array is True."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    edges = np.flatnonzero(m[1:] != m[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average along axis 0 with edge padding (window in frames, odd or even)."""
    window = max(int(window), 1)
    if window == 1 or len(x) < 2:
        return np.array(x, float, copy=True)
    pad_a, pad_b = window // 2, window - 1 - window // 2
    xp = np.concatenate([np.repeat(x[:1], pad_a, axis=0), x, np.repeat(x[-1:], pad_b, axis=0)], axis=0)
    c = np.cumsum(np.concatenate([np.zeros_like(x[:1]), xp], axis=0), axis=0)
    return (c[window:] - c[:-window]) / window


def smoothed_heading(clip: Clip) -> np.ndarray:
    """Pelvis heading (simulator convention, fwd(h) = (-sin h, 0, -cos h)) averaged over about
    one stride, so the heading frame does not swing with the pelvis yaw of each step."""
    h = np.unwrap(clip.angles[:, 0])
    return moving_average(h, int(round(HEADING_WINDOW_S * clip.fs)))


def heading_frame(heading: np.ndarray) -> np.ndarray:
    """(N,3,3) rotation whose columns are right(h), up and fwd(h): world = R @ local, so a
    local vector reads (x right, y up, z forward)."""
    s, c = np.sin(heading), np.cos(heading)
    zero, one = np.zeros_like(heading), np.ones_like(heading)
    right = np.stack([c, zero, -s], axis=-1)
    up = np.stack([zero, one, zero], axis=-1)
    forward = np.stack([-s, zero, -c], axis=-1)
    return np.stack([right, up, forward], axis=-1)


def to_local(R: np.ndarray, v: np.ndarray) -> np.ndarray:
    """World vectors v (N,3) into the frames R (N,3,3)."""
    return np.einsum("nji,nj->ni", R, v)


def to_world(R: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.einsum("nij,nj->ni", R, v)


def segment_matrix(clip: Clip, index: int) -> np.ndarray:
    """(N,3,3) segment -> world rotation (0 pelvis ... 3 upper thoracic), simulator convention
    (+X wearer-right, +Y up the spine, +Z posterior)."""
    return Rotation.from_quat(clip.seg_q[:, index]).as_matrix()


def inner_angle_deg(a: np.ndarray, mid: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle at `mid` between mid->a and mid->b (degrees); 180 = straight limb."""
    u, v = a - mid, b - mid
    cos = np.sum(u * v, axis=-1) / np.maximum(np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1), 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def bend_pole(root: np.ndarray, mid: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Unit direction of the mid joint away from the root-end line (the pole of two-bone IK).
    Where the limb is almost straight the previous frame's direction is kept."""
    axis = end - root
    axis = axis / np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-12)
    off = mid - root
    off = off - axis * np.sum(off * axis, axis=-1, keepdims=True)
    length = np.linalg.norm(off, axis=-1)
    pole = off / np.maximum(length[:, None], 1e-12)
    good = length > 2e-3
    if not good.any():
        return np.full_like(off, np.nan)
    last = pole[np.argmax(good)]
    for k in range(len(pole)):
        if good[k]:
            last = pole[k]
        else:
            pole[k] = last
    return pole


def trunk_tilt_deg(clip: Clip) -> np.ndarray:
    """Pelvis -> neck base line from vertical (degrees)."""
    v = clip.j("neck") - clip.j("pelvis")
    return np.degrees(np.arccos(np.clip(v[:, 1] / np.maximum(np.linalg.norm(v, axis=1), 1e-12), -1, 1)))
