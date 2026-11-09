# File: readers.py
# Purpose: Our own parsers for downloaded motion files (ASF/AMC, BVH, CML JSON, NIOSH CSV) and
#          forward kinematics. Downloads are treated as untrusted text: nothing is executed, only
#          numbers and names are read, and every size is checked.
# Contributors: Unassigned
"""Each reader returns plain numpy arrays in the source's own units and axes; canonical.py
converts them to metres, +Y up and 100 Hz.

Forward kinematics results are a `Motion`:
  names        joint names (for ASF/AMC: bone names; a bone's 'start' is its joint)
  start, end   (N, J, 3) joint positions and bone end positions
  rot          (N, J, 3, 3) global rotation of each bone/joint (bone -> world)
  parent       index of each joint's parent (-1 for the root)
  fps          source frame rate
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

MAX_FRAMES = 2_000_000
MAX_JOINTS = 400


@dataclass
class Motion:
    names: list[str]
    parent: list[int]
    start: np.ndarray
    end: np.ndarray
    rot: np.ndarray
    fps: float

    def index(self, name: str) -> int:
        return self.names.index(name)


# --- Acclaim ASF/AMC (CMU) -----------------------------------------------------------------
@dataclass
class AsfBone:
    name: str
    direction: np.ndarray     # unit vector in the global rest frame
    length: float
    axis: np.ndarray          # (3,3) C matrix from the 'axis' angles
    dof: list[str]


def read_asf(text: str) -> tuple[dict, dict, float, list[str]]:
    """Returns (bones by name, children by parent name, length unit, root dof order)."""
    unit = 1.0
    angle_deg = True
    bones: dict[str, AsfBone] = {}
    children: dict[str, list[str]] = {}
    root_order: list[str] = []
    section = None
    current: dict | None = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith(":"):
            section = line.split()[0][1:]
            continue
        parts = line.split()
        if section == "units":
            if parts[0] == "length":
                unit = float(parts[1])
            elif parts[0] == "angle":
                angle_deg = parts[1].lower().startswith("deg")
        elif section == "root" and parts[0] == "order":
            root_order = [p.lower() for p in parts[1:]]
        elif section == "bonedata":
            if parts[0] == "begin":
                current = {"dof": []}
            elif parts[0] == "end" and current is not None:
                bone = AsfBone(current["name"], current["direction"], current["length"], current["axis"], current["dof"])
                bones[bone.name] = bone
                current = None
                if len(bones) > MAX_JOINTS:
                    raise ValueError("ASF: too many bones")
            elif current is not None:
                key = parts[0]
                if key == "name":
                    current["name"] = parts[1]
                elif key == "direction":
                    v = np.array([float(x) for x in parts[1:4]])
                    current["direction"] = v / max(np.linalg.norm(v), 1e-12)
                elif key == "length":
                    current["length"] = float(parts[1])
                elif key == "axis":
                    angles = np.array([float(x) for x in parts[1:4]])
                    order = parts[4].lower() if len(parts) > 4 else "xyz"
                    # Acclaim: C = Rz Ry Rx for order XYZ, i.e. extrinsic rotations in the listed order.
                    current["axis"] = Rotation.from_euler(order, angles, degrees=angle_deg).as_matrix()
                elif key == "dof":
                    current["dof"] = [p.lower() for p in parts[1:]]
        elif section == "hierarchy":
            if parts[0] in ("begin", "end"):
                continue
            children.setdefault(parts[0], []).extend(parts[1:])
    if not angle_deg:
        raise ValueError("ASF: only degree angles are supported")
    return bones, children, unit, root_order


def read_amc(text: str) -> list[dict[str, list[float]]]:
    """One dict per frame: bone name -> its values in dof order."""
    frames: list[dict[str, list[float]]] = []
    current: dict[str, list[float]] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(":"):
            continue
        parts = line.split()
        if len(parts) == 1 and parts[0].isdigit():
            current = {}
            frames.append(current)
            if len(frames) > MAX_FRAMES:
                raise ValueError("AMC: too many frames")
            continue
        if current is None:
            raise ValueError("AMC: values before the first frame number")
        current[parts[0]] = [float(x) for x in parts[1:]]
    return frames


def asf_amc_motion(asf_text: str, amc_text: str, fps: float) -> tuple[Motion, float]:
    """Forward kinematics of an ASF/AMC pair. Returns (motion in source units, length unit)."""
    bones, children, unit, root_order = read_asf(asf_text)
    frames = read_amc(amc_text)
    n = len(frames)
    if n < 2:
        raise ValueError("AMC: fewer than 2 frames")
    order = ["root"]
    parent_of = {"root": None}
    queue = ["root"]
    while queue:
        name = queue.pop(0)
        for child in children.get(name, []):
            if child not in bones or child in parent_of:
                raise ValueError(f"ASF hierarchy: bad or repeated bone {child!r}")
            parent_of[child] = name
            order.append(child)
            queue.append(child)
    index = {name: i for i, name in enumerate(order)}
    J = len(order)
    start = np.zeros((n, J, 3))
    end = np.zeros((n, J, 3))
    rot = np.zeros((n, J, 3, 3))

    def dof_matrix(dof: list[str], values: list[float]) -> np.ndarray:
        angles = {"rx": 0.0, "ry": 0.0, "rz": 0.0}
        for name, value in zip(dof, values):
            if name in angles:
                angles[name] = value
        return Rotation.from_euler("xyz", [angles["rx"], angles["ry"], angles["rz"]], degrees=True).as_matrix()

    for k, frame in enumerate(frames):
        root_vals = frame.get("root")
        if root_vals is None or len(root_vals) != len(root_order):
            raise ValueError(f"AMC frame {k}: root has {0 if root_vals is None else len(root_vals)} values")
        values = dict(zip(root_order, root_vals))
        start[k, 0] = [values.get("tx", 0.0), values.get("ty", 0.0), values.get("tz", 0.0)]
        rot[k, 0] = Rotation.from_euler("xyz", [values.get("rx", 0.0), values.get("ry", 0.0), values.get("rz", 0.0)],
                                        degrees=True).as_matrix()
        end[k, 0] = start[k, 0]
        for name in order[1:]:
            bone = bones[name]
            p = index[parent_of[name]]
            local = dof_matrix(bone.dof, frame.get(name, []))
            rot[k, index[name]] = rot[k, p] @ bone.axis @ local @ bone.axis.T
            start[k, index[name]] = end[k, p]
            end[k, index[name]] = end[k, p] + bone.length * (rot[k, index[name]] @ bone.direction)
    parent = [-1] + [index[parent_of[name]] for name in order[1:]]
    return Motion(order, parent, start, end, rot, fps), unit


# --- BVH ---------------------------------------------------------------------------------------
def read_bvh(text: str) -> Motion:
    """BVH hierarchy + motion. Joint 'end' is the End Site (or the mean of child offsets)."""
    head, sep, tail = text.partition("MOTION")
    if not sep:
        raise ValueError("BVH: no MOTION section")
    tokens = re.findall(r"[{}]|[^\s{}]+", head)
    names: list[str] = []
    parent: list[int] = []
    offsets: list[np.ndarray] = []
    channels: list[list[str]] = []
    end_sites: dict[int, np.ndarray] = {}
    stack: list[int] = []
    pending: tuple[str, int] | None = None   # ("joint"|"end", parent index)
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("ROOT", "JOINT"):
            names.append(tokens[i + 1])
            parent.append(stack[-1] if stack else -1)
            offsets.append(np.zeros(3))
            channels.append([])
            pending = ("joint", len(names) - 1)
            i += 2
            if len(names) > MAX_JOINTS:
                raise ValueError("BVH: too many joints")
            continue
        if tok == "End":
            pending = ("end", stack[-1])
            i += 2
            continue
        if tok == "{":
            stack.append(pending[1] if pending and pending[0] == "joint" else -2 - (pending[1] if pending else 0))
            i += 1
            continue
        if tok == "}":
            stack.pop()
            i += 1
            continue
        if tok == "OFFSET":
            value = np.array([float(x) for x in tokens[i + 1:i + 4]])
            top = stack[-1]
            if top >= 0:
                offsets[top] = value
            else:
                end_sites[-2 - top] = value
            i += 4
            continue
        if tok == "CHANNELS":
            count = int(tokens[i + 1])
            channels[stack[-1]] = [c.lower() for c in tokens[i + 2:i + 2 + count]]
            i += 2 + count
            continue
        i += 1
    lines = [ln for ln in tail.strip().splitlines() if ln.strip()]
    frames_n = int(re.search(r"Frames:\s*(\d+)", lines[0]).group(1))
    frame_time = float(re.search(r"Frame Time:\s*([0-9.eE+-]+)", lines[1]).group(1))
    if not (0 < frames_n <= MAX_FRAMES) or frame_time <= 0:
        raise ValueError("BVH: bad frame count or frame time")
    data = np.array([[float(x) for x in ln.split()] for ln in lines[2:2 + frames_n]])
    total = sum(len(c) for c in channels)
    if data.shape != (frames_n, total):
        raise ValueError(f"BVH: motion data is {data.shape}, expected ({frames_n}, {total})")
    J = len(names)
    start = np.zeros((frames_n, J, 3))
    rot = np.zeros((frames_n, J, 3, 3))
    col = 0
    for j in range(J):
        ch = channels[j]
        block = data[:, col:col + len(ch)]
        col += len(ch)
        position = np.tile(offsets[j], (frames_n, 1))
        rot_names = [c for c in ch if c.endswith("rotation")]
        for axis, c in enumerate(("xposition", "yposition", "zposition")):
            if c in ch:
                position[:, axis] = block[:, ch.index(c)] if parent[j] < 0 else offsets[j][axis] + block[:, ch.index(c)]
        if rot_names:
            order = "".join(c[0].upper() for c in rot_names)       # intrinsic, in channel order
            angles = np.stack([block[:, ch.index(c)] for c in rot_names], axis=1)
            local = Rotation.from_euler(order, angles, degrees=True).as_matrix()
        else:
            local = np.tile(np.eye(3), (frames_n, 1, 1))
        if parent[j] < 0:
            start[:, j] = position
            rot[:, j] = local
        else:
            p = parent[j]
            start[:, j] = start[:, p] + np.einsum("nij,nj->ni", rot[:, p], position)
            rot[:, j] = rot[:, p] @ local
    end = np.zeros_like(start)
    for j in range(J):
        kids = [c for c in range(J) if parent[c] == j]
        if j in end_sites:
            local_end = end_sites[j]
        elif kids:
            local_end = np.mean([offsets[c] for c in kids], axis=0)
        else:
            local_end = np.zeros(3)
        end[:, j] = start[:, j] + np.einsum("nij,j->ni", rot[:, j], local_end)
    return Motion(names, parent, start, end, rot, 1.0 / frame_time)


# --- Resampling that keeps the skeleton rigid ---------------------------------------------------
def resample_motion(motion: Motion, fs: float) -> Motion:
    """Resample a forward-kinematics Motion to fs Hz without breaking bone lengths.

    Interpolating joint positions directly shortens and stretches bones during fast motion.
    Instead: slerp every joint's global rotation, spline the root position, and rebuild the
    positions from each joint's constant offset in its parent's frame (and each bone end's
    constant offset in its own frame). A joint whose offset is not constant (a translation
    channel below the root) has that offset splined instead.
    """
    from scipy.interpolate import CubicSpline
    from scipy.spatial.transform import Slerp

    n_src = motion.start.shape[0]
    t_in = np.arange(n_src) / motion.fps
    n_out = int(np.floor(t_in[-1] * fs + 1e-3)) + 1   # tolerate frame-time rounding
    t_out = np.minimum(np.arange(n_out) / fs, t_in[-1])     # guard the last sample against rounding
    n, J = len(t_out), len(motion.names)
    rot = np.empty((n, J, 3, 3))
    for j in range(J):
        rot[:, j] = Slerp(t_in, Rotation.from_matrix(motion.rot[:, j]))(t_out).as_matrix()
    start = np.empty((n, J, 3))
    end = np.empty((n, J, 3))
    for j in range(J):        # parents come before children in both readers
        p = motion.parent[j]
        if p < 0:
            start[:, j] = CubicSpline(t_in, motion.start[:, j], axis=0)(t_out)
        else:
            local = np.einsum("nji,nj->ni", motion.rot[:, p], motion.start[:, j] - motion.start[:, p])
            if np.abs(local - local[0]).max() < 1e-6 * max(1.0, np.abs(local[0]).max()):
                start[:, j] = start[:, p] + np.einsum("nij,j->ni", rot[:, p], local[0])
            else:
                start[:, j] = start[:, p] + np.einsum("nij,nj->ni", rot[:, p], CubicSpline(t_in, local, axis=0)(t_out))
        tip = np.einsum("nji,nj->ni", motion.rot[:, j], motion.end[:, j] - motion.start[:, j])
        end[:, j] = start[:, j] + np.einsum("nij,j->ni", rot[:, j], np.median(tip, axis=0))
    return Motion(list(motion.names), list(motion.parent), start, end, rot, fs)


# --- CML JSON (Construction Motion Data Library) ----------------------------------------------
def read_cml_json(text: str) -> tuple[dict[str, np.ndarray], dict]:
    """Joint name -> (N,3) positions in metres in the file's axes, plus the metadata fields."""
    data = json.loads(text)
    bdata = data.get("bdata")
    if not isinstance(bdata, dict):
        raise ValueError("CML: no bdata")
    joints = {}
    n = None
    for name, xyz in bdata.items():
        arr = np.stack([np.asarray(xyz[a], float) for a in ("x", "y", "z")], axis=1)
        if n is None:
            n = len(arr)
        if len(arr) != n or n > MAX_FRAMES:
            raise ValueError("CML: joints have different frame counts")
        joints[str(name)] = arr
    meta = {k: data[k] for k in ("data source", "original label", "source file", "label", "frames", "coordinates",
                                 "action type", "calibrated") if k in data}
    return joints, meta


# --- NIOSH CSV -----------------------------------------------------------------------------
def read_csv_table(text: str) -> tuple[list[str], np.ndarray]:
    """Header names and a float matrix (NaN for empty or non-numeric cells)."""
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    rows = []
    for row in reader:
        if not row:
            continue
        values = []
        for cell in row[:len(header)]:
            try:
                values.append(float(cell))
            except ValueError:
                values.append(np.nan)
        values += [np.nan] * (len(header) - len(values))
        rows.append(values)
        if len(rows) > MAX_FRAMES * 10:
            raise ValueError("CSV: too many rows")
    return [h.strip() for h in header], np.array(rows, float)
