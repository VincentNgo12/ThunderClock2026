# File: geom.py
# Purpose: Small vector helpers shared by the viewer's scene and rig builders.
# Contributors: Unassigned
"""World frame = simulator frame: metres, right-handed, +Y up. A heading h faces
fwd(h) = (-sin h, 0, -cos h), which is also three.js's convention for rotation.y = h on an
object modelled facing -Z, so yaws pass to the browser unchanged."""

from __future__ import annotations

import numpy as np

UP = np.array([0.0, 1.0, 0.0])


def fwd(heading) -> np.ndarray:
    h = np.asarray(heading, float)
    return np.stack([-np.sin(h), np.zeros_like(h), -np.cos(h)], axis=-1)


def right(heading) -> np.ndarray:
    h = np.asarray(heading, float)
    return np.stack([np.cos(h), np.zeros_like(h), -np.sin(h)], axis=-1)


def yaw_of(direction) -> float:
    """Heading whose fwd() is this horizontal direction."""
    d = np.asarray(direction, float)
    return float(np.arctan2(-d[0], -d[2]))


def flat(v) -> np.ndarray:
    out = np.array(v, float, copy=True)
    out[..., 1] = 0.0
    return out


def unit(v, axis: int = -1) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, 1e-9)


def perp(d) -> np.ndarray:
    """Horizontal vector 90 deg clockwise from d seen from above (right of a walker facing d)."""
    d = np.asarray(d, float)
    return np.array([-d[2], 0.0, d[0]])


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def r3(v) -> list:
    return [round(float(x), 3) for x in v]
