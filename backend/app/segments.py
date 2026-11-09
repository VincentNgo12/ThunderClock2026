# File: segments.py
# Purpose: Heading-free spine segment model. Each segment's tilt comes only from the direction of gravity
#          in its sensor's own frame (from the fused quaternion), mapped to body axes by that sensor's
#          physical mounting. Heading (rotation about vertical) is never used, so gyro heading drift has no
#          effect.
# Contributors: Unassigned
"""Why: on the 2026-11-08 recording each sensor's fused heading drifted 10-40 deg/min (gyro bias about
1 deg/s) while the fused tilt matched measured gravity to 1-2 deg. Methods that compare headings
(the old frontend mapping, backend joint angles) bent 17 of 48 held segments the wrong way; this model
bent 0 of 48, including holds it was not fitted on.

Frames
* Sensor frame: the quaternion's own axes (firmware protocol axes of that board). Each board is
  mounted differently on the vest, so "up" is a different sensor axis per board.
* Body frame of a segment: +X wearer right, +Y up along the spine, +Z posterior.
* Mounting: 3x3 rows = body X, Y, Z expressed in sensor axes (an orthonormal, right-handed basis).

Per segment
  u = R(q)^-1 · (0, 1, 0)          world up seen from the sensor (heading-free; q maps sensor -> world)
  g = Mounting · u                 world up seen from the segment, in body axes
  segment rotation  S = shortest rotation taking g onto +Y   (no twist, heading aligned with the body)
  long axis a = S · (0, 1, 0);  flexion = atan2(-a_z, a_y) (+ forward), lateral = atan2(a_x, a_y) (+ right),
  inclination = angle between a and vertical.
Plain Python on purpose: the backend has no numpy dependency.
"""
import math

UP = (0.0, 1.0, 0.0)

# Expected tilt of "world up seen from the segment" (body axes) for each calibration movement:
# bending forward moves the top anteriorly, so world up leans posterior (+Z) relative to the segment.
MOVEMENT_TILT = {"forward": (0.0, 0.0, 1.0), "backward": (0.0, 0.0, -1.0),
                 "right": (-1.0, 0.0, 0.0), "left": (1.0, 0.0, 0.0)}
MIN_TILT_DEG = 4.0


# ---------------------------------------------------------------- small vector / quaternion helpers
def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def norm(a):
    return math.sqrt(dot(a, a))


def unit(a):
    n = norm(a)
    if n < 1e-12:
        raise ValueError("zero vector")
    return (a[0] / n, a[1] / n, a[2] / n)


def scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def mean_vector(vectors):
    total = (0.0, 0.0, 0.0)
    for v in vectors:
        total = add(total, v)
    return unit(total)


def rotate(q, v):
    """Rotate vector v by unit quaternion q = (w, x, y, z)."""
    w, x, y, z = q
    t = scale(cross((x, y, z), v), 2.0)
    return add(add(v, scale(t, w)), cross((x, y, z), t))


def conjugate(q):
    return (q[0], -q[1], -q[2], -q[3])


def quat_unit(q):
    n = math.sqrt(sum(c * c for c in q))
    return tuple(c / n for c in q)


def from_to(a, b):
    """Shortest rotation (w, x, y, z) taking unit vector a onto unit vector b."""
    c = dot(a, b)
    if c < -1.0 + 1e-9:                        # opposite: any perpendicular axis, 180 degrees
        axis = cross(a, (1.0, 0.0, 0.0))
        if norm(axis) < 1e-6:
            axis = cross(a, (0.0, 0.0, 1.0))
        axis = unit(axis)
        return (0.0, axis[0], axis[1], axis[2])
    v = cross(a, b)
    return quat_unit((1.0 + c, v[0], v[1], v[2]))


# ---------------------------------------------------------------- model
def world_up_in_sensor(q):
    """q: (w, x, y, z) sensor -> world. Returns the world up direction in sensor axes."""
    return unit(rotate(conjugate(quat_unit(q)), UP))


def apply_mounting(mounting, v):
    return tuple(dot(row, v) for row in mounting)


def segment_state(q, mounting):
    """Heading-free pose of one segment from its sensor quaternion and mounting."""
    g = unit(apply_mounting(mounting, world_up_in_sensor(q)))
    rotation = from_to(g, UP)
    a = unit(rotate(rotation, UP))
    return {
        "rotation": {"w": rotation[0], "x": rotation[1], "y": rotation[2], "z": rotation[3]},
        "flexion_deg": math.degrees(math.atan2(-a[2], a[1])),
        "lateral_deg": math.degrees(math.atan2(a[0], a[1])),
        "inclination_deg": math.degrees(math.acos(max(-1.0, min(1.0, a[1])))),
    }


def fit_mounting(stand_up, bend_up):
    """Mounting of one sensor from held poses.

    stand_up: world-up in sensor axes while standing upright (unit vector).
    bend_up:  {"forward"|"backward"|"right"|"left": world-up in sensor axes while holding that bend}.
    The body up axis is the standing gravity direction; the rotation about it is the least-squares fit
    that makes each bend tilt the expected way (forward alone is enough; more bends make it robust).
    Returns (mounting rows, report).
    """
    b_y = unit(stand_up)
    seed = cross(b_y, (1.0, 0.0, 0.0))
    if norm(seed) < 0.3:
        seed = cross(b_y, (0.0, 0.0, 1.0))
    ref_z = unit(seed)                         # provisional body back, perpendicular to up
    ref_x = cross(b_y, ref_z)                  # provisional body right (Y x Z = X)
    s = c = 0.0
    used = {}
    for movement, up in bend_up.items():
        target = MOVEMENT_TILT[movement]
        h = add(up, scale(b_y, -dot(up, b_y)))  # horizontal (relative to the standing up axis) part
        tilt = math.degrees(math.asin(min(1.0, norm(h))))
        used[movement] = round(tilt, 1)
        if tilt < MIN_TILT_DEG:
            continue
        hx, hz = dot(h, ref_x), dot(h, ref_z)
        s += hz * target[0] - hx * target[2]
        c += hx * target[0] + hz * target[2]
    if s == 0.0 and c == 0.0:
        raise ValueError("no bend tilted this sensor enough to find its forward direction")
    phi = math.atan2(s, c)
    b_x = unit(add(scale(ref_x, math.cos(phi)), scale(ref_z, math.sin(phi))))
    b_z = cross(b_x, b_y)
    mounting = (b_x, b_y, b_z)
    report = {"tilt_deg": used}
    errors = []
    for movement, up in bend_up.items():
        if used[movement] < MIN_TILT_DEG:
            continue
        g = apply_mounting(mounting, up)
        h = unit((g[0], 0.0, g[2]))
        target = MOVEMENT_TILT[movement]
        errors.append(math.degrees(math.acos(max(-1.0, min(1.0, h[0] * target[0] + h[2] * target[2])))))
    report["direction_error_deg"] = round(sum(errors) / len(errors), 1) if errors else None
    return mounting, report


class SegmentModel:
    """Mounting per region; turns a quaternion Sample into heading-free segment poses."""

    def __init__(self, mountings, source="unknown", report=None):
        self.mountings = mountings                # {region: ((bx), (by), (bz))}
        self.source = source
        self.report = report or {}

    def compute(self, sample, regions):
        if sample is None or sample.format != "quaternion":
            return None
        out = {}
        for imu in sample.imus:
            region = regions.get(imu.sensor_id)
            mounting = self.mountings.get(region)
            if region is None or mounting is None or not imu.valid or imu.quaternion is None:
                continue
            out[region] = {key: (round(value, 3) if isinstance(value, float) else
                                 {k: round(v, 6) for k, v in value.items()})
                           for key, value in segment_state(imu.quaternion, mounting).items()}
        return out or None

    def to_json(self):
        return {"source": self.source, "mountings": {r: [list(row) for row in m] for r, m in self.mountings.items()},
                "report": self.report}

    @classmethod
    def from_json(cls, data):
        return cls({r: tuple(tuple(float(v) for v in row) for row in m) for r, m in data["mountings"].items()},
                   data.get("source", "file"), data.get("report"))
