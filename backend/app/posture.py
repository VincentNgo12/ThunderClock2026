# File: posture.py
# Purpose: Capture neutral joint rotations and track sustained relative bending.
# Contributors: Unassigned

import math

from .models import REGIONS, STALE_AFTER_SECONDS

JOINTS = (("lumbar", 1, 2), ("thoracic", 2, 3), ("upper_thoracic", 3, 4))


def conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)


def multiply(a, b):
    # Hamilton product in wire order [w, x, y, z]. Apply b first, then a.
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def unit(q):
    # Wire quaternions can be scaled. Derived rotations use unit length.
    length = math.hypot(*q)
    return tuple(component / length for component in q)


def joint_rotations(sample):
    if sample is None or sample.format != "quaternion" or len(sample.imus) != len(REGIONS):
        return None
    orientations = {imu.sensor_id: imu.quaternion for imu in sample.imus if imu.valid and imu.quaternion is not None}
    if orientations.keys() != REGIONS.keys():
        return None
    # Express the upper segment in the lower segment's frame to cancel common motion.
    return {
        name: unit(multiply(conjugate(orientations[lower]), orientations[upper]))
        for name, lower, upper in JOINTS
    }


def bend_angles(q):
    # Rotate local up (+Y) and project it onto the sagittal and lateral planes.
    # All components have the same scale, which cancels in atan2.
    w, x, y, z = q
    right = 2 * (x * y - w * z)
    up = w * w - x * x + y * y - z * z
    back = 2 * (y * z + w * x)
    return math.degrees(math.atan2(-back, up)), math.degrees(math.atan2(-right, up))


class Posture:
    def __init__(self, warning_deg=20.0, warning_seconds=2.0):
        if not 0 < warning_deg <= 180 or not math.isfinite(warning_seconds) or warning_seconds <= 0:
            raise ValueError("Posture warning angle must be in (0, 180] and hold seconds must be finite and positive")
        self.warning_deg = warning_deg
        self.warning_seconds = warning_seconds
        self.reset()

    def reset(self):
        self.neutral = None
        self.neutral_root = None
        self.captured_at = None
        self.angles = None
        self.joint_deltas = None
        self.alerts = []
        self._bend_started = {}
        self._previous_receipt = None

    @property
    def calibration(self):
        reference = None
        if self.neutral is not None:
            reference = {
                "root_orientation": dict(zip("wxyz", self.neutral_root)),
                "joint_rotations": {name: dict(zip("wxyz", q)) for name, q in self.neutral.items()},
            }
        return {
            "status": "captured" if reference is not None else "uncalibrated",
            "captured_at": self.captured_at,
            "reference": reference,
        }

    def capture(self, sample, received, received_at, now):
        if received is None or received_at is None or now - received > STALE_AFTER_SECONDS:
            raise ValueError("Neutral capture requires a fresh quaternion frame")
        neutral = joint_rotations(sample)
        if neutral is None:
            raise ValueError("Neutral capture requires all four valid IMU orientations")
        root = unit(next(imu.quaternion for imu in sample.imus if imu.sensor_id == 1))

        # Commit the reference only after validation so a failed recapture preserves it.
        self.reset()
        self.neutral = neutral
        self.neutral_root = root
        self.captured_at = received_at.isoformat()
        self.update(sample, received)

    def update(self, sample, received):
        self.angles = None
        self.joint_deltas = None
        self.alerts = []
        if self.neutral is None:
            return
        joints = joint_rotations(sample)
        if joints is None or self._previous_receipt is None or received - self._previous_receipt > STALE_AFTER_SECONDS:
            self._bend_started.clear()
        self._previous_receipt = received
        if joints is None:
            return

        self.angles = {}
        self.joint_deltas = {}
        for name, rotation in joints.items():
            # D = inverse(neutral) * current. A rig recovers the local joint as neutral * D.
            delta = unit(multiply(conjugate(self.neutral[name]), rotation))
            self.joint_deltas[name] = delta
            flexion, lateral = bend_angles(delta)
            self.angles[f"{name}_flexion_deg"] = flexion
            self.angles[f"{name}_lateral_bend_deg"] = lateral
            if max(abs(flexion), abs(lateral)) < self.warning_deg:
                self._bend_started.pop(name, None)
                continue
            # Count consecutive received samples above the threshold, not WS sends.
            duration = received - self._bend_started.setdefault(name, received)
            if duration >= self.warning_seconds:
                self.alerts.append({
                    "severity": "warning",
                    "category": "posture",
                    "region": name,
                    "summary": f"Sustained {name.replace('_', ' ')} bend",
                    "active": True,
                    "sustained_seconds": round(duration, 2),
                })
