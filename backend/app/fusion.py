# File: fusion.py
# Purpose: Estimate demo orientations from timestamped raw accel and gyro on the laptop.
# Contributors: Unassigned

import math

from .models import Imu, REGIONS, Sample
from .posture import multiply, unit

GRAVITY_MS2 = 9.80665
STILL_MS = 500
MAX_GAP_MS = 100
SIGNED_AXES = {"x": 1, "y": 2, "z": 3, "-x": -1, "-y": -2, "-z": -3}


def parse_axes(value):
    # Match the firmware convention by default: board forward, left, up to right, up, back.
    groups = ["-y,z,-x"] * len(REGIONS) if value is None else value.split(";")
    if len(groups) != len(REGIONS):
        raise ValueError("RAW_IMU_AXES needs four semicolon-separated axis maps in sensor ID order")
    maps = []
    for group in groups:
        axes = tuple(SIGNED_AXES.get(axis.strip(), 0) for axis in group.split(","))
        order = tuple(abs(axis) for axis in axes)
        if sorted(order) != [1, 2, 3]:
            raise ValueError("Each RAW_IMU_AXES map must use x, y and z once, with optional minus signs")
        # An odd number of swaps or sign flips reflects the frame.
        inversions = sum(left > right for i, left in enumerate(order) for right in order[i + 1:])
        if (inversions + sum(axis < 0 for axis in axes)) % 2:
            raise ValueError("Each RAW_IMU_AXES map must be right-handed")
        maps.append(axes)
    return maps


def remap(vector, axes):
    return tuple(vector[axis - 1] if axis > 0 else -vector[-axis - 1] for axis in axes)


class ImuFusion:
    def __init__(self):
        self.reset()

    def reset(self):
        self.rotation = None
        self.bias = (0.0, 0.0, 0.0)
        self.gyro_sum = (0.0, 0.0, 0.0)
        self.still_ms = 0
        self.still_count = 0

    def update(self, accel, gyro, elapsed_ms):
        magnitude = math.hypot(*accel)
        if magnitude == 0 and not any(gyro):
            # A zero-filled slot can come from a failed sensor read in the raw format.
            self.reset()
            return None
        near_gravity = abs(magnitude - GRAVITY_MS2) <= 1.5
        if self.rotation is None:
            if not near_gravity or math.hypot(*gyro) > 0.15:
                self.reset()
                return None
            if self.still_count:
                self.still_ms += elapsed_ms
            self.still_count += 1
            self.gyro_sum = tuple(total + rate for total, rate in zip(self.gyro_sum, gyro))
            if self.still_ms < STILL_MS:
                return None
            self.bias = tuple(total / self.still_count for total in self.gyro_sum)
            ax, ay, az = (value / magnitude for value in accel)
            # Shortest rotation from measured up to world +Y. Heading is unobserved.
            self.rotation = unit((1 + ay, -az, 0, ax)) if ay > -0.999999 else (0, 1, 0, 0)
            return self.rotation

        gx, gy, gz = (rate - bias for rate, bias in zip(gyro, self.bias))
        if near_gravity:
            ax, ay, az = (value / magnitude for value in accel)
            w, x, y, z = self.rotation
            # Rotate world up into body axes, then use measured × predicted up as tilt feedback.
            ux = 2 * (x * y + w * z)
            uy = 1 - 2 * (x * x + z * z)
            uz = 2 * (y * z - w * x)
            gx += 2 * (ay * uz - az * uy)
            gy += 2 * (az * ux - ax * uz)
            gz += 2 * (ax * uy - ay * ux)

        # Body-frame angular velocity updates a segment-local-to-world quaternion.
        derivative = multiply(self.rotation, (0, gx, gy, gz))
        half_seconds = elapsed_ms / 2000
        self.rotation = unit(tuple(q + half_seconds * dq for q, dq in zip(self.rotation, derivative)))
        return self.rotation


class RawFusion:
    def __init__(self, stats, axes=None):
        self.stats = stats
        self.axes = parse_axes(axes)
        self.filters = [ImuFusion() for _ in REGIONS]
        self.last_time_ms = None
        self.latest = None
        self.received_monotonic = None
        self.received_at = None

    def update(self, sample, received, received_at):
        elapsed_ms = 0
        restart = False
        if self.last_time_ms is not None:
            # Device uptime wraps at 32 bits. Duplicate strips cannot refresh the pose.
            elapsed_ms = (sample.sample_time_ms - self.last_time_ms) & 0xFFFFFFFF
            if elapsed_ms == 0:
                return False
            if elapsed_ms > MAX_GAP_MS:
                restart = True
                elapsed_ms = 0
                for sensor in self.filters:
                    sensor.reset()
        self.last_time_ms = sample.sample_time_ms

        imus = []
        for imu in sample.imus:
            index = imu.sensor_id - 1
            sensor = self.filters[index]
            had_rotation = sensor.rotation is not None
            axes = self.axes[index]
            rotation = sensor.update(remap(imu.accel, axes), remap(imu.gyro, axes), elapsed_ms)
            if had_rotation and rotation is None:
                restart = True
            imus.append(Imu(imu.sensor_id, quaternion=rotation, valid=rotation is not None))
        if restart:
            # Existing posture reset logic discards neutral when the estimator loses its frame.
            self.stats.resets += 1

        self.latest = Sample("quaternion", sample.sample_time_ms, tuple(imus))
        self.received_monotonic = received
        self.received_at = received_at
        return True
