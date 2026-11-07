"""Shared constants: anchors, frame, units and the two firmware (device) profiles.

Frame (Hackathon2026/protocol/ORIENTATION.md, "proposed"): right-handed,
+X wearer-right, +Y up, +Z posterior. All detector inputs are *segment-frame*
vectors, i.e. the sensor axes have already been rotated so that, in the upright
neutral pose, segment +Y points up and segment +Z points backwards.

Units on the detector input (same as firmware/proto/telemetry.proto ImuData):
    accel  m/s^2  (specific force: reads about +9.81 on +Y when standing still)
    gyro   rad/s
"""

# Order matters everywhere: index 0..3 == sensor_id 1..4 in ORIENTATION.md.
ANCHORS = ("pelvis", "lumbar", "thoracic", "upper_thoracic")
N_ANCHORS = 4

FS_HZ = 100.0          # detector sample rate (target firmware: kAcquisitionPeriodMs = 10)
DT_S = 1.0 / FS_HZ
G = 9.80665            # m/s^2 per g

CLASSES = ("normal", "near_fall", "fall_confirmed")

# MPU-6050 configurations (register values from the InvenSense register map).
DEVICE_PROFILES = {
    # what the research report recommends the firmware switch to
    "target": dict(fs_hz=100.0, acc_range_g=16.0, gyro_range_dps=2000.0, dlpf_hz=44.0),
    # what ImuManager.cpp sets today (sampling rate kept at 100 Hz here; the real
    # firmware also samples at only 0.5 Hz, which makes fall detection impossible)
    "current": dict(fs_hz=100.0, acc_range_g=2.0, gyro_range_dps=250.0, dlpf_hz=10.0),
}
