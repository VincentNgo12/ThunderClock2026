# File: protocol.py
# Purpose: Explicit raw-prototype and quaternion decoding; no fusion or format guessing.
# Contributors: Unassigned

import math

from google.protobuf.message import DecodeError

from .generated import orientation_pb2, telemetry_pb2
from .models import Imu, REGIONS, Sample

# Generated nanopb allocation bounds; conformance tests check the C headers.
RAW_MAX_BYTES = 150
QUATERNION_MAX_BYTES = 153
NORM_TOLERANCE = 0.001


class PacketError(ValueError):
    pass


def decode(payload: bytes, packet_format: str) -> Sample:
    if packet_format not in ("raw", "quaternion"):
        raise ValueError("Select raw or quaternion explicitly")
    limit = RAW_MAX_BYTES if packet_format == "raw" else QUATERNION_MAX_BYTES
    if not 0 < len(payload) <= limit:
        raise PacketError(f"Expected one complete {packet_format} notification of 1–{limit} bytes; got {len(payload)}")
    message = telemetry_pb2.ImuStripData() if packet_format == "raw" else orientation_pb2.OrientationPacket()
    try:
        message.ParseFromString(payload)
    except DecodeError as error:
        raise PacketError("Malformed/truncated protobuf; check selected format and notification capacity") from error
    if packet_format == "raw":
        if len(message.data) != 4:
            raise PacketError("Raw prototype requires exactly four sensor records")
        imus = []
        for sensor_id, item in enumerate(message.data, 1):
            if not item.HasField("accel") or not item.HasField("gyro"):
                raise PacketError("Raw record lacks accel/gyro messages; wrong format or incomplete data")
            accel = (item.accel.x, item.accel.y, item.accel.z)
            gyro = (item.gyro.x, item.gyro.y, item.gyro.z)
            if not all(math.isfinite(value) for value in (*accel, *gyro)):
                raise PacketError("Raw values must be finite")
            imus.append(Imu(sensor_id, accel=accel, gyro=gyro))
        return Sample("raw", message.time, tuple(imus))
    if not all(message.HasField(field) for field in ("version", "sequence", "sample_time_ms")):
        raise PacketError("Missing quaternion header. Check that the selected format is quaternion")
    if message.version != 1:
        raise PacketError(f"Unsupported orientation version {message.version}; expected 1")
    if len(message.orientations) != 4:
        raise PacketError("Orientation packet requires exactly four sensors")
    imus = []
    seen = set()
    for item in message.orientations:
        if not item.HasField("sensor_id") or item.sensor_id not in REGIONS or item.sensor_id in seen:
            raise PacketError("Sensor IDs must be exactly 1,2,3,4, each once")
        seen.add(item.sensor_id)
        if not item.HasField("valid"):
            raise PacketError("Each sensor must explicitly state valid=true or false")
        rotation = None
        if item.HasField("rotation"):
            rotation = (item.rotation.w, item.rotation.x, item.rotation.y, item.rotation.z)
            if not all(math.isfinite(value) for value in rotation):
                raise PacketError("Quaternion values must be finite, even for unavailable sensors")
        if not item.valid and rotation is not None:
            raise PacketError("Unavailable sensors must omit rotation")
        if item.valid:
            if rotation is None or abs(math.hypot(*rotation) - 1) > NORM_TOLERANCE:
                raise PacketError("Valid orientation needs a unit quaternion (norm tolerance 0.001)")
        imus.append(Imu(item.sensor_id, quaternion=rotation if item.valid else None, valid=item.valid))
    return Sample("quaternion", message.sample_time_ms, tuple(sorted(imus, key=lambda imu: imu.sensor_id)), message.sequence)
