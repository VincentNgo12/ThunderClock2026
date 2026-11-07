# File: generate_protocol.py
# Purpose: Reproduce bindings and fixtures from the canonical orientation and exact prototype raw schemas.
# Contributors: Unassigned

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from google.protobuf.json_format import MessageToDict

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "backend/app/generated"
PROTOCOL = ROOT / "protocol"


def load(name):
    spec = importlib.util.spec_from_file_location(name, PYTHON / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    PYTHON.mkdir(parents=True, exist_ok=True)
    for directory, name in ((PROTOCOL, "orientation"), (ROOT / "firmware/proto", "telemetry")):
        with tempfile.TemporaryDirectory() as temporary:
            descriptor = Path(temporary) / f"{name}.pb"
            subprocess.run([sys.executable, "-m", "grpc_tools.protoc", f"-I{directory}",
                            f"--python_out={PYTHON}", f"--descriptor_set_out={descriptor}", f"{name}.proto"], check=True)
            # The C bindings also support the host conformance check.
            subprocess.run([sys.executable, "-m", "nanopb.generator.nanopb_generator", "--error-on-unmatched",
                            f"-I{directory}", f"-D{PROTOCOL / 'generated'}", str(descriptor)], check=True)
    orientation, raw = load("orientation_pb2"), load("telemetry_pb2")
    identity = orientation.OrientationPacket(version=1, sequence=0, sample_time_ms=0)
    for sensor_id in range(1, 5):
        item = identity.orientations.add(sensor_id=sensor_id, valid=True)
        item.rotation.w = 1
    partial = orientation.OrientationPacket()
    partial.CopyFrom(identity)
    partial.sequence = 1
    partial.sample_time_ms = 100
    partial.orientations[2].valid = False
    partial.orientations[2].ClearField("rotation")
    maximum = orientation.OrientationPacket(version=0xFFFFFFFF, sequence=0xFFFFFFFF, sample_time_ms=0xFFFFFFFF)
    for _ in range(4):
        item = maximum.orientations.add(sensor_id=0xFFFFFFFF, valid=True)
        item.rotation.w = item.rotation.x = item.rotation.y = item.rotation.z = 0.5
    strip = raw.ImuStripData(time=1234)
    for index in range(1, 5):
        item = strip.data.add()
        item.accel.x, item.accel.y, item.accel.z = -1.25 * index, 2.5 * index, 9.75
        item.gyro.x, item.gyro.y, item.gyro.z = -0.125 * index, 0.25 * index, -0.5 * index
    fixtures = {"raw_signed": strip, "quaternion_identity": identity,
                "quaternion_partial": partial, "quaternion_maximum": maximum}
    manifest = {"synthetic_fixtures_not_hardware": True,
                "raw_schema_sha256": hashlib.sha256((ROOT / 'firmware/proto/telemetry.proto').read_bytes().replace(b'\r\n', b'\n')).hexdigest(), "fixtures": {}}
    for name, packet in fixtures.items():
        data = packet.SerializeToString(deterministic=True)
        (PROTOCOL / "fixtures" / f"{name}.bin").write_bytes(data)
        manifest["fixtures"][name] = {"bytes": len(data), "packet": MessageToDict(packet, preserving_proto_field_name=True)}
        print(f"{name}: {len(data)} bytes")
    (PROTOCOL / "fixtures/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
