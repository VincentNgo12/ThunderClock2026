# File: test_protocol.py
# Purpose: Check exact raw compatibility, quaternion semantics and generated size bounds.
# Contributors: Unassigned

import hashlib
import math
from pathlib import Path
import re
import struct
import unittest

from app.generated import orientation_pb2 as wire, telemetry_pb2 as raw
from app.protocol import decode, PacketError, QUATERNION_MAX_BYTES, RAW_MAX_BYTES

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "protocol/fixtures"


def orientation():
    return wire.OrientationPacket.FromString((FIXTURES / "quaternion_identity.bin").read_bytes())


class ProtocolTests(unittest.TestCase):
    def test_raw_matches_independently_encoded_signed_values(self):
        payload = b"\x08\xd2\x09"  # time=1234 ms
        for index in range(1, 5):
            accel = b"".join(bytes([tag]) + struct.pack('<f', value) for tag, value in zip((13, 21, 29), (-1.25*index, 2.5*index, 9.75)))
            gyro = b"".join(bytes([tag]) + struct.pack('<f', value) for tag, value in zip((13, 21, 29), (-.125*index, .25*index, -.5*index)))
            imu = b'\x0a\x0f' + accel + b'\x12\x0f' + gyro
            payload += bytes([18, len(imu)]) + imu
        self.assertEqual(payload, (FIXTURES / 'raw_signed.bin').read_bytes())
        sample = decode(payload, 'raw')
        self.assertEqual(sample.sample_time_ms, 1234)
        for index, imu in enumerate(sample.imus, 1):
            self.assertEqual(imu.accel, (-1.25*index, 2.5*index, 9.75))
            self.assertEqual(imu.gyro, (-.125*index, .25*index, -.5*index))
        self.assertNotAlmostEqual(math.hypot(*sample.imus[-1].accel), 9.81)

    def test_raw_missing_records_messages_and_nonfinite_are_rejected(self):
        for count in (0, 3, 5):
            packet = raw.ImuStripData()
            for _ in range(count):
                packet.data.add().accel.SetInParent()
            with self.assertRaises(PacketError):
                decode(packet.SerializeToString(), 'raw')
        packet = raw.ImuStripData.FromString((FIXTURES / 'raw_signed.bin').read_bytes())
        packet.data[0].gyro.x = float('nan')
        with self.assertRaises(PacketError):
            decode(packet.SerializeToString(), 'raw')
        packet.data[0].gyro.x = 0
        packet.data[0].ClearField('accel')
        with self.assertRaises(PacketError):
            decode(packet.SerializeToString(), 'raw')

    def test_quaternion_zero_components_and_explicit_zero_header_are_valid(self):
        sample = decode(orientation().SerializeToString(), 'quaternion')
        self.assertEqual((sample.sequence, sample.sample_time_ms), (0, 0))
        self.assertTrue(sample.fully_usable)
        self.assertEqual(sample.imus[0].quaternion, (1, 0, 0, 0))
        packet = orientation()
        packet.orientations.reverse()
        self.assertEqual([s.sensor_id for s in decode(packet.SerializeToString(), 'quaternion').imus], [1, 2, 3, 4])

    def test_partial_strip_has_no_fabricated_orientation(self):
        sample = decode((FIXTURES / 'quaternion_partial.bin').read_bytes(), 'quaternion')
        self.assertFalse(sample.fully_usable)
        self.assertFalse(sample.imus[2].valid)
        self.assertIsNone(sample.imus[2].quaternion)

    def test_quaternion_validation(self):
        mutations = [
            lambda p: setattr(p, 'version', 2), lambda p: p.ClearField('version'),
            lambda p: p.ClearField('sequence'), lambda p: p.ClearField('sample_time_ms'),
            lambda p: p.orientations.pop(), lambda p: p.orientations.add(sensor_id=4, valid=False),
            lambda p: setattr(p.orientations[0], 'sensor_id', 0),
            lambda p: setattr(p.orientations[0], 'sensor_id', 2),
            lambda p: p.orientations[0].ClearField('valid'),
            lambda p: p.orientations[0].ClearField('rotation'),
            lambda p: setattr(p.orientations[0].rotation, 'w', 0),
            lambda p: setattr(p.orientations[0].rotation, 'w', 2),
            lambda p: setattr(p.orientations[0].rotation, 'x', float('inf')),
            lambda p: setattr(p.orientations[0].rotation, 'z', float('nan')),
            lambda p: setattr(p.orientations[0], 'valid', False),
        ]
        for mutate in mutations:
            packet = orientation()
            mutate(packet)
            with self.subTest(mutate=mutate), self.assertRaises(PacketError):
                decode(packet.SerializeToString(), 'quaternion')

    def test_wrong_formats_oversize_and_truncation(self):
        for name, packet_format in [('raw_signed', 'raw'), ('quaternion_identity', 'quaternion')]:
            payload = (FIXTURES / f'{name}.bin').read_bytes()
            for length in range(len(payload)):
                with self.subTest(name=name, length=length), self.assertRaises(PacketError):
                    decode(payload[:length], packet_format)
            with self.assertRaises(PacketError):
                decode(payload, 'quaternion' if packet_format == 'raw' else 'raw')
        for payload in (b'\xff', b'x' * 200):
            with self.assertRaises(PacketError):
                decode(payload, 'quaternion')
        with self.assertRaises(ValueError):
            decode(b'', 'auto')

    def test_bounds_and_raw_schema_hash(self):
        for name, symbol, bound in [('orientation', 'telemetry_orientation_v1_OrientationPacket_size', QUATERNION_MAX_BYTES), ('telemetry', 'telemetry_ImuStripData_size', RAW_MAX_BYTES)]:
            header = (ROOT / f'protocol/generated/{name}.pb.h').read_text()
            self.assertEqual(int(re.search(r'#define ' + symbol + r'\s+(\d+)', header)[1]), bound)
        import json
        manifest = json.loads((FIXTURES / 'manifest.json').read_text())
        self.assertEqual(hashlib.sha256((ROOT / 'firmware/proto/telemetry.proto').read_bytes().replace(b'\r\n', b'\n')).hexdigest(), manifest['raw_schema_sha256'])
        self.assertEqual(len((FIXTURES / 'quaternion_maximum.bin').read_bytes()), QUATERNION_MAX_BYTES)
