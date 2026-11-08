# File: test_dashboard.py
# Purpose: Check shared WebSocket delivery and availability in dashboard snapshots.
# Contributors: Unassigned

import asyncio
import json
import math
import os
import unittest
from unittest.mock import patch

import httpx

from app.ble import DeviceStream, NOTIFY_UUID, ORIENTATION_UUID
from app.dashboard import snapshot
from app.generated import orientation_pb2, telemetry_pb2
from app.main import create_app
from test_ble_cli import Client, DEVICE, ADVERTISEMENT, FIXTURES


class DashboardTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.clients = []
        def factory(*args, **kwargs):
            client = Client(*args, **kwargs)
            client.payloads_by_characteristic[ORIENTATION_UUID] = [(FIXTURES / 'quaternion_identity.bin').read_bytes()]
            self.clients.append(client)
            return client
        def stream(device, packet_format):
            return DeviceStream(device, packet_format, client_factory=factory)
        async def scan():
            return [(DEVICE, ADVERTISEMENT)]
        for patcher in (patch.dict(os.environ, {'TELEMETRY_SOURCE': 'hardware', 'DEVICE_ID': DEVICE.address,
                                                'DEVICE_FORMAT': 'quaternion'}),
                        patch('app.main.scan', scan), patch('app.main.DeviceStream', stream)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.app = create_app()
        self.lifespan = self.app.router.lifespan_context(self.app)
        await self.lifespan.__aenter__()
        self.addAsyncCleanup(self.lifespan.__aexit__, None, None, None)
        async with asyncio.timeout(1):
            while self.app.state.stream.stats.decoded == 0:
                await asyncio.sleep(0)

    async def socket(self, blocked=False, *, app=None):
        app = self.app if app is None else app
        incoming, outgoing = asyncio.Queue(), asyncio.Queue()
        incoming.put_nowait({'type': 'websocket.connect'})
        async def send(message):
            if blocked and message['type'] == 'websocket.send':
                await asyncio.Event().wait()
            await outgoing.put(message)
        scope = {'type': 'websocket', 'path': '/ws/telemetry', 'scheme': 'ws', 'query_string': b'',
                 'headers': [], 'subprotocols': [], 'client': ('127.0.0.1', 1), 'server': ('127.0.0.1', 8000)}
        task = asyncio.create_task(app(scope, incoming.get, send))
        async def cleanup():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.addAsyncCleanup(cleanup)
        self.assertEqual((await asyncio.wait_for(outgoing.get(), 1))['type'], 'websocket.accept')
        return task, outgoing

    async def test_api_reuses_core_and_preserves_missing_values(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url='http://test') as client:
            self.assertEqual((await client.get('/diagnostics')).json()['decoded'], 1)
        _a, aq = await self.socket()
        _b, bq = await self.socket()
        left = json.loads((await aq.get())['text'])
        right = json.loads((await bq.get())['text'])
        for _ in range(8):
            if left['sequence'] == right['sequence']:
                break
            if left['sequence'] < right['sequence']:
                left = json.loads((await aq.get())['text'])
            else:
                right = json.loads((await bq.get())['text'])
        self.assertEqual(left, right)
        self.assertIsNone(left['device'])
        for field in ('posture', 'eeg', 'overall_risk', 'environment', 'heart_rate_bpm'):
            self.assertIsNone(left[field])
        stream = self.app.state.stream
        with patch('app.dashboard.time') as clock:
            clock.monotonic.return_value = stream.received_monotonic + 1
            stale = snapshot(stream, 500)
        self.assertTrue(all(imu['orientation'] is None for imu in stale['imus']))
        self.assertEqual(stale['imus'][0]['freshness']['status'], 'stale')

    async def test_stalled_client_does_not_block_receipt(self):
        task, outgoing = await self.socket(blocked=True)
        self.clients[0].callback(None, b'\xff')
        async with asyncio.timeout(1):
            while self.app.state.stream.stats.rejected == 0:
                await asyncio.sleep(0)
        close = await asyncio.wait_for(outgoing.get(), 2)
        self.assertEqual(close['code'], 1013)
        await asyncio.wait_for(task, 1)

    async def test_environment_keeps_present_zero_and_expires_with_packet(self):
        packet = orientation_pb2.OrientationPacket.FromString((FIXTURES / 'telemetry_v2.bin').read_bytes())
        packet.sequence = 1
        packet.environment.ClearField('temperature_c')
        self.clients[0].callback(None, packet.SerializeToString())
        stream = self.app.state.stream
        async with asyncio.timeout(1):
            while stream.stats.decoded < 2:
                await asyncio.sleep(0)

        current = snapshot(stream, 1)
        self.assertEqual(current['environment']['tvoc_ppb'], 0)
        self.assertIsNone(current['environment']['temperature_c'])
        self.assertEqual(current['freshness']['environment']['status'], 'valid')
        with patch('app.dashboard.time') as clock:
            clock.monotonic.return_value = stream.received_monotonic + 1
            stale = snapshot(stream, 2)
        self.assertIsNone(stale['environment'])
        self.assertEqual(stale['freshness']['environment']['status'], 'stale')

    async def test_api_needs_explicit_settings(self):
        with patch.dict(os.environ, {'DEVICE_ID': '', 'DEVICE_FORMAT': ''}):
            app = create_app()
            with self.assertRaises(ValueError):
                async with app.router.lifespan_context(app):
                    pass

    async def test_dual_mode_feeds_raw_consumer_and_keeps_posture(self):
        samples = []
        initial_connections = len(self.clients)
        with patch.dict(os.environ, {'DEVICE_FORMAT': 'both'}):
            app = create_app(on_raw_sample=samples.append)
            async with app.router.lifespan_context(app):
                stream = app.state.stream
                async with asyncio.timeout(1):
                    while stream.latest is None:
                        await asyncio.sleep(0)
                self.assertEqual(len(self.clients), initial_connections + 1)
                client = self.clients[-1]
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as http:
                    self.assertEqual((await http.post('/posture/calibrate')).status_code, 200)
                    raw = telemetry_pb2.ImuStripData.FromString((FIXTURES / 'raw_signed.bin').read_bytes())
                    for _ in range(5):
                        client.callbacks[NOTIFY_UUID](None, raw.SerializeToString())
                        raw.time += 10
                    async with asyncio.timeout(1):
                        while len(samples) < 5:
                            await asyncio.sleep(0)
                    self.assertEqual([sample.sample_time_ms for sample in samples], [1234, 1244, 1254, 1264, 1274])
                    self.assertEqual(samples[0].imus[0].accel[0], -1.25)
                    self.assertAlmostEqual(app.state.posture.angles['lumbar_flexion_deg'], 0)
                    stats = (await http.get('/diagnostics')).json()
                    self.assertEqual((stats['decoded'], stats['raw']['decoded']), (1, 5))

    async def test_ble_neutral_capture_and_partial_frame_conflict(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url='http://test') as client:
            response = await client.post('/posture/calibrate')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['status'], 'captured')
            self.assertAlmostEqual(self.app.state.posture.angles['lumbar_flexion_deg'], 0)
            stream = self.app.state.stream
            pose = self.app.state.posture
            with patch('app.dashboard.time') as clock:
                clock.monotonic.return_value = stream.received_monotonic + 1
                stale = snapshot(stream, 1, posture=pose)
            self.assertIsNone(stale['posture'])
            self.assertEqual(stale['freshness']['posture']['status'], 'stale')
            self.assertEqual(stale['calibration'], response.json())

            packet = orientation_pb2.OrientationPacket.FromString((FIXTURES / 'quaternion_partial.bin').read_bytes())
            self.clients[0].callback(None, packet.SerializeToString())
            async with asyncio.timeout(1):
                while self.app.state.stream.stats.decoded < 2:
                    await asyncio.sleep(0)
            self.assertEqual((await client.post('/posture/calibrate')).status_code, 409)
            self.assertEqual(self.app.state.posture.calibration, response.json())
            self.assertIsNone(self.app.state.posture.angles)
            partial = snapshot(stream, 2, posture=pose)
            self.assertIsNone(partial['posture'])
            self.assertIsNone(partial['imus'][2]['orientation'])
            self.assertEqual(partial['imus'][2]['freshness']['status'], 'unavailable')

    async def test_backend_fusion_calibrates_and_streams_from_raw_only(self):
        samples = []
        with patch.dict(os.environ, {'DEVICE_FORMAT': 'raw_fusion',
                                     'RAW_IMU_AXES': '-y,z,-x;-y,z,-x;-y,z,-x;-y,z,-x'}):
            app = create_app(on_raw_sample=samples.append)
            async with app.router.lifespan_context(app):
                stream = app.state.stream
                async with asyncio.timeout(1):
                    while stream.stats.state != 'subscribed':
                        await asyncio.sleep(0)
                client = self.clients[-1]
                self.assertEqual(set(client.callbacks), {NOTIFY_UUID})
                packet = telemetry_pb2.ImuStripData()
                for _ in range(4):
                    imu = packet.data.add()
                    imu.accel.z = 9.80665
                    imu.gyro.SetInParent()

                async def send_raw(time_ms):
                    packet.time = time_ms
                    expected = stream.stats.decoded + 1
                    client.callbacks[NOTIFY_UUID](None, packet.SerializeToString())
                    async with asyncio.timeout(1):
                        while stream.stats.decoded < expected:
                            await asyncio.sleep(0)

                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as http:
                    self.assertEqual((await http.post('/posture/calibrate')).status_code, 409)
                    for time_ms in range(0, 501, 10):
                        await send_raw(time_ms)
                    self.assertEqual((await http.post('/posture/calibrate')).status_code, 200)
                    for step in range(1, 11):
                        angle = -math.pi * step / 200
                        packet.data[1].accel.x = 9.80665 * math.sin(angle)
                        packet.data[1].accel.z = 9.80665 * math.cos(angle)
                        packet.data[1].gyro.y = math.pi / 2
                        await send_raw(500 + step * 10)

                    _task, outgoing = await self.socket(app=app)
                    async with asyncio.timeout(1):
                        while True:
                            message = json.loads((await outgoing.get())['text'])
                            if message['posture'] is not None and message['posture']['lumbar_flexion_deg'] > 8.5:
                                break
                    self.assertEqual(message['connection']['source'], 'hardware')
                    self.assertAlmostEqual(message['posture']['lumbar_flexion_deg'], 9, delta=0.5)
                    self.assertLess(message['posture']['joint_rotations']['lumbar']['x'], 0)
                    self.assertEqual(samples[-1].format, 'raw')
                    self.assertAlmostEqual(samples[-1].imus[1].gyro[1], math.pi / 2, places=6)

                    await send_raw(800)
                    self.assertEqual((await http.post('/posture/calibrate')).status_code, 409)
                    self.assertEqual(app.state.posture.calibration['status'], 'uncalibrated')

    async def test_explicit_simulation_publishes_calibrated_posture(self):
        with patch.dict(os.environ, {'TELEMETRY_SOURCE': 'simulated'}), patch(
            'app.main.scan', side_effect=AssertionError('Simulation must not scan for BLE')
        ):
            app = create_app()
            async with app.router.lifespan_context(app):
                async with asyncio.timeout(1):
                    while app.state.stream.latest is None:
                        await asyncio.sleep(0)
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                    self.assertEqual((await client.get('/health')).json()['source'], 'simulated')
                    calibration = await client.post('/posture/calibrate')
                    self.assertEqual(calibration.status_code, 200)
                _task, outgoing = await self.socket(app=app)
                async with asyncio.timeout(1):
                    while True:
                        message = json.loads((await outgoing.get())['text'])
                        if message['posture'] is not None:
                            break
                self.assertEqual(message['connection']['source'], 'simulated')
                self.assertEqual(message['freshness']['posture']['source'], 'simulated')
                self.assertAlmostEqual(message['posture']['lumbar_flexion_deg'], 0)
                self.assertEqual(message['alerts'], [])
                identity = {'w': 1, 'x': 0, 'y': 0, 'z': 0}
                self.assertEqual(message['calibration'], calibration.json())
                self.assertEqual(message['calibration']['reference']['root_orientation'], identity)
                self.assertEqual(message['posture']['joint_rotations']['lumbar'], identity)
                self.assertEqual(message['imus'][0]['orientation'], identity)

                # Jump into the simulator's held bend without waiting 12 real seconds.
                app.state.stream.stats.started_at -= 12
                async with asyncio.timeout(1):
                    while True:
                        message = json.loads((await outgoing.get())['text'])
                        if message['posture']['lumbar_flexion_deg'] > 30:
                            break
                delta = message['posture']['joint_rotations']['lumbar']
                self.assertLess(delta['x'], 0)
                self.assertAlmostEqual(sum(value * value for value in delta.values()), 1)
                self.assertEqual(message['imus'][0]['orientation'], identity)
                examples = FIXTURES.parents[1] / 'docs/telemetry-examples.json'
                example = json.loads(examples.read_text(encoding='utf-8'))['simulated_bend']
                self.assertEqual(message.keys(), example.keys())
                self.assertEqual(message['posture'].keys(), example['posture'].keys())
                self.assertEqual(message['calibration']['reference'], example['calibration']['reference'])
                for name, expected in example['posture']['joint_rotations'].items():
                    actual = message['posture']['joint_rotations'][name]
                    self.assertEqual(actual.keys(), expected.keys())
                    for component in expected:
                        self.assertAlmostEqual(actual[component], expected[component])
