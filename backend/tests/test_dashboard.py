# File: test_dashboard.py
# Purpose: Check shared WebSocket delivery and availability in dashboard snapshots.
# Contributors: Unassigned

import asyncio
import json
import os
import unittest
from unittest.mock import patch

import httpx

from app.ble import DeviceStream
from app.dashboard import snapshot
from app.generated import orientation_pb2
from app.main import create_app
from test_ble_cli import Client, DEVICE, ADVERTISEMENT, FIXTURES


class DashboardTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.clients = []
        def factory(*args, **kwargs):
            client = Client(*args, **kwargs)
            client.payloads = [(FIXTURES / 'quaternion_identity.bin').read_bytes()]
            self.clients.append(client)
            return client
        def stream(device, packet_format):
            return DeviceStream(device, packet_format, client_factory=factory)
        async def scan():
            return [(DEVICE, ADVERTISEMENT)]
        for patcher in (patch.dict(os.environ, {'DEVICE_ID': DEVICE.address, 'DEVICE_FORMAT': 'quaternion'}),
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

    async def socket(self, blocked=False):
        incoming, outgoing = asyncio.Queue(), asyncio.Queue()
        incoming.put_nowait({'type': 'websocket.connect'})
        async def send(message):
            if blocked and message['type'] == 'websocket.send':
                await asyncio.Event().wait()
            await outgoing.put(message)
        scope = {'type': 'websocket', 'path': '/ws/telemetry', 'scheme': 'ws', 'query_string': b'',
                 'headers': [], 'subprotocols': [], 'client': ('127.0.0.1', 1), 'server': ('127.0.0.1', 8000)}
        task = asyncio.create_task(self.app(scope, incoming.get, send))
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
        stale = snapshot(self.app.state.stream, 500, stale_after=0)
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
        stale = snapshot(stream, 2, stale_after=0)
        self.assertIsNone(stale['environment'])
        self.assertEqual(stale['freshness']['environment']['status'], 'stale')

    async def test_api_needs_explicit_settings(self):
        with patch.dict(os.environ, {'DEVICE_ID': '', 'DEVICE_FORMAT': ''}):
            app = create_app()
            with self.assertRaises(ValueError):
                async with app.router.lifespan_context(app):
                    pass
