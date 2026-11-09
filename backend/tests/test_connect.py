# File: test_connect.py
# Purpose: Manual source: the dashboard's POST /device/connect scans, connects and streams; disconnect stops it.
# Contributors: Unassigned

import asyncio
import os
import unittest
from unittest.mock import patch

import httpx

from app.ble import DeviceStream, ORIENTATION_UUID
from app.main import DEFAULT_DEVICE_ID, create_app
from test_ble_cli import Client, DEVICE, ADVERTISEMENT, FIXTURES


class ConnectTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.selected = []

        def factory(*args, **kwargs):
            client = Client(*args, **kwargs)
            client.payloads_by_characteristic[ORIENTATION_UUID] = [(FIXTURES / 'quaternion_identity.bin').read_bytes()]
            return client

        def stream(device, packet_format):
            return DeviceStream(device, packet_format, client_factory=factory)

        async def scan():
            return [(DEVICE, ADVERTISEMENT)]

        def select(devices, device_id):
            self.selected.append(device_id)
            if device_id == 'missing':
                raise ValueError('Device missing not found')
            return devices[0][0]

        env = {'TELEMETRY_SOURCE': 'manual'}
        for patcher in (patch.dict(os.environ, env), patch('app.main.scan', scan), patch('app.main.DeviceStream', stream),
                        patch('app.main.select_device', select)):
            patcher.start()
            self.addCleanup(patcher.stop)
        os.environ.pop('DEVICE_ID', None)
        os.environ.pop('DEVICE_FORMAT', None)
        self.app = create_app()
        self.lifespan = self.app.router.lifespan_context(self.app)
        await self.lifespan.__aenter__()
        self.addAsyncCleanup(self.lifespan.__aexit__, None, None, None)
        self.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url='http://test')
        self.addAsyncCleanup(self.http.aclose)

    async def test_idle_until_connect_then_streams(self):
        self.assertEqual(self.app.state.stream.stats.state, 'disconnected')
        response = await self.http.post('/device/connect')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'device_id': DEFAULT_DEVICE_ID, 'format': 'quaternion'})
        self.assertEqual(self.selected, [DEFAULT_DEVICE_ID])
        async with asyncio.timeout(1):
            while self.app.state.stream.stats.decoded == 0:
                await asyncio.sleep(0)
        self.assertEqual(self.app.state.source, 'hardware')
        self.assertEqual((await self.http.post('/device/connect')).status_code, 409)   # already connected
        response = await self.http.post('/device/disconnect')
        self.assertEqual(response.json(), {'state': 'disconnected'})

    async def test_device_not_found(self):
        response = await self.http.post('/device/connect', params={'device_id': 'missing'})
        self.assertEqual(response.status_code, 404)
        self.assertFalse(self.app.state.connecting)


if __name__ == '__main__':
    unittest.main()
