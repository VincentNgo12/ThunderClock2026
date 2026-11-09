# File: test_recording.py
# Purpose: "Collect data" endpoints: start (with client metadata), labels, samples recorded, file written.
# Contributors: Unassigned

import asyncio
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import httpx

from app.ble import DeviceStream, ORIENTATION_UUID
from app.main import create_app
from test_ble_cli import Client, DEVICE, ADVERTISEMENT, FIXTURES


class RecordingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

        def factory(*args, **kwargs):
            client = Client(*args, **kwargs)
            client.payloads_by_characteristic[ORIENTATION_UUID] = [(FIXTURES / 'quaternion_identity.bin').read_bytes()] * 5
            return client

        async def scan():
            return [(DEVICE, ADVERTISEMENT)]

        env = {'TELEMETRY_SOURCE': 'manual', 'RECORDINGS_DIR': self.folder.name}
        for patcher in (patch.dict(os.environ, env), patch('app.main.scan', scan),
                        patch('app.main.DeviceStream', lambda d, f: DeviceStream(d, f, client_factory=factory)),
                        patch('app.main.select_device', lambda devices, _id: devices[0][0])):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.app = create_app()
        self.lifespan = self.app.router.lifespan_context(self.app)
        await self.lifespan.__aenter__()
        self.addAsyncCleanup(self.lifespan.__aexit__, None, None, None)
        self.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url='http://test')
        self.addAsyncCleanup(self.http.aclose)

    async def test_needs_a_stream(self):
        response = await self.http.post('/recording/start')
        self.assertEqual(response.status_code, 409)

    async def test_labelled_session_is_saved(self):
        start = await self.http.post('/recording/start', params={'name': 'bend test'},
                                     content=json.dumps({'wearer': 'W1'}), headers={'Content-Type': 'text/plain'})
        self.assertEqual(start.status_code, 409)   # not connected yet
        await self.http.post('/device/connect')
        async with asyncio.timeout(1):
            while self.app.state.stream.stats.state != 'subscribed':
                await asyncio.sleep(0)
        start = await self.http.post('/recording/start', params={'name': 'bend test'},
                                     content=json.dumps({'wearer': 'W1'}), headers={'Content-Type': 'text/plain'})
        self.assertEqual(start.status_code, 200)
        self.assertEqual((await self.http.post('/recording/start')).status_code, 409)   # one at a time
        await self.http.post('/recording/label', params={'name': 'forward#1:hold'})
        # The fake strip sends its packets on connect, so reconnect while recording (a recording also
        # survives a reconnect).
        await self.http.post('/device/disconnect')
        self.assertEqual((await self.http.post('/device/connect')).status_code, 200)
        async with asyncio.timeout(1):
            while self.app.state.recorder.samples == 0:
                await asyncio.sleep(0)
        saved = (await self.http.post('/recording/stop')).json()
        self.assertGreater(saved['samples'], 0)
        lines = [json.loads(line) for line in open(saved['path'], encoding='utf-8')]
        self.assertEqual(lines[0]['type'], 'meta')
        self.assertEqual(lines[0]['client'], {'wearer': 'W1'})
        self.assertEqual(lines[0]['device_format'], 'quaternion')
        self.assertIn({'type': 'label', 'name': 'forward#1:hold'}, [{k: l[k] for k in ('type', 'name')} for l in lines if l['type'] == 'label'])
        sample = next(l for l in lines if l['type'] == 'sample')
        self.assertEqual(len(sample['imus']), 4)
        self.assertEqual(lines[-1]['type'], 'end')
        self.assertTrue(os.path.basename(saved['path']).endswith('_bend_test.jsonl'))
        self.assertEqual((await self.http.get('/recording/status')).json(), {'active': False})


if __name__ == '__main__':
    unittest.main()
