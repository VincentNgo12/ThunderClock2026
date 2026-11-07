# File: test_ble_cli.py
# Purpose: Exercise notification ownership, bounded receipt and standalone CLI behavior with one simple fake.
# Contributors: Unassigned

import asyncio
from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.ble import DeviceStream, NOTIFY_UUID, ORIENTATION_UUID, SERVICE_UUID, scan, select_device, uuid
from app.cli import main, parser, run
from app.generated import orientation_pb2 as wire

FIXTURES = Path(__file__).resolve().parents[2] / 'protocol/fixtures'
DEVICE = SimpleNamespace(address='opaque-os-identifier', name='THUNDER CLOCK')
ADVERTISEMENT = SimpleNamespace(local_name='THUNDER CLOCK', service_uuids=[SERVICE_UUID])


async def discover(**kwargs):
    return {DEVICE.address: (DEVICE, ADVERTISEMENT)}


class Client:
    def __init__(self, device, disconnected_callback):
        self.device = device
        self.lost = disconnected_callback
        self.is_connected = False
        self.services = self
        self.properties = ['read', 'notify']
        self.callback = None
        self.payloads = []
        self.stopped = self.closed = False
        self.block = None
        self.drop = False

    def get_service(self, identifier):
        return self if identifier == SERVICE_UUID else None

    def get_characteristic(self, identifier):
        return self if identifier in (NOTIFY_UUID, ORIENTATION_UUID) else None

    async def connect(self):
        if self.block == 'connect':
            await asyncio.Event().wait()
        self.is_connected = True

    async def start_notify(self, characteristic, callback):
        self.callback = callback
        if self.block == 'subscribe':
            await asyncio.Event().wait()
        for payload in self.payloads:
            callback(None, payload)
        if self.drop:
            self.is_connected = False
            self.lost(self)

    async def stop_notify(self, characteristic):
        self.stopped = True

    async def disconnect(self):
        self.closed = True
        self.is_connected = False


class BleCliTests(unittest.IsolatedAsyncioTestCase):
    def setup_stream(self, **kwargs):
        clients = []
        def factory(*args, **options):
            client = Client(*args, **options)
            clients.append(client)
            return client
        return DeviceStream(DEVICE, 'quaternion', client_factory=factory, timeout=.05, **kwargs), clients

    async def test_scan_exact_selection_and_uuid_expansion(self):
        devices = await scan(.1, discover)
        self.assertIs(select_device(devices, DEVICE.address), DEVICE)
        self.assertEqual(uuid('AAAA'), SERVICE_UUID)
        self.assertEqual(uuid('8888'), NOTIFY_UUID)
        self.assertEqual(uuid('8889'), ORIENTATION_UUID)
        self.assertEqual(DeviceStream(DEVICE, 'raw').characteristic, NOTIFY_UUID)
        self.assertEqual(DeviceStream(DEVICE, 'quaternion').characteristic, ORIENTATION_UUID)
        with self.assertRaises(ValueError):
            select_device(devices, 'THUNDER CLOCK')
        with self.assertRaises(ValueError):
            select_device(devices + devices, DEVICE.address)
        async def blocked(**kwargs):
            await asyncio.Event().wait()
        with self.assertRaises(TimeoutError):
            await scan(.01, blocked)

    async def test_pressure_valid_age_and_silence(self):
        stream, clients = self.setup_stream(queue_size=2)
        updates = stream.updates(.06)
        await anext(updates)  # connected
        await anext(updates)  # subscribed
        payload = (FIXTURES / 'quaternion_identity.bin').read_bytes()
        for _ in range(10):
            clients[0].callback(None, payload)
        await anext(updates)
        receipt = stream.stats.last_valid
        clients[0].callback(None, b'\xff')
        async for _ in updates:
            pass
        self.assertEqual(stream.stats.received, 11)
        self.assertEqual(stream.stats.dropped, 8)
        self.assertEqual(stream.stats.decoded, 1)
        self.assertEqual(stream.stats.rejected, 2)  # duplicate sequence + malformed bytes
        self.assertEqual(stream.stats.last_valid, receipt)
        self.assertGreater(stream.stats.age(), .04)
        self.assertTrue(clients[0].stopped and clients[0].closed)
        clients[0].callback(None, payload)
        self.assertEqual(stream.stats.received, 11)

    async def test_setup_timeout_missing_profile_and_cleanup(self):
        for failure in ('connect', 'subscribe', 'profile'):
            client = Client(DEVICE, lambda _: None)
            client.block = failure
            if failure == 'profile':
                client.properties = ['read']
            stream = DeviceStream(DEVICE, 'raw', timeout=.02, client_factory=lambda *a, **k: client)
            with self.subTest(failure=failure), self.assertRaises((TimeoutError, ValueError)):
                async for _ in stream.updates(.01):
                    pass
            self.assertTrue(client.closed)

    async def test_cancellation_cleans_subscription(self):
        stream, clients = self.setup_stream()
        async def consume():
            async for _ in stream.updates():
                pass
        task = asyncio.create_task(consume())
        while not clients or clients[0].callback is None:
            await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(clients[0].stopped and clients[0].closed)

    async def test_cli_duration_no_valid_and_signed_raw_recording(self):
        payload = (FIXTURES / 'raw_signed.bin').read_bytes()
        clients = []
        def factory(*args, **kwargs):
            client = Client(*args, **kwargs)
            client.payloads = [payload]
            clients.append(client)
            return client
        args = parser().parse_args(['stream', '--device', DEVICE.address, '--format', 'raw', '--duration', '.02'])
        output = io.StringIO()
        self.assertEqual(await run(args, scanner=discover, client_factory=factory, output=output), 0)
        self.assertIn('ax=-1.2500', output.getvalue())
        self.assertIn('rad/s', output.getvalue())
        self.assertIn('sensor health unknown', output.getvalue())
        import tempfile, json
        with tempfile.TemporaryDirectory() as directory:
            args.record = Path(directory) / 'capture.jsonl'
            self.assertEqual(await run(args, scanner=discover, client_factory=factory, output=io.StringIO()), 0)
            rows = [json.loads(row) for row in args.record.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['units']['gyro'], 'rad/s')
            self.assertEqual(rows[0]['imus'][0]['accel'][0], -1.25)
        args.record = None
        clients.clear()
        output = io.StringIO()
        self.assertEqual(await run(args, scanner=discover, client_factory=Client, output=output), 3)
        self.assertIn('SILENCE', output.getvalue())

    async def test_normal_cleanup_delay_does_not_turn_good_capture_into_failure(self):
        def factory(*args, **kwargs):
            client = Client(*args, **kwargs)
            client.payloads = [(FIXTURES / 'raw_signed.bin').read_bytes()]
            disconnect = client.disconnect
            async def slow_disconnect():
                await asyncio.sleep(.08)
                await disconnect()
            client.disconnect = slow_disconnect
            return client
        args = parser().parse_args(['stream', '--device', DEVICE.address, '--format', 'raw',
                                    '--duration', '.02', '--stale-after', '.04'])
        self.assertEqual(await run(args, scanner=discover, client_factory=factory, output=io.StringIO()), 0)

    async def test_disconnect_is_reported(self):
        stream, clients = self.setup_stream()
        updates = stream.updates()
        await anext(updates)
        await anext(updates)
        clients[0].is_connected = False
        clients[0].lost(clients[0])
        async for _ in updates:
            pass
        self.assertIn('disconnected', stream.stats.last_error)
        self.assertTrue(clients[0].closed)

    async def test_quaternion_wrap_and_restart(self):
        stream, _ = self.setup_stream()
        from app.protocol import decode
        packet = wire.OrientationPacket.FromString((FIXTURES / 'quaternion_identity.bin').read_bytes())
        for sequence, uptime in [(0xFFFFFFFF, 0xFFFFFFFF), (0, 0), (5, 100), (0, 0)]:
            packet.sequence, packet.sample_time_ms = sequence, uptime
            stream._check_sequence(decode(packet.SerializeToString(), 'quaternion'))
        self.assertEqual(stream.stats.sequence_gaps, 4)
        self.assertEqual(stream.stats.resets, 1)


class EntryPointTests(unittest.TestCase):
    def test_cli_import_does_not_import_http(self):
        import subprocess
        result = subprocess.run([sys.executable, '-c', "import app.cli, sys; assert 'fastapi' not in sys.modules; assert 'uvicorn' not in sys.modules"], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_ctrl_c_returns_without_traceback(self):
        async def interrupted(*args, **kwargs):
            raise KeyboardInterrupt()
        with patch('app.cli.run', interrupted), redirect_stderr(io.StringIO()) as error:
            self.assertEqual(main(['scan']), 130)
        self.assertNotIn('Traceback', error.getvalue())
