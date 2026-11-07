# File: ble.py
# Purpose: One BLE owner: bounded notification receipt, validation and diagnostic updates.
# Contributors: Unassigned

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
import time
from uuid import UUID

from bleak import BleakClient, BleakScanner

from .models import Diagnostics, Sample
from .protocol import decode, PacketError, QUATERNION_MAX_BYTES, RAW_MAX_BYTES

SERVICE_UUID = "0000aaaa-0000-1000-8000-00805f9b34fb"
NOTIFY_UUID = "00008888-0000-1000-8000-00805f9b34fb"
ORIENTATION_UUID = "00008889-0000-1000-8000-00805f9b34fb"


def uuid(value: str) -> str:
    """Expand prototype 16-bit UUIDs exactly as Bluetooth/Bleak do."""
    if len(value) == 4:
        value = f"0000{value}-0000-1000-8000-00805f9b34fb"
    return str(UUID(value))


async def scan(timeout: float = 5, scanner=None) -> list[tuple[object, object]]:
    discover = BleakScanner.discover if scanner is None else scanner
    advertisements = await asyncio.wait_for(discover(timeout=timeout, return_adv=True), timeout + 2)
    return list(advertisements.values())


def select_device(devices: list[tuple[object, object]], device_id: str):
    matches = [device for device, _advertisement in devices if device.address == device_id]
    if len(matches) != 1:
        raise ValueError("Device identifier not uniquely present in scan; run scan again and copy its exact OS ID")
    return matches[0]


class DeviceStream:
    def __init__(self, device, packet_format: str, *, service: str = SERVICE_UUID,
                 characteristic: str | None = None, timeout: float = 10,
                 queue_size: int = 8, client_factory=None):
        if packet_format not in ("raw", "quaternion"):
            raise ValueError("Select raw or quaternion explicitly")
        if not 1 <= queue_size <= 64 or timeout <= 0:
            raise ValueError("Queue size must be 1–64 and timeout positive")
        self.device = device
        self.packet_format = packet_format
        self.service = uuid(service)
        self.characteristic = uuid(characteristic or (NOTIFY_UUID if packet_format == "raw" else ORIENTATION_UUID))
        self.timeout = timeout
        self.client_factory = BleakClient if client_factory is None else client_factory
        self.queue: asyncio.Queue = asyncio.Queue(queue_size)
        self.stats = Diagnostics()
        self.latest: Sample | None = None
        self.received_at: datetime | None = None
        self.received_monotonic: float | None = None
        self._previous: tuple[int, int] | None = None
        self._lost = False
        self._used = False

    def _notify(self, _sender, data):
        if self._lost:
            return
        now = time.monotonic()
        self.stats.received += 1
        self.stats.last_notification = now
        maximum = RAW_MAX_BYTES if self.packet_format == "raw" else QUATERNION_MAX_BYTES
        if len(data) > maximum:
            self.stats.rejected += 1
            self.stats.last_error = f"Oversized notification ({len(data)} bytes); maximum {maximum} for {self.packet_format}"
            return
        # Copy bounded bytes only. Decode, record and print outside the callback.
        if self.queue.full():
            self.queue.get_nowait()
            self.queue.task_done()
            self.stats.dropped += 1
        self.queue.put_nowait((bytes(data), now, datetime.now(UTC)))

    def _disconnected(self, _client):
        self._lost = True
        self.stats.state = "disconnected"
        self.stats.last_error = "BLE device disconnected; restart stream to reconnect"

    def _check_sequence(self, sample: Sample):
        if sample.sequence is None:
            return
        if self._previous is not None:
            sequence, timestamp = self._previous
            time_delta = (sample.sample_time_ms - timestamp) & 0xFFFFFFFF
            delta = (sample.sequence - sequence) & 0xFFFFFFFF
            if time_delta >= 0x80000000:
                self.stats.resets += 1  # Backwards uptime suggests restart, not a mapped UTC time.
            elif not 0 < delta < 0x80000000:
                raise PacketError("Duplicate/backwards sequence; reconnect if the device restarted")
            else:
                self.stats.sequence_gaps += delta - 1
        self._previous = (sample.sequence, sample.sample_time_ms)

    async def updates(self, duration: float | None = None) -> AsyncIterator[Sample | None]:
        """Yield decoded samples and 200 ms status ticks; duration starts after subscription."""
        if self._used:
            raise RuntimeError("Create a new DeviceStream when reconnecting")
        self._used = True
        self._lost = False
        client = self.client_factory(self.device, disconnected_callback=self._disconnected)
        subscribed = False
        self.stats.started_at = time.monotonic()
        try:
            self.stats.state = "connecting"
            await asyncio.wait_for(client.connect(), self.timeout)
            if self._lost or not client.is_connected:
                raise ConnectionError("Device disconnected during connection")
            self.stats.state = "connected"
            yield None
            service = client.services.get_service(self.service)
            characteristic = None if service is None else service.get_characteristic(self.characteristic)
            if characteristic is None or "notify" not in characteristic.properties:
                raise ValueError(f"Missing notify characteristic {self.characteristic} in service {self.service}; check profile/UUIDs")
            await asyncio.wait_for(client.start_notify(characteristic, self._notify), self.timeout)
            subscribed = True
            if self._lost or not client.is_connected:
                raise ConnectionError("Device disconnected during subscription")
            self.stats.state = "subscribed"
            yield None
            deadline = None if duration is None else time.monotonic() + duration
            while not self._lost:
                remaining = 0.2 if deadline is None else min(0.2, deadline - time.monotonic())
                if remaining <= 0:
                    break
                try:
                    payload, received, utc = await asyncio.wait_for(self.queue.get(), remaining)
                except TimeoutError:
                    yield None
                    continue
                try:
                    sample = decode(payload, self.packet_format)
                    self._check_sequence(sample)
                except PacketError as error:
                    self.stats.rejected += 1
                    self.stats.last_error = str(error)
                    yield None  # Invalid traffic must not starve the terminal status refresh.
                else:
                    self.stats.decoded += 1
                    self.latest = sample
                    self.received_monotonic, self.received_at = received, utc
                    if sample.fully_usable:
                        self.stats.last_valid = received
                        self.stats.usable += 1
                    yield sample
                finally:
                    self.queue.task_done()
            if self._lost:
                yield None
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.stats.state = "error"
            self.stats.last_error = f"{type(error).__name__}: {error}"
            raise
        finally:
            self.stats.finished_at = time.monotonic()
            # Cleanup is bounded and still runs on Ctrl+C, timeout or partial setup failure.
            self._lost = True
            for operation in ([lambda: client.stop_notify(self.characteristic)] if subscribed and client.is_connected else []) + [client.disconnect]:
                try:
                    await asyncio.wait_for(operation(), self.timeout)
                except Exception as error:
                    self.stats.last_error = f"Cleanup failed: {type(error).__name__}: {error}"
                    self.stats.state = "error"
            if self.stats.state != "error":
                self.stats.state = "disconnected"
