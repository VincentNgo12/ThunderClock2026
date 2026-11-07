# File: ble.py
# Purpose: One BLE owner: bounded notification receipt, validation and diagnostic updates.
# Contributors: Unassigned

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from functools import partial
import time
from uuid import UUID

from bleak import BleakClient, BleakScanner

from .models import Diagnostics, Sample
from .protocol import decode, PacketError, QUATERNION_MAX_BYTES, RAW_MAX_BYTES

SERVICE_UUID = "0000aaaa-0000-1000-8000-00805f9b34fb"
NOTIFY_UUID = "00008888-0000-1000-8000-00805f9b34fb"
ORIENTATION_UUID = "00008889-0000-1000-8000-00805f9b34fb"
DEVICE_FORMATS = ("raw", "quaternion", "both")


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
                 queue_size: int | None = None, client_factory=None):
        if packet_format not in DEVICE_FORMATS:
            raise ValueError("Select raw, quaternion or both explicitly")
        if packet_format == "both" and characteristic is not None:
            raise ValueError("Dual streaming uses the standard raw and orientation characteristics")
        if queue_size is None:
            queue_size = 32 if packet_format == "both" else 8
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
        primary_format = "quaternion" if packet_format == "both" else packet_format
        self._characteristics = {primary_format: self.characteristic}
        self.stats_by_format = {primary_format: self.stats}
        if packet_format == "both":
            self._characteristics["raw"] = NOTIFY_UUID
            self.stats_by_format["raw"] = Diagnostics()
        self.latest: Sample | None = None
        self.received_at: datetime | None = None
        self.received_monotonic: float | None = None
        self._previous: tuple[int, int] | None = None
        self._lost = False
        self._used = False

    def _notify(self, packet_format, _sender, data):
        if self._lost:
            return
        now = time.monotonic()
        stats = self.stats_by_format[packet_format]
        stats.received += 1
        stats.last_notification = now
        maximum = RAW_MAX_BYTES if packet_format == "raw" else QUATERNION_MAX_BYTES
        if len(data) > maximum:
            stats.rejected += 1
            stats.last_error = f"Oversized notification ({len(data)} bytes); maximum {maximum} for {packet_format}"
            return
        # Copy bounded bytes only. Decode, record and print outside the callback.
        if self.queue.full():
            dropped_format, *_ = self.queue.get_nowait()
            self.queue.task_done()
            self.stats_by_format[dropped_format].dropped += 1
        self.queue.put_nowait((packet_format, bytes(data), now, datetime.now(UTC)))

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
        subscribed = []
        started_at = time.monotonic()
        for stats in self.stats_by_format.values():
            stats.started_at = started_at
        try:
            self.stats.state = "connecting"
            await asyncio.wait_for(client.connect(), self.timeout)
            if self._lost or not client.is_connected:
                raise ConnectionError("Device disconnected during connection")
            self.stats.state = "connected"
            yield None
            service = client.services.get_service(self.service)
            for packet_format, identifier in self._characteristics.items():
                characteristic = None if service is None else service.get_characteristic(identifier)
                if characteristic is None or "notify" not in characteristic.properties:
                    raise ValueError(f"Missing notify characteristic {identifier} in service {self.service}; check profile/UUIDs")
                callback = partial(self._notify, packet_format)
                await asyncio.wait_for(client.start_notify(characteristic, callback), self.timeout)
                subscribed.append(identifier)
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
                    packet_format, payload, received, utc = await asyncio.wait_for(self.queue.get(), remaining)
                except TimeoutError:
                    yield None
                    continue
                stats = self.stats_by_format[packet_format]
                try:
                    sample = decode(payload, packet_format)
                    self._check_sequence(sample)
                except PacketError as error:
                    stats.rejected += 1
                    stats.last_error = str(error)
                    yield None  # Invalid traffic must not starve the terminal status refresh.
                else:
                    stats.decoded += 1
                    # In dual mode the dashboard and neutral capture track only quaternions.
                    if stats is self.stats:
                        self.latest = sample
                        self.received_monotonic, self.received_at = received, utc
                    if sample.fully_usable:
                        stats.last_valid = received
                        stats.usable += 1
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
            finished_at = time.monotonic()
            for stats in self.stats_by_format.values():
                stats.finished_at = finished_at
            # Cleanup is bounded and still runs on Ctrl+C, timeout or partial setup failure.
            self._lost = True
            cleanup = [partial(client.stop_notify, identifier) for identifier in subscribed if client.is_connected]
            for operation in [*cleanup, client.disconnect]:
                try:
                    await asyncio.wait_for(operation(), self.timeout)
                except Exception as error:
                    self.stats.last_error = f"Cleanup failed: {type(error).__name__}: {error}"
                    self.stats.state = "error"
            if self.stats.state != "error":
                self.stats.state = "disconnected"
