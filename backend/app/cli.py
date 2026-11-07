# File: cli.py
# Purpose: Scan or stream BLE diagnostics without starting HTTP or a browser.
# Contributors: Unassigned

import argparse
import asyncio
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

from .ble import DeviceStream, SERVICE_UUID, scan, select_device, uuid
from .models import REGIONS
from .protocol import decode


def positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="ThunderClock BLE diagnostics; explicit raw or quaternion format")
    commands = result.add_subparsers(dest="command", required=True)
    scanner = commands.add_parser("scan", help="list OS device IDs, names and advertised services")
    scanner.add_argument("--timeout", type=positive, default=5)
    stream = commands.add_parser("stream", help="connect to the exact selected OS identifier")
    stream.add_argument("--device", required=True)
    stream.add_argument("--format", choices=("raw", "quaternion"), required=True)
    stream.add_argument("--service", type=uuid, default=SERVICE_UUID)
    stream.add_argument("--characteristic", type=uuid, help="default: raw 8888, quaternion 8889")
    stream.add_argument("--timeout", type=positive, default=10)
    stream.add_argument("--duration", type=positive)
    stream.add_argument("--refresh", type=positive, default=0.5, help="terminal refresh interval in seconds")
    stream.add_argument("--stale-after", type=positive, default=0.5, help="local receipt freshness limit in seconds")
    stream.add_argument("--record", type=Path, help="best-effort decoded JSONL (not lossless acquisition)")
    fixture = commands.add_parser("decode", help="inspect a fixture without Bluetooth")
    fixture.add_argument("file", type=Path)
    fixture.add_argument("--format", choices=("raw", "quaternion"), required=True)
    return result


def show_sample(sample, output):
    print(f"format={sample.format} device_time={sample.sample_time_ms} ms since boot sequence={sample.sequence} usability={'structural only; sensor health unknown' if sample.format == 'raw' else sample.fully_usable}", file=output)
    for imu in sample.imus:
        if sample.format == "raw":
            ax, ay, az = imu.accel
            gx, gy, gz = imu.gyro
            magnitude = math.hypot(ax, ay, az)
            print(f"  imu {imu.sensor_id}: ax={ax:+.4f} ay={ay:+.4f} az={az:+.4f} m/s^2 |a|={magnitude:.4f} m/s^2 delta_g={magnitude-9.80665:+.4f} m/s^2; gx={gx:+.4f} gy={gy:+.4f} gz={gz:+.4f} rad/s [sensor health unknown]", file=output)
        else:
            values = "unavailable" if imu.quaternion is None else " ".join(f"{key}={value:+.5f}" for key, value in zip("wxyz", imu.quaternion))
            norm = "n/a" if imu.quaternion is None else f"{math.hypot(*imu.quaternion):.6f}"
            print(f"  imu {imu.sensor_id} {REGIONS[imu.sensor_id]}: valid={imu.valid} {values} norm={norm}", file=output)


def show_status(stream, stale_after, output):
    stats = stream.stats
    now = time.monotonic()
    notification_age = None if stats.last_notification is None else now - stats.last_notification
    age = stats.age(now)
    fresh = age is not None and age <= stale_after
    incoming = notification_age is not None and notification_age <= stale_after
    data_state = "PARTIAL" if stream.latest and not stream.latest.fully_usable else "current" if fresh else "STALE/WAITING"
    print(f"state={stats.state} notifications_arriving={incoming} data={data_state} "
          f"received={stats.received} decoded={stats.decoded} rejected={stats.rejected} dropped={stats.dropped} "
          f"valid_hz~{stats.snapshot()['valid_frame_hz']} last_valid_age_ms={None if age is None else round(age*1000)} "
          f"gaps={stats.sequence_gaps} resets={stats.resets}", file=output)
    if not incoming and stats.state == "subscribed":
        print("  SILENCE: subscribed but no recent notifications; check firmware acquisition/notify and connection.", file=output)
    if stats.last_error:
        print(f"  last_error: {stats.last_error}", file=output)
    if stream.latest:
        show_sample(stream.latest, output)


async def run(arguments, *, scanner=None, client_factory=None, output=None) -> int:
    output = sys.stdout if output is None else output
    if arguments.command == "decode":
        show_sample(decode(arguments.file.read_bytes(), arguments.format), output)
        return 0
    devices = await scan(arguments.timeout, scanner)
    if arguments.command == "scan":
        for device, advertisement in devices:
            print(f"id={device.address!r} name={advertisement.local_name or device.name or '(unnamed)'!r} services={advertisement.service_uuids or []}", file=output)
        if not devices:
            print("No devices advertised during this scan; check OS Bluetooth and wearable advertising.", file=output)
        return 0
    device = select_device(devices, arguments.device)
    stream = DeviceStream(device, arguments.format, service=arguments.service,
                          characteristic=arguments.characteristic, timeout=arguments.timeout,
                          client_factory=client_factory)
    recording = None
    try:
        if arguments.record:
            recording = arguments.record.open("w", encoding="utf-8")
        last_print = float("-inf")
        last_state = None
        async for sample in stream.updates(arguments.duration):
            if sample is not None and recording is not None:
                row = asdict(sample)
                row.update(received_at=stream.received_at.isoformat(), units={"accel": "m/s²", "gyro": "rad/s", "time": "ms since boot", "quaternion": "w,x,y,z"})
                recording.write(json.dumps(row, allow_nan=False) + "\n")
            now = time.monotonic()
            if now - last_print >= arguments.refresh or stream.stats.state != last_state:
                show_status(stream, arguments.stale_after, output)
                last_print, last_state = now, stream.stats.state
        show_status(stream, arguments.stale_after, output)
        if not stream.stats.usable or stream.stats.state == "error" or stream.stats.age(stream.stats.finished_at) > arguments.stale_after:
            return 3
        if stream.stats.last_error and stream.stats.last_error.startswith("BLE device disconnected"):
            return 3
        return 0
    finally:
        if recording is not None:
            recording.close()


def main(argv=None) -> int:
    arguments = parser().parse_args(argv)
    try:
        return asyncio.run(run(arguments))
    except KeyboardInterrupt:
        print("Stopped; BLE cleanup finished.", file=sys.stderr)
        return 130
    except (OSError, ValueError, TimeoutError, ConnectionError) as error:
        print(f"Diagnostic failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        # OS/Bleak backends have platform-specific exception classes. Keep the error actionable.
        print(f"BLE operation failed: {type(error).__name__}: {error}. Check OS Bluetooth permissions/settings and UUIDs.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
