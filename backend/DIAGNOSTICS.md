# ThunderClock BLE diagnostics

The standalone CLI discovers a wearable, displays signed raw IMU readings and
validates four orientation quaternions. Use one process per device connection.

## Install and inspect fixtures

Use Python 3.11 or later. Run CLI commands from `backend/`.

```sh
cd backend
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m app.cli decode ../protocol/fixtures/raw_signed.bin --format raw
python -m app.cli decode ../protocol/fixtures/quaternion_identity.bin --format quaternion
python -m app.cli decode ../protocol/fixtures/quaternion_partial.bin --format quaternion
```

On Windows PowerShell, activate the environment with
`\.\.venv\Scripts\Activate.ps1` after creating it. Fixture decoding runs
offline. The packets in `protocol/fixtures/` are synthetic.

## Scan and stream

The prototype profile targets advertising name `THUNDER CLOCK`, service `AAAA`,
and READ + NOTIFY characteristics `8888` for raw and `8889` for quaternions.
Short UUIDs expand using the Bluetooth base UUID
`0000xxxx-0000-1000-8000-00805f9b34fb`.

Use a wearable build that implements the selected profile. Copy the exact OS ID
from discovery. macOS may use an opaque UUID. The stream command rescans to
resolve the ID to a native BLE device.

```sh
python -m app.cli scan --timeout 5
python -m app.cli stream --device "<exact OS id>" --format raw --duration 10
python -m app.cli stream --device "<exact OS id>" --format quaternion --duration 15
python -m app.cli stream --device "<exact OS id>" --format raw --duration 10 --record capture.jsonl
```

Use `--service` and `--characteristic` to override UUIDs for a known profile.
Recording writes the decoded samples the consumer receives. Check the drop
counter when using JSONL captures.

The default queue holds eight notifications, with 200 ms status polling and a
500 ms terminal refresh interval. The local freshness limit is 500 ms. Override
refresh or freshness with `--refresh` or `--stale-after` in seconds. Connect,
subscribe and cleanup each have a 10 s default timeout set by `--timeout`.
The stream duration starts after subscription. Ctrl+C closes the connection.

| Exit code | Meaning |
| --- | --- |
| 0 | Scan completed, fixture decoded, or stream completed with recent usable data |
| 2 | Invalid arguments, file or setup failure |
| 3 | Stream ended with unusable/stale data, an unexpected disconnect, or a cleanup error |
| 130 | Stopped with Ctrl+C |

An empty device list is a valid scan result.

## Follow a notification

1. `ble.scan()` and `select_device()` resolve the native device by exact ID.
2. `DeviceStream.updates()` connects, checks the profile and subscribes.
3. The callback queues bounded bytes with monotonic and UTC receipt times.
   If the queue fills, it drops the oldest item and increments the drop counter.
4. `protocol.decode()` validates the selected format and values.
5. `cli.run()` displays status and writes optional JSONL outside the callback.

`received` counts incoming notifications. `decoded` includes partial quaternion
strips. `valid_frame_hz` and `last_valid_age_ms` refer to fully usable frames.
Raw usability describes structure. Evaluate physical sensor health separately.
Raw acceleration retains gravity and uses m/s². Gyro values use rad/s.

Local monotonic receipt age includes time spent in the queue. Device uptime
uses a separate clock. Radio transit time is unknown.

| Observation | Check |
| --- | --- |
| Empty scan | Board power, advertising and OS Bluetooth permissions |
| Connect failed | Competing device owners and the exact ID from a fresh scan |
| Connected, waiting for subscription | Service UUID, characteristic UUID and NOTIFY property |
| SILENCE | Firmware acquisition and notification path |
| Received, zero decoded | Selected format, complete packet length and last decode error |
| Dropped frames | Sender rate and terminal or recording pressure |
| Partial quaternions | Sensor validity, mounting and stationary calibration |

Each notification carries one complete message. The generated bounds are
150 bytes for raw and 153 for quaternions, requiring ATT MTU at least 153 and
156 respectively to fit the maximum. Check the actual negotiated capacity.
See [the device contract](../protocol/ORIENTATION.md) and
[the firmware acceptance checklist](../firmware/ORIENTATION.md).

## Optional API

Export `DEVICE_ID` and `DEVICE_FORMAT` in the shell. The CLI uses flags and the
API reads process environment variables. `.env.example` lists the two settings.
Use one Uvicorn worker and stop CLI streaming before starting the API.

```sh
# From backend/ with the environment active
python -m pip install -r requirements-api.txt
DEVICE_ID="<exact OS id>" DEVICE_FORMAT=quaternion python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In PowerShell, set `$env:DEVICE_ID` and `$env:DEVICE_FORMAT` before running the
Uvicorn command. Startup resolves the wearable with a scan.

`GET /health` reports service status and `GET /diagnostics` reports reader
status and counters. `WS /ws/telemetry` publishes a shared latest snapshot at
about 20 Hz with a one-second send deadline per client. The diagnostic mapping
uses `WorkerTelemetry` 2.0. Raw mode leaves orientations unavailable. Device
metadata and per-IMU acquisition times are null. Posture, fall state,
environment, EEG, heart rate, and risk remain unavailable.

## Tests and generation

```sh
# From backend/
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v

# From the repository root with the same environment active
python backend/tools/generate_protocol.py
python backend/tools/check_nanopb.py --nanopb-root /path/to/nanopb-0.4.9.1
```

Generation reads the raw and orientation schemas and writes the Python/C
bindings and synthetic fixtures. The host conformance check needs a C compiler
and the nanopb 0.4.9.1 runtime sources. Set `--cc` to a compiler command such
as `--cc zig cc` when the compiler is available on PATH.

For physical acceptance, scan for the target wearable, confirm four signed raw
readings, restart the stream, and check Ctrl+C cleanup. For the quaternion
profile, verify sensor IDs, validity, norms and the negotiated MTU after
calibration. Record hardware observations separately from fixture results.
