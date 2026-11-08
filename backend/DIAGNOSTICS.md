# ThunderClock BLE diagnostics

The standalone CLI discovers a wearable, displays signed raw IMU readings and
validates four orientation quaternions and any available v2 environment,
heart rate and SpO₂ readings.
Use one process per device connection.

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
python -m app.cli decode ../protocol/fixtures/telemetry_v2.bin --format quaternion
```

On Windows PowerShell, activate the environment with
`\.\.venv\Scripts\Activate.ps1` after creating it. Fixture decoding runs
offline. The packets in `protocol/fixtures/` are synthetic.

## Scan and stream

The prototype profile targets advertising name `THUNDER CLOCK`, service `AAAA`,
and READ + NOTIFY characteristics `8888` for raw and `8889` for quaternions
with optional v2 sensor readings. Select `--format quaternion` for either
version on 8889.
Short UUIDs expand using the Bluetooth base UUID
`0000xxxx-0000-1000-8000-00805f9b34fb`.

Use a wearable build that implements the selected profile. `stream` scans for a
single device named `ThunderClock` or `THUNDER CLOCK`.

```sh
python -m app.cli stream --format raw --duration 10
python -m app.cli stream --format quaternion --duration 15
python -m app.cli stream --format raw --duration 10 --record capture.jsonl
```

For multiple boards or a different advertising name, run `python -m app.cli
scan --timeout 5` and pass the exact OS ID with `--device "<exact OS id>"`.
Use `--service` and `--characteristic` to override UUIDs for a known profile.
Recording writes the decoded samples the consumer receives. Check the drop
counter when using JSONL captures.

The default queue holds eight notifications, with 200 ms status polling and a
500 ms terminal refresh interval. The local freshness limit is 500 ms. Override
refresh or freshness with `--refresh` or `--stale-after` in seconds. Discovery,
connect, subscribe and cleanup each use a 10 s default timeout set by `--timeout`.
The stream duration starts after subscription. Ctrl+C closes the connection.

| Exit code | Meaning |
| --- | --- |
| 0 | Scan completed, fixture decoded, or stream completed with recent usable data |
| 2 | Invalid arguments, file or setup failure |
| 3 | Stream ended with unusable/stale data, an unexpected disconnect, or a cleanup error |
| 130 | Stopped with Ctrl+C |

An empty device list is a valid scan result.

## Follow a notification

1. `ble.scan()` discovers native devices. The CLI selects by name or explicit OS ID.
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

Each notification carries one complete message. The raw, quaternion v1 and
telemetry v2 bounds are 150, 153 and 201 bytes. They require ATT MTU at least
153, 156 and 204 respectively to fit the maximum.
See [the device contract](../protocol/ORIENTATION.md) and
[the firmware acceptance checklist](../firmware/ORIENTATION.md).

## Optional API

Select `TELEMETRY_SOURCE=simulated` for a hardware-free demo. Hardware is the
default. For hardware, export the exact `DEVICE_ID` from a scan. Select
`DEVICE_FORMAT=both` for raw IMUs on 8888 and fused orientations on 8889 over
one BLE connection. The single-stream settings are `quaternion` for posture
and `raw` for raw diagnostics. Use one Uvicorn worker and stop CLI streaming
before starting the hardware API.

```sh
# From backend/ with the environment active
python -m pip install -r requirements-api.txt
TELEMETRY_SOURCE=simulated python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In PowerShell, set `$env:TELEMETRY_SOURCE = "simulated"` before running the
Uvicorn command. The simulation starts upright and repeats a 20-second cycle.
It stays upright for the first 10 seconds of each cycle, then bends and returns.
Hardware startup resolves the wearable with a scan and reports setup errors.

To capture neutral, stand still in an upright pose or wait for the simulation's
upright phase, then call the API.

```sh
curl -X POST http://127.0.0.1:8000/posture/calibrate
```

The response reports `status: "captured"` and the reference frame's receipt
timestamp. HTTP 409 means the source lacks a fresh frame with all four valid
orientations. Call the same endpoint to recenter. Set `POSTURE_WARNING_DEG`
and `POSTURE_WARNING_SECONDS` to adjust the demo defaults of 20° and 2 seconds.

`GET /health` reports service status and `GET /diagnostics` reports reader
status and counters. `WS /ws/telemetry` publishes a shared latest snapshot at
about 20 Hz with a one-second send deadline per client. Both sources use
`WorkerTelemetry` 2.0 and the same posture processing. Raw mode leaves
orientations unavailable. Fresh v2 environment readings populate the
environment object with nulls for missing
fields. A stale or missing environment block maps to null. Device metadata and
per-IMU acquisition times are null. Fall state, EEG, heart rate and overall risk
remain unavailable. See [application telemetry](../docs/TELEMETRY_PROTOCOL.md)
for endpoint responses, joint quaternions, freshness and warning fields.

### Laptop fusion from raw BLE

For a wearable that streams raw packets on 8888, use the explicit laptop demo
mode. From `backend/` with the API dependencies installed:

```sh
TELEMETRY_SOURCE=hardware DEVICE_ID="<exact OS id>" DEVICE_FORMAT=raw_fusion python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In PowerShell, set `$env:TELEMETRY_SOURCE`, `$env:DEVICE_ID`, and
`$env:DEVICE_FORMAT` before running the same Uvicorn command. Stop CLI
streaming first. The backend subscribes to 8888 and uses the device timestamp
to process each raw frame. The dashboard still receives the same JSON on
`ws://127.0.0.1:8000/ws/telemetry` at about 20 Hz.

1. Start with the wearable upright and still for at least half a second.
   The estimator uses this interval to estimate gyro bias and align tilt with
   gravity. Wait until the dashboard has four valid orientations.
2. Capture the frontend's neutral pose while still upright. Use
   `POST /posture/calibrate`, or the dashboard's calibration control.
3. Bend slowly forward and sideways to confirm each segment moves in the
   expected direction. If axes differ by board, set `RAW_IMU_AXES` and
   restart the backend. The setting takes four semicolon-separated maps in
   pelvis, lumbar, thoracic, upper thoracic order. Each map lists the raw axes
   for wearer-right, up, and posterior.

The default map for each board is `-y,z,-x`, matching the firmware's assumed
forward/left/up board convention. For boards whose raw axes already match the
wearer frame, use `RAW_IMU_AXES="x,y,z;x,y,z;x,y,z;x,y,z"`. Check that an upright
board maps gravity to about `(0, +9.8, 0)` m/s². The map must be right-handed.
Confirm mounting alignment with a slow bend, since the stationary reading
alone cannot determine heading.

This six-axis estimate uses gyro integration with a gentle gravity correction.
Heading can drift and quick motion can disturb the tilt estimate. A raw
timestamp gap over 100 ms, a device restart, or loss of an initialized sensor
clears the neutral reference. Hold still again and recapture. If the pose stays
unavailable, check `GET /diagnostics` and confirm that device timestamps
advance and all four sensors provide usable readings.
[Complementary-filter reference](https://users.cecs.anu.edu.au/~Robert.Mahony/Mahony_Robert/2006_MahHamPfl-C68.pdf)

### Raw IMU handoff for detector code

With `DEVICE_FORMAT=both`, the usual `GET /diagnostics` counters track fused
packets and the nested `raw` counters track raw packets. Check each stream's
`valid_frame_hz`, `dropped`, and `rejected` values to measure what arrives.
The two subscriptions share a bounded 32-packet receive queue.

Detector code can supply a callback when creating the API app.

```python
from app.main import create_app
from my_detector import accept_raw_sample  # Your detector-side callback.

app = create_app(on_raw_sample=accept_raw_sample)
```

Run Uvicorn with that module's `app` and the dual-stream environment settings.
The callback receives each decoded raw `Sample` as the backend consumes it.
It carries `sample_time_ms` since device boot and four ordered IMUs with
`accel` in m/s² including gravity and `gyro` in rad/s. Sensor IDs 1–4 are
pelvis, lumbar, thoracic, and upper thoracic. Keep the callback quick and put
expensive processing in a separate worker with a bounded queue.
If the callback raises, the API logs the error and disables it for that session.

For `FallDetectorStream.push_batch()`, use device timestamps in seconds
and arrays shaped `(N, 4, 3)` for both accel and gyro. Handle uptime wrap and
restart on the detector side. Apply the confirmed sensor-to-segment remap
once before processing. Fused packets independently supply posture and
calibration. The 20 Hz dashboard publication clock does not pace the raw
callback. See the [fall detector input contract](../detection/models/README.md#2-inputs).

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
