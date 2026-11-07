# ThunderClock BLE diagnostic CLI plan

The first backend milestone is a standalone CLI for discovering the wearable,
inspecting signed raw readings and validating four orientation quaternions.
The commands and module names below are planned interfaces. Implement the CLI
before treating them as runnable setup instructions.

## Ownership and hardware profile

One reader owns one device connection. Stop CLI streaming before an optional API
takes ownership of the same wearable.

The prototype profile targets advertising name `THUNDER CLOCK`, service
`AAAA`, raw READ + NOTIFY characteristic `8888` and quaternion READ + NOTIFY
characteristic `8889`. Short UUIDs expand with the Bluetooth base UUID
`0000xxxx-0000-1000-8000-00805f9b34fb`. Confirm the installed firmware profile.

Select an exact OS device identifier from discovery. macOS may use an opaque UUID.
Resolve that identifier to the native BLE device before connecting.

## Planned commands

Target Python 3.11+. Add runtime requirements with the implementation.

```sh
# From backend/, after the CLI and requirements exist
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m app.cli scan --timeout 5
python -m app.cli stream --device "<exact OS id>" --format raw --duration 10
python -m app.cli stream --device "<exact OS id>" --format quaternion --duration 15
```

Provide explicit service/characteristic overrides for a known alternate profile.
Require an explicit raw or quaternion format. An optional JSONL recorder may
record consumed decoded samples, with visible drop counts and timing limits.

## Planned module boundaries

| Module | Responsibility |
| --- | --- |
| `app/models.py` | Plain samples, sensor identities and diagnostic counters |
| `app/protocol.py` | Explicit decoders, packet bounds and value validation |
| `app/ble.py` | Discovery, selection, one connection and bounded receipt |
| `app/cli.py` | Arguments, terminal display, recording and exit codes |
| `app/generated/` | Future generated bindings |
| `tools/` | Future schema generation and host conformance checks |

Keep BLE callbacks to bounded byte copies with original monotonic and UTC
receipt times. Decode, print and record in the consumer. A bounded receipt queue
may replace the oldest entry under pressure and must count drops.

Display connection and subscription independently. Track received notifications,
decoded frames, fully usable frames, malformed frames, drops and last-valid age.
Malformed input must leave the last-valid timestamp intact.

## Validation and timing

Raw readings use m/s² and rad/s in sensor-board axes. Preserve signed values and
gravity. Structurally valid raw data still needs physical sensor-health checks.

Quaternion strips need the expected version, four unique sensor IDs, explicit
validity and finite near-unit rotations. Keep unavailable sensors visible.
See [the device protocol](../protocol/ORIENTATION.md).

Use local monotonic receipt age including queue residence. Device uptime and
laptop UTC have separate meanings. Radio transit delay remains unknown.
Start with a configurable 500 ms freshness limit as a demo policy.

Use bounded setup/cleanup operations and make Ctrl+C release the subscription
and connection. A scan with zero results is a valid empty discovery result.
Report stale-at-end captures and unexpected disconnects explicitly.

## Hardware acceptance checklist

1. Confirm board power, advertising, OS Bluetooth permissions and exact device ID.
2. Connect and verify the expected service, characteristic and NOTIFY property.
3. Inspect complete notification lengths and negotiated MTU. Target raw and
   quaternion bounds require MTU at least 153 and 156 respectively.
4. Check four signed raw readings, approximately 9.81 m/s² stationary gravity,
   and plausible stationary gyro offsets.
5. Follow [firmware calibration](../firmware/ORIENTATION.md), then inspect finite
   near-unit quaternions and per-sensor availability.
6. Stop, restart, disconnect and use Ctrl+C to verify cleanup and freshness.
7. Record the OS/adapter, firmware profile, packet sizes and physical observations.

Synthetic fixtures will support decoder checks once schemas and tools exist.
Keep fixture results separate from physical acceptance observations.

## Later API boundary

An optional single-process FastAPI adapter may share the reader and expose
`/health`, `/diagnostics` and `/ws/telemetry`. Browser clients share one latest
snapshot with bounded sends. Unsupported measurements stay unavailable.

Posture, risk, environment, heart rate and EEG need their own implementations.
See [the planned application contract](../docs/TELEMETRY_PROTOCOL.md).
