# ThunderClock backend

The BLE diagnostic CLI discovers the wearable, streams signed raw readings or
quaternions, and inspects synthetic packets offline. The optional FastAPI service
processes live BLE or explicitly selected synthetic telemetry. It captures a
neutral reference and publishes joint quaternions, angles and sustained-bend
warnings over WebSockets.

Use Python 3.11 or later. Install `requirements.txt` for the CLI,
`requirements-api.txt` for the optional API, or `requirements-dev.txt` for the
test and generation tools.

## BLE checks

From `backend/` with your Python environment active, install the CLI
dependencies and run the automated BLE tests and offline fixture checks.

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_ble_cli.py" -v
python -m app.cli decode ../protocol/fixtures/raw_signed.bin --format raw
python -m app.cli decode ../protocol/fixtures/telemetry_v2.bin --format quaternion
```

For a live wearable check, stop any API session using the wearable, then
scan and copy its exact OS device ID. Run the stream command for the format
your firmware exposes.

```sh
python -m app.cli scan --timeout 5
python -m app.cli stream --device "<exact OS id>" --format raw --duration 10
python -m app.cli stream --device "<exact OS id>" --format quaternion --duration 15
```

See [DIAGNOSTICS.md](DIAGNOSTICS.md) for setup, the posture demo, and hardware
acceptance.

Use `DEVICE_FORMAT=raw_fusion` to estimate posture on the laptop from raw BLE
packets. [Setup and mounting checks](DIAGNOSTICS.md#laptop-fusion-from-raw-ble)

Frontend integration uses the [API and WebSocket contract](../docs/TELEMETRY_PROTOCOL.md)
and [complete JSON examples](../docs/telemetry-examples.json).

## Spine view: gravity model, calibration, posture, recordings

The 6-axis sensors' heading drifts tens of degrees per minute, so the spine view and posture use only
the direction of gravity in each sensor (`app/segments.py`). That needs each sensor's **mounting** on the
vest, measured once per strip:

1. Connect the strip, press **Calibrate…** and follow the prompts (stand, bend forward; backward/left/right
   recommended). The backend captures the poses (`POST /mounting/capture`), fits and saves the mounting
   to `mountings/<device>.json` (`POST /mounting/fit`). It is reloaded on every start.
2. After a reconnect the backend re-zeroes "upright" automatically once the wearer stands still for 2 s;
   **Re-zero upright** (`POST /mounting/zero`) does it on demand. Recalibrate only if the sensors move.

Posture (`app/trunk_posture.py`): trunk inclination against gravity at the upper back, zones 0-20 /
20-60 / > 60 deg forward, extension > 10 deg, side bend > 15 deg, sustained for `POSTURE_WARNING_SECONDS`
(2 s); lying is reported as a state.

Recordings (`recordings/`, gitignored):

```sh
python tools/recording_summary.py recordings/<file>.jsonl          # quick check
python tools/analyze_recording.py recordings/<file>.jsonl          # firmware tilt, drift, per-hold directions
python tools/analyze_recording.py <file> --export-mounting m.json  # MOUNTING_FILE=m.json for live use
python tools/replay_visual_check.py <file> --out /tmp/check [--mode panel] [--calibrate]
                                                                   # what the frontend draws, per hold
```

**Replay** in the dashboard plays a recording through the live pipeline (spine, posture, detection), with
pause, seek and jump-to-step; `TELEMETRY_SOURCE=replay REPLAY_FILE=<file>` does the same at startup.
