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

For a live wearable check, stop any API session using the wearable and select
the format your firmware exposes. The CLI discovers ThunderClock automatically.

```sh
python -m app.cli stream --format raw --duration 10
python -m app.cli stream --format quaternion --duration 15
```

To select a specific board, run `python -m app.cli scan --timeout 5` and pass
its exact OS ID with `--device`.

See [DIAGNOSTICS.md](DIAGNOSTICS.md) for setup, the posture demo, and hardware
acceptance.

Use `DEVICE_FORMAT=raw_fusion` to estimate posture on the laptop from raw BLE
packets. [Setup and mounting checks](DIAGNOSTICS.md#laptop-fusion-from-raw-ble)

Frontend integration uses the [API and WebSocket contract](../docs/TELEMETRY_PROTOCOL.md)
and [complete JSON examples](../docs/telemetry-examples.json).
