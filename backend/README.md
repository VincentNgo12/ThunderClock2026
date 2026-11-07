# ThunderClock backend

The BLE diagnostic CLI discovers the wearable, streams signed raw readings or
quaternions, and inspects synthetic packets offline. The optional FastAPI adapter
shares the same reader and publishes dashboard snapshots over WebSockets.

Use Python 3.11 or later. Install `requirements.txt` for the CLI,
`requirements-api.txt` for the optional API, or `requirements-dev.txt` for the
test and generation tools.

See [DIAGNOSTICS.md](DIAGNOSTICS.md) for setup, commands, and hardware acceptance.
