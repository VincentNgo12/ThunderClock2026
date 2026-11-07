# ThunderClock device protocol

The backend decodes two explicitly selected Protobuf formats. Each BLE
notification carries one complete message.

| Format | Schema | Characteristic |
| --- | --- | --- |
| Raw IMUs | [telemetry.proto](../firmware/proto/telemetry.proto) | 8888 |
| Wearable telemetry v2 and orientation-only v1 | [orientation.proto](orientation.proto) | 8889 |

[The BLE telemetry contract](ORIENTATION.md) defines field presence, units,
coordinates, and packet limits. `generated/` contains nanopb bindings. `fixtures/` contains
synthetic packets, their manifest, and a C conformance harness.

Use the [backend generation commands](../backend/DIAGNOSTICS.md#tests-and-generation)
to rebuild both sets of bindings and the fixtures from the schemas.
