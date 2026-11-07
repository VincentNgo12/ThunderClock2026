# ThunderClock wearable device boundary

The planned device encoding is Protobuf with nanopb. Detailed fields, units,
coordinate conventions and BLE size targets are in
[ORIENTATION.md](../protocol/ORIENTATION.md). Schema files and bindings will be
added with the implementation.

## Responsibilities

- Firmware owns gyro calibration, sensor fusion and sensor-to-segment transforms.
- Backend owns BLE receipt, explicit decoding, validation and local freshness.
- Browser clients consume backend application telemetry over WebSockets.
- Keep connected, subscribed, received, decoded and measurement-valid states separate.
- Worker neutral-reference capture belongs to the backend.

## BLE delivery targets

Use service AAAA with raw characteristic 8888 and quaternion characteristic 8889
for the prototype profile. Both support READ and NOTIFY. Confirm the actual device
profile before connecting. Custom 128-bit UUIDs remain a team decision.

Each notification carries one complete packet. Target bounds are 150 bytes for
raw and 153 bytes for quaternion, requiring ATT MTU at least 153 and 156.
Verify these bounds against generated code and the negotiated transport capacity.
Suppress an oversized packet and report the condition.

## Remaining agreements

- Confirm mux channels, physical placement, axis remaps and startup heading.
- Define environmental/audio/PPG payloads when their acquisition is implemented.
- Confirm freshness policy and preserve original local receipt times.
- Define any future command/result encoding, correlation and bounded execution.
- Agree firmware security, pairing and a device-side bond-reset procedure.

Preserve units, timing and explicit synthetic provenance. Use schema encoding
for wire data. Measurement validation requires more than successful decoding.
