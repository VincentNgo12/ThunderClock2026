# ThunderClock wearable device boundary

The device encoding uses Protobuf with nanopb. The backend decodes the raw and
quaternion schema, including v2 environment and pulse oximeter fields. Firmware encoding and BLE
publication remain planned.
See [ORIENTATION.md](../protocol/ORIENTATION.md) for fields, units, coordinate
conventions and packet bounds.

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

Each notification carries one complete packet. The raw, quaternion v1 and
telemetry v2 bounds are 150, 153 and 201 bytes. Supporting those bounds takes
ATT MTU at least 153, 156 and 204 respectively. Verify the negotiated capacity
on the target laptop.
Suppress an oversized packet and report the condition.

## Remaining agreements

- Confirm mux channels, physical placement, axis remaps and startup heading.
- Confirm environmental expiry limits and the relative noise reference/window.
- Confirm freshness policy and preserve original local receipt times.
- Define any future command/result encoding, correlation and bounded execution.
- Agree firmware security, pairing and a device-side bond-reset procedure.

Preserve units, timing and explicit synthetic provenance. Use schema encoding
for wire data. Measurement validation requires more than successful decoding.
