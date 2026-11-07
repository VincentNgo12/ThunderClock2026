# ThunderClock orientation protocol

The device-side contract for four-IMU diagnostics is defined in
[orientation.proto](orientation.proto), package `telemetry.orientation.v1`.
The backend decodes this schema. Firmware integration and physical validation
remain acceptance tasks.

## BLE profile and raw measurements

The prototype profile targets `THUNDER CLOCK`, service AAAA, raw characteristic
8888 and quaternion characteristic 8889. Each characteristic supports READ and
NOTIFY. Confirm the installed profile and negotiated capacity on the laptop.

Raw acceleration uses m/s² and gyro uses rad/s in sensor-board axes. The raw
schema is [telemetry.proto](../firmware/proto/telemetry.proto). Keep raw and quaternion
formats explicitly selected and on separate characteristics. A raw format
without sensor validity must suppress a strip when a required reading is missing
or stale.

## Quaternion packet

| Field | Tag / type | Meaning |
| --- | --- | --- |
| `version` | 16 / uint32 | Explicit version 1 |
| `sequence` | 17 / uint32 | Produced snapshot counter |
| `sample_time_ms` | 18 / uint32 | End-of-strip collection uptime in milliseconds |
| `orientations` | 19 / repeated message | Exactly four identified sensor slots |

Use explicit presence for header scalars, sensor ID and validity. Nanopb
`has_*` flags must be set even for sequence/time zero or valid=false.
Each orientation uses tag 1 for `sensor_id`, 2 for `valid` and 3 for `rotation`.

Logical IDs are 1 pelvis, 2 lumbar, 3 thoracic and 4 upper thoracic. Each appears
once and order is irrelevant. Mux channels 0–3 are the planned acquisition slots.
Confirm physical placement before using these names for posture.

For an unavailable sensor send `valid=false` and omit rotation while retaining
the ID slot. A valid rotation must be present, finite and unit length within
0.001. Backend validation must reject arbitrary non-unit inputs.

Quaternion component order is `[w, x, y, z]`. Proto3 scalar components may use
zero defaults. An absent rotation has different meaning from identity.
Identity may encode only w=1. Zero-norm rotations are invalid.

## Coordinate convention

Use a right-handed application frame with +X wearer-right, +Y up and +Z posterior.
Each quaternion rotates segment-local vectors into the common world frame.

Firmware owns gyro calibration, fusion and sensor-to-segment mounting transforms.
The proposed Fusion body frame is +X forward, +Y left and +Z up.
Changing both local and world bases to the application convention maps
`(w, x, y, z)` to `(w, -y, z, -x)`.

Apply a proper mounting remap to both accelerometer and gyro inputs. Confirm
board axes, mounting and measured rotations with a jig. Axis-aligned identity
defaults require physical verification. Backend owns later neutral-reference
capture and relative posture interpretation.

Six-axis fusion cannot observe absolute heading. Align sensor startup headings
and measure stationary and relative-heading drift.

## Timing and ordering

FIFO samples have estimated acquisition times based on nominal 100 Hz sampling
and local monotonic drain intervals. Sequential sensor reads are approximately
simultaneous. One strip timestamp cannot establish synchronized acquisition.

The proposed firmware validity limit is 50 ms. Report sample loss and reset the
affected filter when timing or FIFO integrity fails.

Sequence and strip uptime wrap modulo 2^32. A reader should count forward gaps,
accept wraps, reject duplicate/backwards sequences and treat backwards uptime
outside wrap as a possible restart. A new connection resets ordering state.
Without a boot identifier some restart/order ambiguities need reconnection.

Backend freshness uses original local monotonic receipt time including queue
residence. Radio delay and sample age before receipt remain unknown.

## Generation and delivery

The backend generator writes Python and nanopb bindings to
`backend/app/generated/` and `protocol/generated/`. Make schema changes in
the source files and regenerate. The nanopb host conformance check uses
runtime 0.4.9.1.

| Payload | Generated maximum bytes | Minimum unfragmented ATT MTU |
| --- | ---: | ---: |
| Raw strip | 150 | 153 |
| Four-sensor quaternion strip | 153 | 156 |

A preferred MTU of 517 alone cannot establish usable capacity. Each notification
must carry one complete packet. Suppress and diagnose packets exceeding the
actual MTU minus three. Keep framing simple and bounded.

The generator writes host vectors for identity, signed raw values, partial
availability and maximum encoded size. Verify Python/C conformance, negotiated
BLE capacity and physical accuracy separately.

New fields need new tags, regenerated bindings and updated size checks.
Reserve deleted tags and names. Compatible optional additions may retain
version 1. Changed units, meanings, required fields or enum semantics need a
coordinated version change.

See [firmware acquisition and acceptance](../firmware/ORIENTATION.md) and
[backend diagnostics](../backend/DIAGNOSTICS.md).
