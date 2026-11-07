# ThunderClock BLE telemetry contract

The authoritative wearable schema is [orientation.proto](orientation.proto).
Version 2 carries four fused orientations and optional environmental and
pulse oximeter readings.
Version 1 carries the same orientations. The Protobuf package and message name
stay stable for binding compatibility. The packet's `version` field selects the
wire revision. The backend accepts both revisions.

## BLE profile and raw measurements

The prototype profile targets `THUNDER CLOCK`, service AAAA, raw characteristic
8888 and fused telemetry characteristic 8889. Both support READ and NOTIFY.
Use v1 on 8889 for readers that only support orientations. Enable v2 with a
reader that accepts it and a link that fits the complete packet.

Raw acceleration uses m/s² and gyro uses rad/s in sensor-board axes. The raw
schema is [telemetry.proto](../firmware/proto/telemetry.proto). Keep raw and quaternion
formats explicitly selected and on separate characteristics. A raw format
without sensor validity must suppress a strip when a required reading is missing
or stale.

## Quaternion packet

| Field | Tag / type | Meaning |
| --- | --- | --- |
| `version` | 16 / uint32 | Explicit version 1 or 2 |
| `sequence` | 17 / uint32 | Produced snapshot counter |
| `sample_time_ms` | 18 / uint32 | End-of-strip collection uptime in milliseconds |
| `orientations` | 19 / repeated message | Exactly four identified sensor slots |
| `environment` | 20 / message | Optional in v2 and absent in v1 |
| `heart_rate_bpm` | 21 / optional float | Finite positive heart rate in beats per minute |
| `spo2_percent` | 22 / optional float | Estimated peripheral oxygen saturation from 0 to 100% |

Heart rate and SpO₂ are independently optional in v2 and absent in v1.
Omit a reading when it is unavailable, invalid or stale. In nanopb, set
`has_heart_rate_bpm` and `has_spo2_percent` only for usable values. The
snapshot timestamp applies to both readings.

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

## Environment

Each field is optional and individually present when usable at packet
assembly. Missing, faulty or stale values are omitted. An empty or absent
environment block means all readings are unavailable. An explicitly present
zero is a measurement where the field's range permits it. Floats must be finite.

| Field | Tag / type | Source and units |
| --- | --- | --- |
| `temperature_c` | 1 / float | AHT21 temperature in °C |
| `humidity_percent` | 2 / float | AHT21 relative humidity from 0 to 100% |
| `air_quality_index` | 3 / uint32 | ENS160 AQI-UBA category from 1 to 5 |
| `eco2_ppm` | 4 / uint32 | ENS160 equivalent CO₂ estimate from 400 to 65,000 ppm |
| `tvoc_ppb` | 5 / uint32 | ENS160 estimated total VOC from 0 to 65,000 ppb |
| `noise_relative_db` | 6 / float | INMP441 relative level in dB |

ENS160 estimates use the sensor's reported ranges. Compare relative noise
using the same microphone reference and window. The firmware team still needs
to set those parameters and the freshness limit for each slow sensor.
[ENS160 datasheet, section 5](https://www.sciosense.com/wp-content/uploads/2023/12/ENS160-Datasheet.pdf)

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
Environmental readings are the latest usable values at collection time.
Firmware uses its own acquisition times to omit stale values. The packet
carries one boot-relative time for the whole snapshot.

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
| Version 1 orientations | 153 | 156 |
| Version 2 telemetry | 201 | 204 |

Each notification carries one complete packet. The payload budget is the
negotiated ATT MTU minus three bytes. The default MTU 23 allows 20 bytes.
Suppress and diagnose oversized packets. Verify negotiation on the target
ESP32 and laptop before enabling v2 streaming.
[Bluetooth ATT notification format](https://www.bluetooth.com/wp-content/uploads/Files/Specification/HTML/Core-61/out/en/host/attribute-protocol--att-.html)

The synthetic v2 fixture uses 102 bytes and includes both pulse oximeter
readings and a present zero TVOC value.
The 201-byte maximum fixture uses out-of-range integers to exercise nanopb
allocation. Its size is a conservative bound. See the backend guide to
regenerate bindings and run the Python and C checks.
[Regeneration and validation](../backend/DIAGNOSTICS.md#tests-and-generation)

Use new tags for optional fields and reserve removed tags and names. Regenerate
both bindings and fixtures, update decoder and buffer limits, and check the
resulting MTU budget. Coordinate a version bump for changes to units, meanings,
required fields, or packet bounds that older readers cannot accept.

See [firmware acquisition and acceptance](../firmware/ORIENTATION.md) and
[backend diagnostics](../backend/DIAGNOSTICS.md).
