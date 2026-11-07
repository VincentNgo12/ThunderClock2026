# ThunderClock telemetry

Two interfaces connect the wearable, laptop and dashboard. The backend
implements device decoding, posture processing and the application snapshot
described below.

## Device transport

The BLE reader accepts bounded Protobuf packets. Version 2 of the fused packet
adds optional environment readings to the four orientations. Firmware owns calibration,
fusion, mounting transforms and device validity. Backend code owns receipt,
decoding and validation.

See [the device boundary](FIRMWARE_PROTOCOL.md) and
[the orientation packet](../protocol/ORIENTATION.md) for fields, units,
coordinate conventions and BLE delivery targets.

## Application telemetry

The optional API publishes `WorkerTelemetry` JSON with `schema_version: "2.0"`
over `/ws/telemetry` at approximately 20 snapshots per second. It uses one
hardware or explicitly selected synthetic source and publishes four named IMU
slots. Fresh, usable quaternion samples populate orientation values. Raw mode
leaves orientations unavailable.

The diagnostic snapshot leaves `device`, per-IMU `acquired_us` and unsupported
output groups null. The JSON publication counter and UTC time come from the
backend. BLE counters and collection uptime belong to the device clock.
The JSON schema version is independent of the BLE packet revision.
System state is degraded while streaming and offline otherwise. Simulated
snapshots set `connection.source` and available freshness sources to
`simulated`. The hardware source reports hardware data and connection errors.
Keep packet encoding and sensor registers behind the backend boundary.

### Envelope and connection state

| Field | Planned meaning |
| --- | --- |
| `schema_version` | Explicit application contract version |
| `timestamp` | Publication UTC in ISO 8601 format |
| `sequence` | Shared application publication counter |
| Worker identity | One functional worker, T'Worker 1 |
| Source | hardware, simulated or replay |
| ESP32 / OpenBCI link | disconnected, connecting, connected or simulated |
| System state | nominal, degraded or offline |

Publication time is separate from acquisition time and original receipt time.
A browser reconnect shares ongoing acquisition and publication state.

### Availability and provenance

Each IMU and each application output group needs explicit freshness metadata.
Planned groups are posture, fall state, heart rate, environment, EEG and overall
risk.

| Metadata | Meaning |
| --- | --- |
| Status | valid, unavailable, invalid or stale |
| Source | hardware, simulated, replay or null |
| Receipt time | Original backend UTC receipt time or null |
| Age in ms | Age known from available device timing and local monotonic receipt time, or null |

Only valid measurements carry non-null values. Include required slots and use
null consistently for missing measurements.

Retain original monotonic receipt times across queues and shared state.
UTC clock adjustments must leave freshness unchanged. When device timing
supports sample age, add it once to elapsed local receipt time. The initial
orientation packet supplies strip uptime only, so use local receipt age and
keep unknown pre-receipt age explicit.

Start with a configurable 500 ms backend demo freshness threshold. The proposed
firmware encoding limit is 50 ms. These are separate policies with separate
clocks. Device loss marks affected data stale. Keep the source label visible
and require an explicit selection for simulated fallback.

### Orientation and posture

- Four named regions are pelvis, lumbar, thoracic and upper_thoracic.
- Quaternion order is `[w, x, y, z]`, with finite normalized values.
- Device validity and local freshness must both permit use.
- Use the coordinate convention in
  [ORIENTATION.md](../protocol/ORIENTATION.md) and confirm physical mounting.

`POST /posture/calibrate` captures one current frame with four valid
orientations. Ask the wearer to stand upright and still. The backend stores
the reference in memory. It clears the reference at service restart or a
detected wearable restart. Capture returns `calibration` with `status:
"captured"` and `captured_at` set to the frame's UTC receipt time.
An unavailable source, stale frame, raw frame or missing IMU returns HTTP 409
and preserves any existing reference. The same endpoint recenters the pose.

The backend expresses each upper segment in the lower segment's frame and
compares that rotation with the neutral reference. The pairs are pelvis to
lumbar, lumbar to thoracic, and thoracic to upper thoracic. It projects each
joint's neutral-relative up vector into the sagittal and lateral planes.
This estimates local bending and cancels motion shared by both segments.

`posture` contains `lumbar_flexion_deg`, `thoracic_flexion_deg`, and
`upper_thoracic_flexion_deg`, with positive values for forward bend and
negative values for extension. The same joint prefixes have
`_lateral_bend_deg` fields with positive values toward the wearer's left.
Posture is null until calibrated and whenever the complete frame is unavailable
or stale. The snapshot carries `calibration` even when posture is unavailable.

### Movement and acute events

Use the IMUs for motion and fall/impact context. Agree event states, impact
metrics, confirmation and timeline fields before adding detection. Distinguish
ordinary motion and sustained posture from an acute event.

### Environment

| JSON field | Source | Unit / meaning |
| --- | --- | --- |
| `temperature_c` | AHT21 | degrees Celsius |
| `humidity_percent` | AHT21 | percent relative humidity, 0–100 |
| `air_quality_index` | ENS160 | AQI-UBA category from 1 to 5 |
| `eco2_ppm` | ENS160 | Estimated CO2 equivalent in parts per million |
| `tvoc_ppb` | ENS160 | Estimated total VOC in parts per billion |
| `noise_relative_db` | INMP441-derived | Relative demo trend |

For a fresh v2 packet with any environment readings, the backend copies those
values into `environment` using the fields above. Absent fields are null.
`freshness.environment` uses the packet's local receipt time and source and
reports `valid` when at least one reading is available. A stale packet produces
`environment: null` and `stale` freshness. A packet with no readings produces
`environment: null` and `unavailable` freshness. Raw and v1 packets have no
environment readings. Each packet stands on its own, so missing fields do not
carry forward values from earlier packets.

Keep relative-noise qualification visible. Raw speech/audio storage is outside
the demo requirements.

### Physiology and EEG

BLE v2 can carry MAX30102 heart rate in beats per minute and estimated SpO₂
as a percentage. The diagnostic CLI exposes present values. Publishing these
signals in dashboard snapshots is a separate integration step.
[BLE field definitions](../protocol/ORIENTATION.md#quaternion-packet)

OpenBCI Cyton connects independently through its official dongle and BrainFlow
on the laptop. Plan signal quality, artifact context and compact downsampled
waveforms alongside experimental alertness/fatigue indicators. Agree channels,
rates and exact fields during EEG implementation.

### Risk and alerts

Target overall-risk labels are low, moderate, elevated and high. They represent
prototype classifications. Planned alert severities are normal, advisory,
warning and critical.

`alerts` lists active posture warnings for the snapshot. A joint qualifies
after its larger absolute bend angle stays at or above `POSTURE_WARNING_DEG`
(default 20°) for `POSTURE_WARNING_SECONDS` (default 2 s). The backend measures
the hold with monotonic receipt times. A gap over 500 ms, a missing joint,
recapture or a return below threshold clears the hold. Stale snapshots show
an empty list. Each entry has `severity: "warning"`, `category: "posture"`,
`region`, `summary`, `active: true`, and `sustained_seconds`.
EEG and overall risk remain unavailable.

## Publication and versioning

Use one backend-owned latest state shared by browser clients. Bound each
client's pending snapshot and send time so slow consumers can skip snapshots.
Device acquisition and browser connection lifecycles stay independent.

Ignore compatible additive fields. Changes to names, units, required values or
meanings need coordinated schema versioning and backend/frontend updates.
Generate fixtures covering live, simulated, partial, invalid and stale states
when adding the contract.

All measurements and derived outputs are hackathon prototype indicators.
Visually interpolated geometry, relative noise and experimental EEG need visible
measurement context. Do not make medical or clinically validated claims.
