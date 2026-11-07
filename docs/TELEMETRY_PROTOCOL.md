# ThunderClock telemetry

Two interfaces connect the wearable, laptop and dashboard. The diagnostic
backend implements device decoding and the application snapshot described
below. The remaining application sections describe demo targets.

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
over `/ws/telemetry` at approximately 20 snapshots per second. It shares one
hardware reader and publishes four named IMU slots. Usable, fresh quaternion
samples populate orientation values. Raw mode leaves orientations unavailable.

The diagnostic snapshot leaves `device`, per-IMU `acquired_us` and unsupported
output groups null. The JSON publication counter and UTC time come from the
backend. BLE counters and collection uptime belong to the device clock.
The JSON schema version is independent of the BLE packet revision.
Its system state is degraded while subscribed and offline
otherwise. Simulated and replay input are planned for the wider application
using the same application shape. Keep packet encoding and sensor registers
behind the backend boundary.

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
- Posture angles use degrees relative to a captured neutral pose.
- Target forward flexion, backward extension, lateral bending and sustained
  posture context. Agree exact derived fields when implementing posture logic.
- Use the coordinate convention in
  [ORIENTATION.md](../protocol/ORIENTATION.md) and confirm physical mounting.

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

MAX30102 heart rate is a supporting stretch signal in beats per minute.
Keep it unavailable until reliable readings exist. SpO2 needs a separate
reliability decision before inclusion.

OpenBCI Cyton connects independently through its official dongle and BrainFlow
on the laptop. Plan signal quality, artifact context and compact downsampled
waveforms alongside experimental alertness/fatigue indicators. Agree channels,
rates and exact fields during EEG implementation.

### Risk and alerts

Target overall-risk labels are low, moderate, elevated and high. They represent
prototype classifications. Planned alert severities are normal, advisory,
warning and critical.

Agree contextual thresholds before implementing alerts. Each alert should carry
severity, category, summary, active/cleared state and time. Posture, EEG and risk
remain unavailable until their respective algorithms exist.

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
