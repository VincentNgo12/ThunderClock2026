# ThunderClock frontend telemetry contract

The browser talks to the Python API. The API decodes BLE Protobuf or generates
an explicitly selected synthetic stream and publishes application JSON. The
JSON schema version is independent of the BLE packet version.
[BLE device contract](../protocol/ORIENTATION.md)

## Connect and try the demo

From `backend/` with Python 3.11 or later and a virtual environment active:

```sh
python -m pip install -r requirements-api.txt
TELEMETRY_SOURCE=simulated python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In PowerShell, set `$env:TELEMETRY_SOURCE = "simulated"` and then run the Uvicorn
command. Open `ws://127.0.0.1:8000/ws/telemetry`. The backend sends snapshots
immediately. The browser sends no subscription message. If the deployment
terminates HTTPS, connect through its `wss://` URL.

The simulated spine starts upright for 10 seconds, then bends and returns over
a 20-second loop. Capture neutral while upright with
`curl -X POST http://127.0.0.1:8000/posture/calibrate`. The separate 90-second
synthetic environment scenario cycles through changing site conditions.
Use one API worker so all clients share the same neutral reference.
The default HTTP CORS origins are `http://localhost:5173` and
`http://127.0.0.1:5173` for GET and POST. The local API has no login or token.

For hardware, use `TELEMETRY_SOURCE=hardware` (the default), the exact scan
`DEVICE_ID`, and `DEVICE_FORMAT=quaternion` or `both`. Raw-only mode supplies
diagnostics while posture is unavailable. Startup errors such as a missing
device ID or an unresolved scan prevent the API from starting. A connection
or subscription failure after startup is visible in diagnostics and
telemetry. Switching to simulation requires selecting it at startup.
[Hardware setup and BLE checks](../backend/DIAGNOSTICS.md#optional-api)

## HTTP and WebSocket endpoints

| Method and path | Request | Success |
| --- | --- | --- |
| `GET /health` | None | HTTP 200 with `{"status":"ok","source":"hardware"}` or source `simulated`. Checks API liveness. |
| `GET /diagnostics` | None | HTTP 200 with the diagnostic counters below. |
| `POST /posture/calibrate` | Empty body | HTTP 200 with the `Calibration` object also included in WebSocket snapshots. Use for both capture and recenter. |
| `WS /ws/telemetry` | Open socket | One `WorkerTelemetry` JSON object per text message, at about 20 Hz. |
| `GET /docs`, `GET /redoc`, `GET /openapi.json` | None | FastAPI HTTP documentation and schema. |

`POST /posture/calibrate` captures the latest complete, fresh quaternion frame.
Ask the wearer to stand upright and still before calling it. Success is
confirmed by the HTTP response. It returns HTTP 409 with `{"detail":"..."}`
when capture is unavailable:

| `detail` | Condition |
| --- | --- |
| `Neutral capture requires an active telemetry source` | Source has not subscribed or has stopped. |
| `Neutral capture requires a fresh quaternion frame` | No frame or receipt age greater than 500 ms. |
| `Neutral capture requires all four valid IMU orientations` | Raw-only frame or incomplete orientation strip. |

A failed capture preserves the previous reference. Capture clears hold timers.
The reference lives in memory and clears on service restart or a detected
wearable restart. An ordinary missing or stale frame preserves the reference.
FastAPI uses `{"detail":"Not Found"}` for an unknown HTTP path and
`{"detail":"Method Not Allowed"}` for a method the path does not support.

WebSocket messages are latest snapshots without history. Clients may receive
the same snapshot more than once or skip publication sequence values. A send
that takes over one second closes the socket with code `1013`. On reconnect,
reset sequence tracking and use the latest snapshot. Track WebSocket
connectivity separately from `connection.esp32`, which describes the device.

## WebSocket payload

[Complete JSON examples](telemetry-examples.json) contains a calibrated
synthetic bend with warnings and a hardware-shaped partial frame after
calibration. Each named value is one complete WebSocket message.

```ts
type Source = "hardware" | "simulated"
type Region = "pelvis" | "lumbar" | "thoracic" | "upper_thoracic"
type Joint = "lumbar" | "thoracic" | "upper_thoracic"
type Quaternion = { w: number; x: number; y: number; z: number }
type JointRotations = Record<Joint, Quaternion>

type Freshness = {
  status: "valid" | "stale" | "unavailable"
  source: Source | null
  received_at: string | null
  age_ms: number | null
}

type Calibration =
  | { status: "uncalibrated"; captured_at: null; reference: null }
  | {
      status: "captured"
      captured_at: string
      reference: { root_orientation: Quaternion; joint_rotations: JointRotations }
    }

type Posture = {
  lumbar_flexion_deg: number
  thoracic_flexion_deg: number
  upper_thoracic_flexion_deg: number
  lumbar_lateral_bend_deg: number
  thoracic_lateral_bend_deg: number
  upper_thoracic_lateral_bend_deg: number
  joint_rotations: JointRotations
}

type Environment = {
  temperature_c: number | null
  humidity_percent: number | null
  air_quality_index: number | null
  eco2_ppm: number | null
  tvoc_ppb: number | null
  noise_relative_db: number | null
}

type PostureAlert = {
  severity: "warning"
  category: "posture"
  region: Joint
  summary: string
  active: true
  sustained_seconds: number
}

type WorkerTelemetry = {
  schema_version: "2.0"
  timestamp: string
  sequence: number
  device: null
  system_status: "degraded" | "offline"
  connection: {
    source: Source
    esp32: "connected" | "disconnected" | "simulated"
    openbci: "disconnected"
  }
  imus: {
    region: Region
    orientation: Quaternion | null
    acquired_us: null
    freshness: Freshness
  }[]
  posture: Posture | null
  calibration: Calibration
  alerts: PostureAlert[]
  environment: Environment | null
  fall_state: null
  heart_rate_bpm: null
  eeg: null
  overall_risk: null
  freshness: Record<"posture" | "fall_state" | "heart_rate" | "environment" | "eeg" | "overall_risk", Freshness>
}
```

`timestamp` is backend publication time in ISO 8601 UTC. `sequence` is a
nonnegative integer that increments per publication and starts over when the
service restarts. Each snapshot contains exactly four IMU entries in order of
sensor ID 1 through 4. Match them by `region`. JSON numbers are finite.
Readers may ignore additive fields within version 2.0.

`connection.source` identifies the selected source. `esp32` is `connected`
after BLE subscription, `simulated` while the synthetic stream runs, and
`disconnected` otherwise. `system_status` is `degraded` while streaming
because the full set of subsystems is incomplete, and `offline` otherwise.

### Availability rules

- `received_at` is the original backend receipt time in ISO 8601 UTC.
  `age_ms` is elapsed local monotonic time since receipt. Device uptime and
  radio transit time are separate. `device` and `acquired_us` are reserved
  nulls in this implementation.
- A valid live value has `status: "valid"` and receipt age at most 500 ms
  while its source is active. A previously usable frame that expires or
  loses its source has `status: "stale"` and a null live value.
- An IMU explicitly marked invalid in a decoded frame has a null
  orientation and `status: "unavailable"`, with that frame's receipt metadata.
  Absent output groups use unavailable freshness with null receipt metadata.
  A malformed BLE packet is rejected and cannot refresh the last accepted
  frame. That frame ages normally.
- `posture` and its joint rotations are available only with a captured
  reference and four fresh, valid orientations in the same frame. An
  incomplete frame can still supply valid individual IMU orientations.
  `calibration.reference` is historical data and can remain present while
  live `posture` is null.
- `alerts` lists current warnings. A joint warns when the larger absolute
  bend angle stays at or above `POSTURE_WARNING_DEG` (20° by default) for
  `POSTURE_WARNING_SECONDS` (2 s). The hold uses sample receipt times and
  clears on a gap over 500 ms, an incomplete frame, recapture, or return below
  threshold. Unavailable or stale posture publishes an empty array.

### Drive an articulated spine

The frame is right-handed with +X toward the wearer's right, +Y up, and +Z
posterior. Each `imus[].orientation` is the measured or simulated
segment-local-to-world quaternion, before neutral calibration. Quaternion
components are named `w`, `x`, `y`, `z`. In Three.js, construct them with
`new THREE.Quaternion(q.x, q.y, q.z, q.w).normalize()`. Use quaternion
interpolation that handles the equivalent signs `q` and `-q`.

| Joint key | Parent | Child |
| --- | --- | --- |
| `lumbar` | pelvis | lumbar |
| `thoracic` | lumbar | thoracic |
| `upper_thoracic` | thoracic | upper thoracic |

For each joint, the backend computes the current parent-local rotation
`J = inverse(qParentWorld) * qChildWorld`. Let `N` be that joint's captured
rotation in `calibration.reference.joint_rotations`. The live
`posture.joint_rotations` value is the unit delta `D = inverse(N) * J`.
A delta is identity at capture, even if the neutral joint is bent.

1. Use `imus.find(imu => imu.region === "pelvis").orientation` for the live
   root rotation. Normalize when converting the sensor value to a rig quaternion.
2. Assign each child bone the **local** rotation `N * D`. In Three.js,
   `quat(N).multiply(quat(D))` uses this order. In a flat scene, compute each
   child's world rotation as `qParentWorld * (N * D)` down the chain.
3. Use `calibration.reference.root_orientation` and each `N` to draw a neutral
   ghost. For the live articulated pose, use only snapshots with non-null
   posture and valid posture freshness. If posture becomes unavailable,
   hide or visibly mark the live pose as unavailable. The captured ghost
   can still be shown as a reference.
4. Derive positions from the rig's chosen segment lengths and local offsets.
   The backend supplies rotations. Positions between anchors are visualization
   geometry.

Flexion is positive forward and negative in extension. Lateral bend is positive
to the wearer's left. Angles are the projections of the neutral-relative up
vector. The rig uses the quaternions directly, including twist. Six-axis
orientation can drift in heading. Confirm physical mounting and axis alignment
before relying on a live pose. [Device coordinate convention](../protocol/ORIENTATION.md#coordinate-convention)

### Environment and feature coverage

| Field | Meaning |
| --- | --- |
| `temperature_c` | Temperature in °C |
| `humidity_percent` | Relative humidity from 0 to 100 percent |
| `air_quality_index` | ENS160 AQI-UBA category, integer 1–5 |
| `eco2_ppm` | Estimated CO₂ equivalent in ppm |
| `tvoc_ppb` | Estimated total VOC in ppb |
| `noise_relative_db` | Relative noise trend in dB |

A fresh v2 BLE or synthetic frame can supply an environment object. Missing
fields are null. The group is valid with at least one reading and null when
none is usable. Relative noise is a trend for the prototype.

The dashboard currently publishes IMU orientation, calibrated posture,
warnings and environment. BLE and simulation can produce heart rate and SpO₂
for other backend paths. Dashboard physiology, EEG, fall state and overall
risk are reserved for future integration. This is a prototype posture
approximation with physical validation pending.

## Diagnostics payload

All fields are present on `GET /diagnostics`. In `DEVICE_FORMAT=both`, the
top-level counters track fused packets and the optional `raw` object has the
same fields for raw packets. Both share one connection state. Simulated mode
counts generated frames.

| Field | Type | Meaning |
| --- | --- | --- |
| `state` | string | `disconnected`, `connecting`, `connected`, `subscribed`, `error`, or `simulated` |
| `received`, `decoded`, `usable` | integer | Incoming frames, accepted frames, and fully usable frames |
| `rejected`, `dropped` | integer | Rejected frames and frames evicted from the receive queue |
| `sequence_gaps`, `resets` | integer | Missing fused sequence values and probable wearable restarts |
| `started_at` | number | Monotonic seconds at start, zero before start |
| `finished_at`, `last_notification`, `last_valid` | number or null | Local monotonic seconds for completion, most recent notification, and most recent fully usable frame |
| `last_error` | string or null | Last reader error message |
| `last_valid_age_ms` | number or null | Elapsed local milliseconds since the last fully usable frame |
| `valid_frame_hz` | number | Fully usable frames divided by elapsed stream time |
| `raw` | optional object | Per-raw-stream diagnostics in dual mode |

```json
{
  "state": "subscribed",
  "received": 41,
  "decoded": 41,
  "usable": 40,
  "rejected": 0,
  "dropped": 0,
  "sequence_gaps": 0,
  "resets": 0,
  "started_at": 500.0,
  "finished_at": null,
  "last_notification": 502.05,
  "last_valid": 502.0,
  "last_error": null,
  "last_valid_age_ms": 60.0,
  "valid_frame_hz": 19.4
}
```

Raw usability describes the packet structure. Physical sensor health needs its
own assessment. Diagnostic monotonic timestamps are useful for durations
within the same process.
