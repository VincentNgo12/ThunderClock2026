# Product definition

## Product pitch

ThunderClock is a multimodal wearable worker-safety prototype for NatHacks 2026.
It combines posture, movement, environment, optional physiology, and experimental
EEG indicators in one responsive supervisor dashboard. The five-minute demo
should feel technically credible and visually exceptional while being honest
about what is measured, inferred, simulated, and not validated.

## Problem

Supervisors cannot easily see physical strain, changing environmental conditions,
and acute events in one place. Existing signals are fragmented and often become
useful only after an incident. ThunderClock explores whether a unified, real-time
view can make changing conditions easier to notice and discuss.

## Users and scope

The hackathon MVP has one user: a supervisor viewing one instrumented worker.
There is no worker-facing application in the MVP.

The architecture should leave room for future worker feedback, additional
sensors, multiple workers, historical analysis, and real deployments. The UI may
show muted or disabled selectors for `T'Worker 2` and `T'Worker 3` to communicate
that direction, but only `T'Worker 1` is functional. The prototype must not imply
that inactive workers are being monitored.

## Current hardware

### Wearable and posture

- Adafruit ESP32-S3 Feather, product #5477, using PlatformIO and Arduino
- Four simultaneously active MPU-6050 IMUs, with a fifth MPU-6050 as a spare
- PCA9548/TCA9548A I2C multiplexer
- Planned measured anchors are pelvis/root, lumbar, thoracic, and upper
  thoracic/shoulder region

Each MPU-6050 provides a three-axis accelerometer and gyroscope, with no
magnetometer. Accurate absolute yaw or twist is not a primary requirement. The
primary motion goals are forward flexion, backward extension, lateral bending,
sustained poor posture, and sudden movement/fall/impact context.

The product presents a **3D posture approximation** from four measured anchors.
It is not medically accurate anatomical spinal-curvature reconstruction. Any
curve rendered between anchors is visual interpolation, not additional measured
anatomical data.

### Environment and auxiliary sensing

- AHT21 for temperature and relative humidity
- ENS160 for air-quality-related AQI, eCO2, and TVOC estimates
- INMP441 I2S microphone for relative noise trends, not certified sound-level
  measurement; raw speech/audio does not need to be stored
- MAX30102 for heart-rate/PPG as a supporting stretch feature
- Existing IMUs as the primary source of fall/impact context

### EEG

OpenBCI Cyton connects to the laptop through its official wireless dongle.
BrainFlow will acquire it in the Python backend. Raw EEG never routes through the
ESP32. EEG output is an experimental alertness/fatigue indicator with visible
signal-quality and artifact context, not a diagnosis.

## Primary demo capabilities

- Acquire or simulate four normalized IMU orientation quaternions in globally
  standardized `[w, x, y, z]` order.
- Capture a naturally upright neutral pose and interpret later orientations
  relative to that reference.
- Show a dominant, responsive 3D posture approximation with four visible
  measured anchors and clearly visual interpolation.
- Show lumbar/thoracic posture, lateral/forward movement where available,
  sustained posture context, and fall/impact state.
- Group temperature, humidity, ENS160 values and relative noise.
- Show heart rate when reliable without making it demo-critical.
- Show a small EEG waveform area, experimental alertness/fatigue indication,
  signal quality, and artifact awareness.
- Present localized alerts and an understandable combined risk classification.
- Remain demonstrable from live, synthetic, or recorded/replayed telemetry.

## MVP and stretch scope

### Hackathon MVP

- One supervisor dashboard and one functional worker, `T'Worker 1`
- Four active MPU-6050 anchors and a 3D posture approximation
- Neutral calibration/recenter experience
- FastAPI application-state aggregation and WebSocket delivery
- Clearly labelled live, synthetic, or replay source state
- Posture and acute-event demonstration
- Environment readings when available
- Small EEG panel using live or prerecorded input
- Visible connection state, alert severity, event timeline, and fallback controls

The repository currently contains the directory scaffold and design documents.
The capabilities above are implementation targets. Record physical acceptance
results when each hardware path becomes available.

### Stretch features

- Reliable MAX30102 heart rate and optional SpO2 (only if demonstrably reliable)
- More detailed EEG spectral summaries
- Historical analysis and reporting
- Worker-side feedback
- Real multi-worker fleet monitoring
- Persistent calibration profiles and events
- Additional sensors and production deployment controls

## Non-goals

- Medical diagnosis, clinical validation, or fitness-for-duty determination
- A certified occupational-safety product or certified sound-level meter
- Anatomically accurate spinal-curvature reconstruction
- Accurate absolute yaw without a magnetometer
- Advanced ML for the hackathon
- A production alert policy, compliance workflow, or long-term data platform
- Functional multi-worker switching in the MVP
- Direct browser communication with ESP32, OpenBCI, BrainFlow, or sensor registers

## Safety and claims

All risk, posture, fatigue, fall, physiology, and environmental outputs are
prototype indicators. They may be synthetic, replayed, estimated, or affected by
sensor placement and artifacts. The demo must identify the active data source and
must not use medical, diagnostic, certified, or clinically validated language.
