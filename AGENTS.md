# ThunderClock

ThunderClock is a NatHacks 2026 multimodal worker-safety prototype.

## Repository structure

- `firmware/` — Adafruit ESP32-S3 Feather firmware using PlatformIO,
  Arduino framework, and FreeRTOS where appropriate.
- `backend/` — Python/FastAPI hardware gateway, signal processing,
  telemetry aggregation, EEG integration, and risk logic.
- `frontend/` — Vite + React + TypeScript supervisor dashboard.
- `simulator/` — synthetic telemetry and recorded-data replay tools.
- `docs/` — hardware scope, architecture and interface specifications.
- `protocol/` — shared wearable protocol specifications and future schemas.

## Architectural rules

- The browser never communicates directly with the ESP32 or OpenBCI.
- Hardware data enters through the Python backend.
- Backend-to-frontend realtime communication uses WebSockets.
- The frontend must work entirely from simulated telemetry.
- ESP32 firmware owns IMU calibration and sensor fusion and will produce
  quaternions from four IMUs (two ICM-20948, two MPU-6050) using 6-axis fusion.
- Heavy posture interpretation and visualization remain on the laptop.
- OpenBCI Cyton communicates independently with the laptop.
- Do not make medical or clinically validated claims.
- Optimize for hackathon demo reliability over production complexity.

## Development principles

This is a student project, and an enjoyable learning experience matters. Start with
an approachable Arduino/PlatformIO MVP: familiar setup()/loop(), plain shared structs,
a single pin/debug configuration and optional empty FreeRTOS task shells. Learn
professional embedded practices gradually, starting with basic ownership and a mutex
for shared data. Add complexity only for an actual need the team understands; do not
build a framework or application pipeline during scaffold work.

- Keep modules small and independently testable.
- Prefer descriptive names, explicit units, braces and straightforward control flow over compact expressions.
- Use brief contextual comments so a newcomer understands a file's role and important
  method inputs/results. Keep TODOs clear; avoid verbose headers or line-by-line narration.
- Do not put the entire firmware into `main.cpp`.
- Preserve clear hardware/backend/frontend boundaries.
- Do not add dependencies without a concrete need.
- Do not implement advanced ML unless explicitly requested.
- Do not substantially redesign the UI unless explicitly requested.

## Hardware scope

Use the hardware inventory in `docs/PRODUCT.md`. Four IMUs are active: ICM-20948
sensors at the pelvis and upper thoracic, and MPU-6050 sensors at the lumbar and
thoracic. Remaining MPU-6050 boards are spares. Only accel and gyro are used. Do not
read or record ICM-20948 magnetometer data. AHT21, ENS160 and INMP441 provide
environmental inputs.
MAX30102 is a supporting stretch feature. OpenBCI Cyton uses its own laptop dongle.
Keep sensor requirements aligned across firmware, backend and dashboard documents.

## Implementation status

The standalone BLE diagnostic CLI and optional FastAPI adapter are available.
Use `backend/DIAGNOSTICS.md` for commands and physical acceptance. Firmware
contains module interfaces, plain shared structs and optional task shells.
Hardware initialization and task startup are disabled by default.
Keep sensor fusion in firmware. Preserve raw acceleration including gravity.

Consult the following guides.

- `docs/ARCHITECTURE.md` for service boundaries.
- `docs/TELEMETRY_PROTOCOL.md` when changing telemetry.
- `docs/PRODUCT.md` for the hardware inventory and MVP scope.
- `firmware/ORIENTATION.md` for calibration and physical acceptance targets.
- `protocol/ORIENTATION.md` for the device packet and coordinate convention.
