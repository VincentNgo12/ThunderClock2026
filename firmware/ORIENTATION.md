# ThunderClock four-IMU orientation plan

Firmware owns calibration and fusion. The laptop validates packets and later
interprets relative posture. This guide records implementation targets for the
Adafruit ESP32-S3 Feather with two ICM-20948 and two MPU-6050 sensors.

## Sensor layout

| ID | Segment | Mux channel | Sensor | Inputs used | Fusion |
| --- | --- | --- | --- | --- | --- |
| 1 | Pelvis | 0 | ICM-20948 | Accel, gyro | 6-axis |
| 2 | Lumbar | 1 | MPU-6050 | Accel, gyro | 6-axis |
| 3 | Thoracic | 2 | MPU-6050 | Accel, gyro | 6-axis |
| 4 | Upper thoracic | 3 | ICM-20948 | Accel, gyro | 6-axis |

The ICM-20948 contains a magnetometer, but this project does not read, record or
fuse magnetometer data. All four sensors use the same 6-axis fusion. Mux channels
are confirmed. `ImuManager` records the sensor type and channel for each slot.
Remaining MPU-6050 boards are spares.

## Acquisition targets

- PCA9548/TCA9548A mux address 0x70, channels 0–3, one sensor per channel.
- MPU-6050 address 0x68. ICM-20948 address 0x69 (Adafruit breakout default),
  or 0x68 with the AD0 jumper bridged.
- ±8 g acceleration and ±1000 deg/s gyro on both sensor types, so fall impacts
  and fast trunk rotation do not clip.
- About 10 Hz low-pass: 10 Hz on the MPU-6050, 11.5 Hz accel and 11.6 Hz gyro on
  the ICM-20948. Further filtering happens on the laptop.
- ICM-20948 internal rates set near 100 Hz. The driver starts the magnetometer,
  but firmware never reads it.
- 100 Hz strip reads and approximately 20 Hz snapshot publication.
- Board-default I2C pins, 400 kHz bus and 10 ms transaction timeout. Confirm
  wiring and pull-ups at that bus speed.

Drivers are Adafruit MPU6050 (vendored in `lib/`) and Adafruit ICM20X. Addresses,
channels, ranges and filters live in `ImuManager`. Confirm actual board wiring
and register settings during bring-up.

## Strip acquisition

After setup, acquisition owns I2C and mux selection. Every 10 ms the acquisition
task reads the latest accel and gyro from all four sensors in turn and stores
them together as one strip. A strip is the unit passed to fusion. The task uses
a fixed 10 ms period, so a strip is produced every 10 ms.

All four sensors are read together, so the strip carries one collection time in
milliseconds. Sequential mux reads take a few milliseconds, so the readings are
close but not simultaneous. Hardware FIFO draining is a possible later
improvement if timing needs to be tighter.

A strip is still produced on time when a sensor fails. The whole strip goes to
fusion with that sensor's slot marked invalid. Valid sensors update their
filters normally. The invalid sensor skips its filter update for that strip, and
its quaternion is marked invalid for that strip. Each sample stores raw
sensor-board accel in m/s² and gyro in rad/s and a validity flag. Pass every
sample through that sensor's filter before replacing a telemetry snapshot.

## Calibration and fusion

Use x-io Fusion for the AHRS filter, with the no-magnetometer update on every
sensor. Pin its version and preserve its license when adding the dependency.
Keep an independent orientation and gyro-bias state per sensor.

Proposed startup calibration averages two continuous stationary seconds of raw
gyro offset, with gyro magnitude below 3 deg/s and stable acceleration near 1 g.
Motion restarts averaging. Allow approximately three additional seconds for
filter startup. Hold all sensors stationary with a common heading for at least
six seconds after boot and verify readiness explicitly.

Keep orientation unavailable until calibration and startup finish. A simple
serial `c` command may reset all calibration/filter states. Recalibrate after
reboot or mounting changes. Document the command when it is implemented.

Nominal sensitivity at the selected ranges is 4096 counts/g and 32.8
counts/(deg/s) on both sensor types. Verify all stationary ±axis readings and
excessive offset/scale error. Stationary slow rotation may resemble gyro bias,
so hold the jig still.

## Mounting and coordinates

Logical anchors are pelvis, lumbar, thoracic and upper thoracic. Confirm physical
placement and per-board remaps before interpreting posture.

Remap both acceleration and gyro into segment axes with a proper rotation.
Axis-aligned remaps need matching board orientation. Arbitrary mounting angles
require a separately agreed calibration.

Follow the application coordinate convention in
[the orientation protocol](../protocol/ORIENTATION.md).

6-axis fusion cannot observe absolute heading. Heading zero is the direction the
wearer faces during startup calibration, and all four sensors start there. Each
sensor's heading then drifts independently. Tilt stays referenced to gravity.
The demo focuses on forward and side bending, which depend on tilt. Firmware
does not correct heading drift. The backend handles relative heading through its
neutral-reference capture. Record tilt and heading drift separately.

## Timing and validity

Each strip carries one collection time in milliseconds from `millis()`. Fusion
uses the time between strips as its update interval.

Proposed reset conditions include a gap above 30 ms between valid samples for a
sensor or repeated read failures. Reset the affected calibration/filter and
expose a loss diagnostic. Nonfinite data, free fall, near-range clipping and
failed reads also need explicit invalidation.

Use a 50 ms local freshness target before encoding. Require finite unit
quaternions within 0.001 tolerance before correcting tiny floating-point error.
Preserve packet sequence, collection uptime and per-sensor validity.

## BLE publication

Use separate raw 8888 and quaternion 8889 READ + NOTIFY characteristics on service
AAAA. Check subscription and negotiated MTU independently. A submission counter
tracks local submissions. It cannot establish laptop receipt.

Complete-packet bounds are 150 raw bytes, 153 orientation-only v1 bytes and
201 telemetry v2 bytes. See [the BLE contract](../protocol/ORIENTATION.md) for
version selection and the negotiated ATT MTU requirement.
Suppress oversized packets and expose a diagnostic counter. Start with one
central and a bounded latest-value queue.

## Physical acceptance checklist

1. Confirm board, mux, addresses, sensor type per channel, I2C wiring and mounts.
2. Inspect signed raw readings. At rest, acceleration magnitude should be near
   9.81 m/s². Record gyro offsets and sample/loss counters.
3. Confirm a strip arrives every 10 ms and all four slots stay valid.
4. Calibrate stationary and confirm each sensor transitions from unavailable to
   finite near-unit orientation.
5. Measure stationary tilt and heading drift over 60 seconds for each sensor.
6. Use a measured jig for +90° about each segment axis. Check sign, axis, angle
   and return-to-start error, allowing quaternion sign equivalence. Agree a demo
   tolerance before calling the behavior accepted.
7. Mount all four boards on one rigid object with confirmed remaps. Calibrate
   together, rotate and return. Compare relative orientations.
8. Induce a missing sensor or pause. Verify affected validity,
   counted loss, continued operation of other sensors and fresh calibration
   after repair/restart.
9. Disconnect BLE while fusion runs, reconnect and confirm current snapshots
   without a backlog.

Host checks should exercise rotations, mounting, independent filter state,
invalid inputs and encoding. Record host results and physical observations
separately. Build and flashing instructions belong with the implementation.
