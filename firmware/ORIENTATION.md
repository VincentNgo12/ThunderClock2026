# ThunderClock four-IMU orientation plan

Firmware owns calibration and fusion. The laptop validates packets and later
interprets relative posture. This guide records implementation targets for the
Adafruit ESP32-S3 Feather and four MPU-6050 sensors.

## Acquisition targets

- PCA9548/TCA9548A mux address 0x70, channels 0–3.
- MPU-6050 address 0x68, ±2 g acceleration and ±250 deg/s gyro.
- Nominal 100 Hz sampling, 10 ms polling and 10 Hz low-pass.
- Approximately 20 Hz latest-snapshot publication.
- Board-default I2C pins, proposed 400 kHz bus and 10 ms transaction timeout.
  Confirm wiring and pull-ups before using that bus speed.

Keep these defaults in one configuration file. Confirm actual board wiring and
register settings during bring-up.

After setup, acquisition owns I2C and mux selection. Drain checked accel/gyro FIFO
packets and pass every acquired sample through the corresponding independent
filter before replacing a telemetry snapshot. An eight-sample batch limit is a
proposed bound. Count overflow, excess backlog, transfer failure and clock gaps.

A failed sensor must leave the other sensor streams available. Preserve raw
sensor-board readings in m/s² and rad/s. A raw format without validity fields
must pause publication if any required reading is unavailable.

## Calibration and fusion

Consider x-io Fusion when implementing the six-axis filter. Pin its version and
preserve its license when adding the dependency. Start with an independent
orientation and gyro-bias state per sensor.

Proposed startup calibration averages two continuous stationary seconds of raw
gyro offset, with gyro magnitude below 3 deg/s and stable acceleration near 1 g.
Motion restarts averaging. Allow approximately three additional seconds for
filter startup. Hold all sensors stationary with a common heading for at least
six seconds after boot and verify readiness explicitly.

Keep orientation unavailable until calibration and startup finish. A simple
serial `c` command may reset all calibration/filter states. Recalibrate after
reboot or mounting changes. Document the command when it is implemented.

Nominal sensitivity is 16384 counts/g and 131 counts/(deg/s) for the selected
ranges. Verify all stationary ±axis readings and excessive offset/scale error.
Stationary slow rotation may resemble gyro bias, so hold the jig still.

## Mounting and coordinates

Logical anchors are pelvis, lumbar, thoracic and upper thoracic. Confirm physical
placement and per-board remaps before interpreting posture.

Remap both acceleration and gyro into segment axes with a proper rotation.
Axis-aligned remaps need matching board orientation. Arbitrary mounting angles
require a separately agreed calibration.

Follow the application coordinate convention in
[the orientation protocol](../protocol/ORIENTATION.md). Six-axis fusion requires a
common initial heading and can accumulate yaw drift. Record tilt and heading
drift separately.

## Timing and validity

FIFO sample times are estimates. Use nominal 10 ms spacing for an initial batch,
then estimate intervals from monotonic drain time and sample count. Anchor the
latest sample to receipt. Polling jitter and sequential I2C reads limit precision.

Proposed reset conditions include more than eight queued samples, lost packet
alignment or a per-sample gap above 30 ms. Reset the affected calibration/filter
and expose a loss diagnostic. Nonfinite data, free fall, near-range clipping and
failed reads also need explicit invalidation.

Use a 50 ms local freshness target before encoding. Require finite unit
quaternions within 0.001 tolerance before correcting tiny floating-point error.
Preserve packet sequence, collection uptime and per-sensor validity.

## BLE publication

Use separate raw 8888 and quaternion 8889 READ + NOTIFY characteristics on service
AAAA. Check subscription and negotiated MTU independently. A submission counter
tracks local submissions. It cannot establish laptop receipt.

Target complete-packet bounds are 150 raw bytes and 153 quaternion bytes.
Suppress oversized packets and expose a diagnostic counter. Start with one
central and a bounded latest-value queue.

## Physical acceptance checklist

1. Confirm board, mux, addresses, four channels, I2C wiring and sensor mounts.
2. Inspect signed raw readings. At rest, acceleration magnitude should be near
   9.81 m/s². Record gyro offsets and sample/loss counters.
3. Calibrate stationary and confirm each sensor transitions from unavailable to
   finite near-unit orientation.
4. Measure stationary tilt and heading drift over 60 seconds.
5. Use a measured jig for +90° about each segment axis. Check sign, axis, angle
   and return-to-start error, allowing quaternion sign equivalence. Agree a demo
   tolerance before calling the behavior accepted.
6. Mount all four boards on one rigid object with confirmed remaps. Calibrate
   together, rotate and return. Compare relative orientations.
7. Induce a missing sensor, pause or FIFO overflow. Verify affected validity,
   counted loss, continued operation of other sensors and fresh calibration
   after repair/restart.
8. Disconnect BLE while fusion runs, reconnect and confirm current snapshots
   without a backlog.

Host checks should exercise rotations, mounting, independent filter state,
invalid inputs and encoding. Record host results and physical observations
separately. Build and flashing instructions belong with the implementation.
