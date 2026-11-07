# ThunderClock firmware

Follow the root AGENTS.md. Build an approachable Arduino/PlatformIO MVP one module
at a time. Start with setup()/loop(), small modules and plain shared structs.

Build the scaffold from the repository root with `pio run -d firmware`.
The pinned esptool 5.0.0 requires Click below 8.2 in the PlatformIO Python
environment. A `get_metavar` error during image generation indicates a version
mismatch to check in that environment.
`ImuManager` has bring-up MPU-6050 reads through the mux. ICM-20948 reads,
fusion, audio, encoding and BLE operations report NotImplemented or are absent.
`SystemConfig.h` disables hardware initialization and all task shells by default.
`setup()` creates the shared snapshot mutex before optional task startup.
The raw schema in `proto/` awaits code-generation integration.

## Planned module ownership

- `SystemConfig.h` owns pinouts, task toggles, timing targets and the debug switch.
  Mark unconfirmed mux channels, physical placements and mounting explicitly.
- `main.cpp` initializes Wire and starts enabled tasks. Keep optional task bodies
  in `FirmwareTasks.cpp`.
- Acquisition owns the I2C bus and mux after setup. Keep sensor reads and checked
  FIFO transfers in `ImuManager`. It drives two sensor types: ICM-20948 at pelvis
  and upper thoracic, and MPU-6050 at lumbar and thoracic. Only accel and gyro are
  read. It assembles four-sensor strips at least every 10 ms. Each sample carries
  its own timestamp and validity.
- `OrientationProcessor` owns per-sensor gyro calibration and 6-axis fusion. It
  must consume each acquired sample before telemetry can replace a display
  snapshot.
- Environment/audio owns AHT21, ENS160 and INMP441 acquisition. MAX30102 is a
  supporting stretch feature.
- Telemetry owns packet encoding and BLE delivery. Coordinate shared structs,
  task wiring and protocol changes with the teammates using them.

## Development rules

- Start shared-resource safety with a snapshot mutex. Keep locks short and bounded
  to copies or field updates. Perform I/O, processing and logging outside locks.
- Preserve units, acquisition timing and validity. Check retained reading ages.
  Represent missing measurements explicitly.
- Enable tasks when useful work exists. Stubs must report NotImplemented.
  Add test files when needed for implemented behavior.
- Every owned .h/.hpp/.c/.cpp begins with File, a short contextual Purpose and
  Contributors using verified human names or Unassigned. Omit dates and preserve
  licenses. Exclude third-party, generated and build files.
- Use brief method comments for unusual inputs, results and ownership rules.
  Prefer descriptive names, braces and straightforward control flow.
- Add dependencies and FreeRTOS machinery only when the team has a concrete need.
- Backend owns worker neutral references and risk interpretation. OpenBCI stays
  on the laptop.

Read [orientation guidance](ORIENTATION.md) for acquisition, calibration and
physical acceptance targets, and [the device boundary](../docs/FIRMWARE_PROTOCOL.md)
before implementing BLE transport. These documents describe planned behavior.
