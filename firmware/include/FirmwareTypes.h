/**
 * File: FirmwareTypes.h
 * Purpose: Shared measurements for drivers, processing and telemetry.
 * Contributors:
 * - Unassigned
 */
#pragma once
#include <cstddef>
#include <cstdint>

namespace fw {
using TimeUs = uint64_t;  // Microseconds since boot, independent of laptop UTC.
constexpr size_t kImuCount = 4;
enum class Region { Pelvis, Lumbar, Thoracic, UpperThoracic };

// Only Ok permits using an operation's output.
enum class Result {
  Ok,
  Fail,
  NotImplemented,
  NoNewSample,
  NotReady,
  InvalidArgument,
  Timeout,
  IoFailure
};
enum class Validity { Unavailable, Valid, Stale, Fault };

struct Stamp {
  TimeUs acquiredUs = 0;
  Validity validity = Validity::Unavailable;
};
struct Vec3 {
  float x = 0;
  float y = 0;
  float z = 0;
};
struct Quaternion {
  float w = 0;
  float x = 0;
  float y = 0;
  float z = 0;
};  // A zero-filled, unavailable value must stay unavailable.

struct ImuData {
  Stamp stamp{};
  Vec3 accel{};  // m/s² in sensor-board axes.
  Vec3 gyro{};  // rad/s in sensor-board axes.
};
struct ImuStripData {
  uint32_t time = 0;  // End-of-strip collection time in milliseconds since boot.
  ImuData data[kImuCount]{};
};
struct Orientation {
  Stamp stamp{};
  Quaternion rotation{};  // w,x,y,z, segment-local to world, +X right/+Y up/+Z back.
};
struct TelemetrySnapshot {
  ImuStripData raw{};
  Orientation orientations[kImuCount]{};
};

// Slow sensor fields keep their own acquisition times.
struct Reading {
  Stamp stamp{};
  float value = 0;
};
struct Environment {
  Reading temperatureC{};
  Reading humidityPercent{};
  Reading eco2Ppm{};  // ENS160 estimate.
  Reading tvocPpb{};
  Reading ens160Aqi{};
};
struct AudioSummary {
  Stamp stamp{};
  float relativeDb = 0;  // Relative trend, requiring calibration for physical SPL.
};
struct HeartRate {
  Stamp stamp{};
  float bpm = 0;  // MAX30102 stretch feature.
};
struct WearableSnapshot {
  ImuStripData stripData{};
  Environment environment{};
  AudioSummary audio{};
  HeartRate heartRate{};
};  // Internal shared state. Wire packets have their own schema.

// Future control work tracks receipt, acceptance and completion separately.
enum class CommandKind { GyroCalibration, ResetProcessing, Diagnostics };
enum class CommandStage { Received, Accepted, Rejected, Completed, Failed };
struct DeviceCommand {
  uint32_t id = 0;
  CommandKind kind = CommandKind::Diagnostics;
  uint8_t imuMask = 0x0f;  // Bits 0..3 select pelvis through upper thoracic.
};
struct CommandResult {
  uint32_t id = 0;
  CommandStage stage = CommandStage::Received;
  Result result = Result::NotImplemented;
};
}  // namespace fw
