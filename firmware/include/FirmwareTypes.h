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

// --------------------------------
// RESULTS
// --------------------------------

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
// enum class Validity { Unavailable, Valid, Stale, Fault };

// --------------------------------
// SENSOR DATA TYPES 
// --------------------------------

struct Vec3 {
  float x;
  float y;
  float z;
}; 

struct IMUData {
  Vec3 accel;
  Vec3 gyro;
  Vec3 magno;
};

// unprocessed IMU data
struct StripData {
  uint32_t time;
  IMUData data[4];
};

struct Quaternion {
  float w;
  float x;
  float y;
  float z;
};  

<<<<<<< HEAD
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
struct TelemetrySnapshot {
  ImuStripData raw{};
  Orientation orientations[kImuCount]{};
  Environment environment{};
  AudioSummary audio{};
};  // Encoder omits stale or unavailable readings using their acquisition stamps.

=======
// proccessed IMU data
struct StripDataProcessed {
  uint32_t time;
  Quaternion data[4];
};

// MAX30102
>>>>>>> firmwareTypesMod
struct HeartRate {
  float bpm; 
  float oxi;
};

// ENS160 + AHT21
struct EnvData {
  float temperature;
  float humidity;
  float air_quality_index;
  float eco2_ppm;
  float tvoc_ppb;
};

// INMP441
struct AudioData {
  float relativeDb;  // Relative trend, requiring calibration for physical SPL.
};

// --------------------------------
// SNAPSHOT
// --------------------------------

struct Snapshot {
  StripDataProcessed stripData;
  EnvData envData;
  HeartData heartData;
  AudioData audioData;
};

// ignore the code below for now

// Future control work tracks receipt, acceptance and completion separately.
// enum class CommandKind { GyroCalibration, ResetProcessing, Diagnostics };
// enum class CommandStage { Received, Accepted, Rejected, Completed, Failed };
// struct DeviceCommand {
//   uint32_t id = 0;
//   CommandKind kind = CommandKind::Diagnostics;
//   uint8_t imuMask = 0x0f;  // Bits 0..3 select pelvis through upper thoracic.
// };
// struct CommandResult {
//   uint32_t id = 0;
//   CommandStage stage = CommandStage::Received;
//   Result result = Result::NotImplemented;
// };

}  // namespace fw
