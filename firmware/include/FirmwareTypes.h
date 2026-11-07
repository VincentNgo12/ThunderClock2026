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

// proccessed IMU data
struct StripDataProcessed {
  uint32_t time;
  Quaternion data[4];
};

// MAX30102
struct HeartRate {
  float bpm; 
  float oxi;
};

// ENS160 + AHT21
struct Environment {
  // Reading temperatureC{};
  // Reading humidityPercent{};
  // Reading eco2Ppm{};  // ENS160 estimate.
  // Reading tvocPpb{};
  // Reading ens160Aqi{};
};

// INMP441
struct AudioSummary {
  float relativeDb = 0;  // Relative trend, requiring calibration for physical SPL.
};

// --------------------------------
// SNAPSHOT
// --------------------------------

struct Snapshot {
  StripDataProcessed stripData;
  // HeartData heartData;
  // EnvData envData;
  // AudioData audioData;
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
