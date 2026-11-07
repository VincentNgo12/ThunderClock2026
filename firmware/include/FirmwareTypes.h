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
struct HeartData {
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


}  // namespace fw
