/**
 * File: SystemConfig.h
 * Purpose: One place for pinouts, debug output and optional task startup.
 * Contributors:
 * - Unassigned
 */
#pragma once
#include "FirmwareTypes.h"

namespace SystemConfig {
constexpr unsigned long kSerialBaud = 115200;
constexpr bool kDebugEnabled = true;

// sensor and telemetry ini 
constexpr bool kInitImuStrip = false;
constexpr bool kInitBLE = false;
constexpr bool kInitAudio = true;

// task ini
constexpr bool kStartAcquisitionTask = false;
constexpr bool kStartTelemetryTask = false;
constexpr bool kStartAudioTask = false;

constexpr uint32_t kAcquisitionPeriodMs = 10;  // Target 100 Hz.
constexpr uint32_t kTelemetryPeriodMs = 50;  // Target 20 Hz.
constexpr uint32_t kAudioIdlePeriodMs = 50;  // Shell wait, independent of I2S sample rate.
constexpr uint32_t kTaskStackBytes = 4096;  // ESP32 FreeRTOS uses bytes.

constexpr float kImuRateHz = 100.0f;
constexpr uint32_t kImuSamplePeriodUs = 10000;
constexpr uint32_t kMaxSampleGapUs = 30000;
constexpr uint32_t kSampleFreshUs = 50000;
constexpr unsigned kGyroCalibrationSamples = 200;
constexpr size_t kFifoBatchSize = 8;

// -1 uses board defaults for Wire pins and leaves microphone pins unconfigured.
constexpr int kSdaPin = -1;
constexpr int kSclPin = -1;
constexpr int kI2sBclkPin = -1;
constexpr int kI2sWsPin = -1;
constexpr int kI2sDataPin = -1;

// Prototype mux defaults. Confirm wiring, placement and mounting before use.
constexpr uint8_t kMuxAddress = 0x70;
constexpr uint8_t kImuAddress = 0x68;
// constexpr int kImuMuxChannels[fw::kImuCount] = {0, 1, 2, 3};
// TODO(FW-IMU): confirm each board's sensor-to-segment mounting transform.

// Zero leaves each auxiliary sensor address unconfirmed.
constexpr uint8_t kAht21Address = 0;
constexpr uint8_t kEns160Address = 0;
constexpr uint8_t kMax30102Address = 0;  // Optional stretch feature.
}  // namespace SystemConfig
