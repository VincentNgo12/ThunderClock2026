#pragma once
#include "FirmwareTypes.h"

namespace SystemConfig {
constexpr unsigned long kSerialBaud = 115200;
constexpr bool kDebugEnabled = true;

// Adafruit Feather ESP32-S3 onboard NeoPixel (separate from the red GPIO 13 LED).
constexpr int kStatusLedPin = 33;
constexpr int kStatusLedPowerPin = 21;
constexpr uint8_t kStatusLedBrightness = 16;
constexpr uint32_t kBleBlinkPeriodMs = 500;

// sensor and telemetry ini 
constexpr bool kInitImuStrip = true;
constexpr bool kInitEnvironment = false;  // AHT21 + ENS160 on kEnvironmentMuxChannel.
constexpr bool kInitAudio = false;
constexpr bool kInitHeartRate = false;  // MAX30102 on kHeartRateMuxChannel.
constexpr bool kInitBLE = true;

// task ini
constexpr bool kStartAcquisitionTask = true;
constexpr bool kStartIMUTask = true;

constexpr bool kPrintHeartRateRaw = false;  // Prints "IR red" samples for the Serial Plotter.

constexpr uint32_t kAcquisitionPeriodMs = 10;       // Target 100 Hz.
constexpr uint32_t kAudioIdlePeriodMs = 50;         // Shell wait, independent of I2S sample rate.
constexpr uint32_t kSlowSensorPollMs = 50;          // MAX30102 FIFO holds 320 ms at 100 Hz.
constexpr uint32_t kEnvironmentPeriodMs = 1000;     // One AHT21/ENS160 cycle per second.
constexpr uint32_t kAht21ConversionMs = 80;         // Datasheet measurement time.
constexpr uint32_t kHeartRateBeatTimeoutMs = 3000;  // No beat for this long -> unavailable.
constexpr uint32_t kTaskStackBytes = 4096;          // ESP32 FreeRTOS uses bytes.

constexpr float kImuRateHz = 100.0f;
constexpr uint32_t kImuSamplePeriodUs = 10000;
constexpr uint32_t kMaxSampleGapUs = 100000;  // Longer gap restarts that sensor's filter.

// IMU ranges used for scaling and fusion, ImuManager's register settings must match.
constexpr float kImuAccelRangeG = 8.0f;
constexpr float kImuGyroRangeDps = 1000.0f;
constexpr uint32_t kSampleFreshUs = 50000;
constexpr unsigned kGyroCalibrationSamples = 200;
constexpr size_t kFifoBatchSize = 8;

// i2c pin config
constexpr int kSdaPin = -1; // -1 for default
constexpr int kSclPin = -1; // -1 for default
constexpr int kI2sBclkPin = 9;
// i2s pin config
constexpr int kI2sWsPin = 10;
constexpr int kI2sDataPin = 11;

// i2c peripherals adresses
constexpr uint8_t kMuxAddress = 0x70;
constexpr uint8_t kImuAddress = 0x68;

// i2c timing, sensor addresses and mux channels
constexpr uint32_t kI2cClockHz = 100000;  // Long harness wires: stay at 100 kHz.
constexpr uint16_t kI2cTimeoutMs = 10;
constexpr int kHeartRateMuxChannel = 4;  // TODO(FW-HR): confirm wiring.
constexpr int kEnvironmentMuxChannel = 5;  // TODO(FW-ENV): confirm wiring.
constexpr uint8_t kAht21Address = 0x38;
constexpr uint8_t kEns160Address = 0x53;  // 0x52 when the breakout ties ADDR low.
constexpr uint8_t kMax30102Address = 0x57;
// MAX30102 LED current, 0.2 mA per step. Raise for earlobe if the IR level is low.
constexpr uint8_t kMax30102LedCurrent = 0x24;
// Raw IR (18-bit) below this means the sensor is not touching skin. Tune on the body.
constexpr uint32_t kHeartRateContactThreshold = 50000;

}  // namespace SystemConfig
