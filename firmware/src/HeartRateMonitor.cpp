/**
 * File: HeartRateMonitor.cpp
 * Purpose: MAX30102 configuration, FIFO draining, skin contact and bpm validity.
 * Contributors:
 * - Suvir Duggal
 */
#include "HeartRateMonitor.h"

#include <Arduino.h>
#include <esp_timer.h>

#include "Debug.h"
#include "I2cMux.h"
#include "SystemConfig.h"

namespace {
// MAX30102 registers (datasheet, register map).
constexpr uint8_t kFifoWritePointer = 0x04;
constexpr uint8_t kOverflowCounter = 0x05;
constexpr uint8_t kFifoReadPointer = 0x06;
constexpr uint8_t kFifoData = 0x07;
constexpr uint8_t kFifoConfig = 0x08;
constexpr uint8_t kModeConfig = 0x09;
constexpr uint8_t kSpo2Config = 0x0A;
constexpr uint8_t kLedRedAmplitude = 0x0C;
constexpr uint8_t kLedIrAmplitude = 0x0D;
constexpr uint8_t kPartId = 0xFF;
constexpr uint8_t kExpectedPartId = 0x15;

constexpr uint8_t kModeReset = 0x40;
constexpr uint8_t kModeSpo2 = 0x03;  // Red + IR; heart rate uses IR.
// 4-sample averaging, FIFO rollover, almost-full at 17 unread samples.
constexpr uint8_t kFifoSetting = 0x40 | 0x10 | 0x0F;
// 4096 nA ADC range, 400 samples/s (100/s after averaging), 411 us pulses (18-bit).
constexpr uint8_t kSpo2Setting = 0x20 | 0x0C | 0x03;

constexpr size_t kBytesPerSample = 6;  // 3 bytes red, then 3 bytes IR.
constexpr size_t kFifoDepth = 32;
constexpr size_t kSamplesPerRead = 20;  // 120 bytes stays inside the Wire buffer.
}  // namespace

fw::Result HeartRateMonitor::begin() {
  if (i2cmux::select(wire_, SystemConfig::kHeartRateMuxChannel) != fw::Result::Ok) {
    return fw::Result::IoFailure;
  }
  uint8_t partId = 0;
  if (readRegisters(kPartId, &partId, 1) != fw::Result::Ok) {
    debugPrintln("MAX30102 not found.");
    return fw::Result::IoFailure;
  }
  if (partId != kExpectedPartId) {
    debugPrintln("MAX30102 part ID mismatch.");
    return fw::Result::Fail;
  }

  writeRegister(kModeConfig, kModeReset);
  delay(10);  // Startup only, before tasks run.
  const uint8_t settings[][2] = {
      {kFifoConfig, kFifoSetting},
      {kSpo2Config, kSpo2Setting},
      {kLedRedAmplitude, SystemConfig::kMax30102LedCurrent},
      {kLedIrAmplitude, SystemConfig::kMax30102LedCurrent},
      {kFifoWritePointer, 0},
      {kOverflowCounter, 0},
      {kFifoReadPointer, 0},
      {kModeConfig, kModeSpo2},
  };
  for (const auto& setting : settings) {
    if (writeRegister(setting[0], setting[1]) != fw::Result::Ok) {
      return fw::Result::IoFailure;
    }
  }
  present_ = true;
  debugPrintln("MAX30102 ready.");
  return fw::Result::Ok;
}

fw::Result HeartRateMonitor::poll(fw::HeartRate& output) {
  output = latest_;
  if (!present_) {
    return fw::Result::NotReady;
  }
  if (i2cmux::select(wire_, SystemConfig::kHeartRateMuxChannel) != fw::Result::Ok) {
    return fw::Result::IoFailure;
  }

  uint8_t pointers[3] = {0};  // Write pointer, overflow counter, read pointer.
  if (readRegisters(kFifoWritePointer, pointers, 3) != fw::Result::Ok) {
    return fw::Result::IoFailure;
  }
  size_t pending = (pointers[0] - pointers[2]) & (kFifoDepth - 1);
  if (pointers[1] != 0) {
    overflows_ += pointers[1];
    pending = kFifoDepth;  // Full; samples were lost, so restart detection cleanly.
    detector_.reset();
  }

  uint8_t buffer[kSamplesPerRead * kBytesPerSample];
  while (pending > 0) {
    const size_t batch = pending < kSamplesPerRead ? pending : kSamplesPerRead;
    if (readRegisters(kFifoData, buffer, batch * kBytesPerSample) != fw::Result::Ok) {
      return fw::Result::IoFailure;
    }
    for (size_t i = 0; i < batch; i++) {
      const uint8_t* ir = &buffer[i * kBytesPerSample + 3];
      const uint32_t value = ((static_cast<uint32_t>(ir[0]) << 16) |
                              (static_cast<uint32_t>(ir[1]) << 8) | ir[2]) & 0x3FFFF;
      lastIr_ = value;
      if (SystemConfig::kPrintHeartRateRaw) {
        Serial.println(value);
      }
      const bool contact = value >= SystemConfig::kHeartRateContactThreshold;
      if (contact != skinContact_) {
        detector_.reset();  // Touching or lifting off makes a huge step; start over.
        skinContact_ = contact;
      }
      if (contact) {
        detector_.addSample(static_cast<float>(value));
      }
    }
    pending -= batch;
  }

  const float bpm = detector_.bpm();
  const bool fresh = detector_.msSinceLastBeat() <= SystemConfig::kHeartRateBeatTimeoutMs;
  if (skinContact_ && bpm > 0 && fresh) {
    latest_.bpm = bpm;
    latest_.stamp.acquiredUs = static_cast<fw::TimeUs>(esp_timer_get_time());
    latest_.stamp.validity = fw::Validity::Valid;
  } else {
    latest_.bpm = 0;
    latest_.stamp.validity = fw::Validity::Unavailable;
  }
  output = latest_;
  return latest_.stamp.validity == fw::Validity::Valid ? fw::Result::Ok : fw::Result::NotReady;
}

fw::Result HeartRateMonitor::writeRegister(uint8_t reg, uint8_t value) {
  wire_.beginTransmission(SystemConfig::kMax30102Address);
  wire_.write(reg);
  wire_.write(value);
  return wire_.endTransmission() == 0 ? fw::Result::Ok : fw::Result::IoFailure;
}

fw::Result HeartRateMonitor::readRegisters(uint8_t reg, uint8_t* data, size_t length) {
  wire_.beginTransmission(SystemConfig::kMax30102Address);
  wire_.write(reg);
  if (wire_.endTransmission(false) != 0) {
    return fw::Result::IoFailure;
  }
  if (wire_.requestFrom(SystemConfig::kMax30102Address, static_cast<uint8_t>(length)) != length) {
    return fw::Result::IoFailure;
  }
  for (size_t i = 0; i < length; i++) {
    data[i] = static_cast<uint8_t>(wire_.read());
  }
  return fw::Result::Ok;
}
