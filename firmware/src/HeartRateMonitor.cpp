#include "HeartRateMonitor.h"

#include <Arduino.h>

#include <cmath>

#include "Debug.h"
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
constexpr uint8_t kModeSpo2 = 0x03;  // Red + IR.
// 4-sample averaging, FIFO rollover, almost-full at 17 unread samples.
constexpr uint8_t kFifoSetting = 0x40 | 0x10 | 0x0F;
// 4096 nA ADC range, 400 samples/s (100/s after averaging), 411 us pulses (18-bit).
constexpr uint8_t kSpo2Setting = 0x20 | 0x0C | 0x03;

constexpr size_t kBytesPerSample = 6;  // 3 bytes red, then 3 bytes IR.
constexpr size_t kFifoDepth = 32;
constexpr size_t kSamplesPerRead = 20;  // 120 bytes stays inside the Wire buffer.

uint32_t sample18(const uint8_t* bytes) {
  return ((static_cast<uint32_t>(bytes[0]) << 16) | (static_cast<uint32_t>(bytes[1]) << 8) |
          bytes[2]) & 0x3FFFF;
}
}  // namespace

fw::Result HeartRateMonitor::begin() {
  if (selectChannel() != fw::Result::Ok) {
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

fw::Result HeartRateMonitor::selectChannel() {
  wire_.beginTransmission(SystemConfig::kMuxAddress);
  wire_.write(1 << SystemConfig::kHeartRateMuxChannel);
  return wire_.endTransmission() == 0 ? fw::Result::Ok : fw::Result::IoFailure;
}

void HeartRateMonitor::resetSignal() {
  detector_.reset();
  spo2_.reset();
}

fw::Result HeartRateMonitor::read(fw::HeartData& output) {
  output.bpm = NAN;
  output.oxi = NAN;
  if (!present_) {
    return fw::Result::NotReady;
  }
  if (selectChannel() != fw::Result::Ok) {
    return fw::Result::IoFailure;
  }

  uint8_t pointers[3] = {0};  // Write pointer, overflow counter, read pointer.
  if (readRegisters(kFifoWritePointer, pointers, 3) != fw::Result::Ok) {
    return fw::Result::IoFailure;
  }
  size_t pending = (pointers[0] - pointers[2]) & (kFifoDepth - 1);
  if (pointers[1] != 0) {
    pending = kFifoDepth;  // Full; samples were lost, so restart detection cleanly.
    resetSignal();
  }

  uint8_t buffer[kSamplesPerRead * kBytesPerSample];
  while (pending > 0) {
    const size_t batch = pending < kSamplesPerRead ? pending : kSamplesPerRead;
    if (readRegisters(kFifoData, buffer, batch * kBytesPerSample) != fw::Result::Ok) {
      return fw::Result::IoFailure;
    }
    for (size_t i = 0; i < batch; i++) {
      const uint32_t red = sample18(&buffer[i * kBytesPerSample]);
      const uint32_t ir = sample18(&buffer[i * kBytesPerSample + 3]);
      if (SystemConfig::kPrintHeartRateRaw) {
        Serial.printf("%lu %lu\n", static_cast<unsigned long>(ir), static_cast<unsigned long>(red));
      }
      const bool contact = ir >= SystemConfig::kHeartRateContactThreshold;
      if (contact != skinContact_) {
        resetSignal();  // Touching or lifting off makes a huge step; start over.
        skinContact_ = contact;
      }
      if (contact) {
        const bool beat = detector_.addSample(static_cast<float>(ir));
        spo2_.addSample(static_cast<float>(red), static_cast<float>(ir), beat);
      }
    }
    pending -= batch;
  }

  const bool fresh = detector_.msSinceLastBeat() <= SystemConfig::kHeartRateBeatTimeoutMs;
  if (!skinContact_ || !fresh) {
    return fw::Result::NotReady;
  }
  const float bpm = detector_.bpm();
  const float spo2 = spo2_.percent();
  if (bpm > 0) {
    output.bpm = bpm;
  }
  if (spo2 > 0) {
    output.oxi = spo2;
  }
  return bpm > 0 ? fw::Result::Ok : fw::Result::NotReady;
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
