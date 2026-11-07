/**
 * File: EnvironmentManager.cpp
 * Purpose: AHT21 temperature/humidity and ENS160 air-quality acquisition.
 * Contributors:
 * - Suvir Duggal
 */
#include "EnvironmentManager.h"

#include <Arduino.h>
#include <cmath>

#include "Debug.h"
#include "SystemConfig.h"

namespace {
// AHT21 commands (AHT20/21 datasheet).
constexpr uint8_t kAhtSoftReset = 0xBA;
constexpr uint8_t kAhtInitialize[] = {0xBE, 0x08, 0x00};
constexpr uint8_t kAhtTrigger[] = {0xAC, 0x33, 0x00};
constexpr uint8_t kAhtBusyBit = 0x80;
constexpr uint8_t kAhtCalibratedBit = 0x08;

// ENS160 registers (ENS160 datasheet, section 16).
constexpr uint8_t kEnsPartId = 0x00;
constexpr uint8_t kEnsOpMode = 0x10;
constexpr uint8_t kEnsTempIn = 0x13;
constexpr uint8_t kEnsDataStatus = 0x20;
constexpr uint8_t kEnsDataAqi = 0x21;
constexpr uint16_t kEnsExpectedPartId = 0x0160;
constexpr uint8_t kEnsModeReset = 0xF0;
constexpr uint8_t kEnsModeIdle = 0x01;
constexpr uint8_t kEnsModeStandard = 0x02;
constexpr uint8_t kEnsNewDataBit = 0x02;

// CRC-8, polynomial 0x31, initial 0xFF, as specified for the AHT21 frame.
uint8_t aht21Crc(const uint8_t* data, size_t length) {
  uint8_t crc = 0xFF;
  for (size_t i = 0; i < length; i++) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; bit++) {
      crc = (crc & 0x80) ? static_cast<uint8_t>((crc << 1) ^ 0x31) : static_cast<uint8_t>(crc << 1);
    }
  }
  return crc;
}

// A reading older than this is reported as unavailable (NaN).
constexpr uint32_t kStaleAfterMs = 5000;

bool isFresh(uint32_t acquiredMs, uint32_t nowMs) {
  return acquiredMs != 0 && nowMs - acquiredMs <= kStaleAfterMs;
}
}  // namespace

fw::Result EnvironmentManager::begin() {
  if (mux_.setChannel(SystemConfig::kEnvironmentMuxChannel) != fw::Result::Ok) {
    return fw::Result::IoFailure;
  }
  aht21Present_ = beginAht21() == fw::Result::Ok;
  ens160Present_ = beginEns160() == fw::Result::Ok;
  debugPrintln(aht21Present_ ? "AHT21 ready." : "AHT21 not found.");
  debugPrintln(ens160Present_ ? "ENS160 ready; warming up (about 3 min)." : "ENS160 not found.");
  return (aht21Present_ || ens160Present_) ? fw::Result::Ok : fw::Result::IoFailure;
}

fw::Result EnvironmentManager::beginAht21() {
  const uint8_t reset = kAhtSoftReset;
  if (writeRegister(SystemConfig::kAht21Address, reset, nullptr, 0) != fw::Result::Ok) {
    return fw::Result::IoFailure;
  }
  delay(20);  // Startup only, before tasks run.
  uint8_t status = 0;
  if (wire_.requestFrom(SystemConfig::kAht21Address, static_cast<uint8_t>(1)) != 1) {
    return fw::Result::IoFailure;
  }
  status = static_cast<uint8_t>(wire_.read());
  if ((status & kAhtCalibratedBit) == 0) {
    wire_.beginTransmission(SystemConfig::kAht21Address);
    wire_.write(kAhtInitialize, sizeof(kAhtInitialize));
    if (wire_.endTransmission() != 0) {
      return fw::Result::IoFailure;
    }
    delay(10);
  }
  return fw::Result::Ok;
}

fw::Result EnvironmentManager::beginEns160() {
  uint8_t id[2] = {0, 0};
  if (readRegister(SystemConfig::kEns160Address, kEnsPartId, id, 2) != fw::Result::Ok) {
    return fw::Result::IoFailure;
  }
  const uint16_t partId = static_cast<uint16_t>(id[0] | (id[1] << 8));
  if (partId != kEnsExpectedPartId) {
    return fw::Result::Fail;
  }
  const uint8_t modes[] = {kEnsModeReset, kEnsModeIdle, kEnsModeStandard};
  for (uint8_t mode : modes) {
    if (writeRegister(SystemConfig::kEns160Address, kEnsOpMode, &mode, 1) != fw::Result::Ok) {
      return fw::Result::IoFailure;
    }
    delay(10);
  }
  ens160WarmingUp_ = true;
  return fw::Result::Ok;
}

fw::Result EnvironmentManager::startConversion() {
  if (!aht21Present_) {
    return ens160Present_ ? fw::Result::Ok : fw::Result::NotReady;
  }
  if (conversionPending_) {
    return fw::Result::NotReady;
  }
  if (mux_.setChannel(SystemConfig::kEnvironmentMuxChannel) != fw::Result::Ok) {
    return fw::Result::IoFailure;
  }
  wire_.beginTransmission(SystemConfig::kAht21Address);
  wire_.write(kAhtTrigger, sizeof(kAhtTrigger));
  if (wire_.endTransmission() != 0) {
    return fw::Result::IoFailure;
  }
  conversionPending_ = true;
  conversionStartedMs_ = millis();
  return fw::Result::Ok;
}

fw::Result EnvironmentManager::read(fw::EnvData& output) {
  const uint32_t nowMs = millis();
  if (mux_.setChannel(SystemConfig::kEnvironmentMuxChannel) != fw::Result::Ok) {
    fillOutput(nowMs, output);
    return fw::Result::IoFailure;
  }

  bool updated = false;
  fw::Result ahtResult = fw::Result::Ok;
  if (aht21Present_ && conversionPending_ &&
      nowMs - conversionStartedMs_ >= SystemConfig::kAht21ConversionMs) {
    ahtResult = readAht21(nowMs, updated);
  }
  fw::Result ensResult = fw::Result::Ok;
  if (ens160Present_) {
    ensResult = readEns160(nowMs, updated);
  }

  fillOutput(nowMs, output);
  if (ahtResult == fw::Result::IoFailure || ensResult == fw::Result::IoFailure) {
    return fw::Result::IoFailure;
  }
  return updated ? fw::Result::Ok : fw::Result::NoNewSample;
}

void EnvironmentManager::fillOutput(uint32_t nowMs, fw::EnvData& output) const {
  const bool ahtFresh = isFresh(aht21AcquiredMs_, nowMs);
  const bool ensFresh = isFresh(ens160AcquiredMs_, nowMs) && !ens160WarmingUp_;
  output.temperature = ahtFresh ? temperatureC_ : NAN;
  output.humidity = ahtFresh ? humidityPercent_ : NAN;
  output.air_quality_index = ensFresh ? aqi_ : NAN;
  output.eco2_ppm = ensFresh ? eco2Ppm_ : NAN;
  output.tvoc_ppb = ensFresh ? tvocPpb_ : NAN;
}

fw::Result EnvironmentManager::readAht21(uint32_t nowMs, bool& updated) {
  uint8_t frame[7] = {0};
  if (wire_.requestFrom(SystemConfig::kAht21Address, static_cast<uint8_t>(7)) != 7) {
    conversionPending_ = false;
    return fw::Result::IoFailure;
  }
  for (uint8_t& byte : frame) {
    byte = static_cast<uint8_t>(wire_.read());
  }
  if (frame[0] & kAhtBusyBit) {
    return fw::Result::NoNewSample;  // Still converting; try again next poll.
  }
  conversionPending_ = false;
  if (aht21Crc(frame, 6) != frame[6]) {
    return fw::Result::Fail;  // Keep the previous reading; it expires if this repeats.
  }

  const uint32_t rawHumidity = (static_cast<uint32_t>(frame[1]) << 12) |
                               (static_cast<uint32_t>(frame[2]) << 4) | (frame[3] >> 4);
  const uint32_t rawTemperature = (static_cast<uint32_t>(frame[3] & 0x0F) << 16) |
                                  (static_cast<uint32_t>(frame[4]) << 8) | frame[5];
  humidityPercent_ = rawHumidity * 100.0f / 1048576.0f;
  temperatureC_ = rawTemperature * 200.0f / 1048576.0f - 50.0f;
  aht21AcquiredMs_ = nowMs == 0 ? 1 : nowMs;
  updated = true;

  // ENS160 estimates improve with ambient compensation.
  if (ens160Present_) {
    writeEns160Compensation(temperatureC_, humidityPercent_);
  }
  return fw::Result::Ok;
}

fw::Result EnvironmentManager::readEns160(uint32_t nowMs, bool& updated) {
  uint8_t status = 0;
  if (readRegister(SystemConfig::kEns160Address, kEnsDataStatus, &status, 1) != fw::Result::Ok) {
    return fw::Result::IoFailure;
  }
  if ((status & kEnsNewDataBit) == 0) {
    return fw::Result::NoNewSample;
  }
  // Validity flag, bits 3:2: 0 normal, 1 warm-up, 2 initial start-up, 3 invalid.
  const uint8_t validityFlag = (status >> 2) & 0x03;
  ens160WarmingUp_ = validityFlag != 0;

  uint8_t data[5] = {0};  // AQI, TVOC (LE16), eCO2 (LE16).
  if (readRegister(SystemConfig::kEns160Address, kEnsDataAqi, data, 5) != fw::Result::Ok) {
    return fw::Result::IoFailure;
  }
  if (validityFlag != 0) {
    return fw::Result::NotReady;
  }
  aqi_ = static_cast<float>(data[0] & 0x07);
  tvocPpb_ = static_cast<float>(data[1] | (data[2] << 8));
  eco2Ppm_ = static_cast<float>(data[3] | (data[4] << 8));
  ens160AcquiredMs_ = nowMs == 0 ? 1 : nowMs;
  updated = true;
  return fw::Result::Ok;
}

fw::Result EnvironmentManager::writeEns160Compensation(float temperatureC, float humidityPercent) {
  // TEMP_IN is Kelvin * 64 and RH_IN is percent * 512, both little-endian.
  const uint16_t kelvin64 = static_cast<uint16_t>((temperatureC + 273.15f) * 64.0f);
  const uint16_t humidity512 = static_cast<uint16_t>(humidityPercent * 512.0f);
  const uint8_t data[4] = {static_cast<uint8_t>(kelvin64 & 0xFF), static_cast<uint8_t>(kelvin64 >> 8),
                           static_cast<uint8_t>(humidity512 & 0xFF),
                           static_cast<uint8_t>(humidity512 >> 8)};
  return writeRegister(SystemConfig::kEns160Address, kEnsTempIn, data, sizeof(data));
}

fw::Result EnvironmentManager::writeRegister(uint8_t address, uint8_t reg, const uint8_t* data,
                                             size_t length) {
  wire_.beginTransmission(address);
  wire_.write(reg);
  if (length > 0) {
    wire_.write(data, length);
  }
  return wire_.endTransmission() == 0 ? fw::Result::Ok : fw::Result::IoFailure;
}

fw::Result EnvironmentManager::readRegister(uint8_t address, uint8_t reg, uint8_t* data,
                                            size_t length) {
  wire_.beginTransmission(address);
  wire_.write(reg);
  if (wire_.endTransmission(false) != 0) {
    return fw::Result::IoFailure;
  }
  if (wire_.requestFrom(address, static_cast<uint8_t>(length)) != length) {
    return fw::Result::IoFailure;
  }
  for (size_t i = 0; i < length; i++) {
    data[i] = static_cast<uint8_t>(wire_.read());
  }
  return fw::Result::Ok;
}
