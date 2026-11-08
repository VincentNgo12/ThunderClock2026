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
  return nowMs - acquiredMs <= kStaleAfterMs;
}
}  // namespace

fw::Result EnvironmentManager::begin() {
  aht21Present_ = false;
  ens160Present_ = false;
  conversionPending_ = false;
  cycleStarted_ = false;
  latest_ = {NAN, NAN, NAN, NAN, NAN};
  aht21AcquiredMs_ = 0;
  ens160AcquiredMs_ = 0;
  if (selectChannel() != fw::Result::Ok) {
    return fw::Result::IoFailure;
  }
  aht21Present_ = beginAht21() == fw::Result::Ok;
  ens160Present_ = beginEns160() == fw::Result::Ok;
  debugPrintln(aht21Present_ ? "AHT21 ready." : "AHT21 not found.");
  debugPrintln(ens160Present_ ? "ENS160 ready; warming up (about 3 min)." : "ENS160 not found.");
  return (aht21Present_ || ens160Present_) ? fw::Result::Ok : fw::Result::IoFailure;
}

fw::Result EnvironmentManager::selectChannel() {
  wire_.beginTransmission(SystemConfig::kMuxAddress);
  wire_.write(1 << SystemConfig::kEnvironmentMuxChannel);
  return wire_.endTransmission() == 0 ? fw::Result::Ok : fw::Result::IoFailure;
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
  return fw::Result::Ok;
}

fw::Result EnvironmentManager::startConversion() {
  wire_.beginTransmission(SystemConfig::kAht21Address);
  wire_.write(kAhtTrigger, sizeof(kAhtTrigger));
  if (wire_.endTransmission() != 0) {
    return fw::Result::IoFailure;
  }
  conversionPending_ = true;
  return fw::Result::Ok;
}

fw::Result EnvironmentManager::read(fw::EnvData& output) {
  output = {NAN, NAN, NAN, NAN, NAN};
  if (!aht21Present_ && !ens160Present_) {
    return fw::Result::NotReady;
  }

  const uint32_t nowMs = millis();
  const bool cycleDue = !cycleStarted_ || nowMs - lastCycleMs_ >= SystemConfig::kEnvironmentPeriodMs;
  const bool conversionReady = conversionPending_ &&
      nowMs - lastCycleMs_ >= SystemConfig::kAht21ConversionMs;

  fw::Result ahtResult = fw::Result::NotReady;
  fw::Result ensResult = fw::Result::NotReady;
  if (cycleDue || conversionReady) {
    if (selectChannel() != fw::Result::Ok) {
      return fw::Result::IoFailure;
    }
    // Collect the previous AHT21 conversion before starting another cycle.
    if (conversionReady) {
      ahtResult = readAht21(nowMs);
    }
    if (cycleDue && !conversionPending_) {
      cycleStarted_ = true;
      lastCycleMs_ = nowMs;
      if (aht21Present_) {
        const fw::Result triggerResult = startConversion();
        if (triggerResult != fw::Result::Ok) {
          ahtResult = triggerResult;
        }
      }
      if (ens160Present_) {
        ensResult = readEns160(nowMs);
      }
    }
  }

  fillOutput(nowMs, output);
  // Like ImuManager, one usable sensor is enough for Ok; unavailable fields
  // stay NaN. Cached readings remain usable until their original time expires.
  if (std::isfinite(output.temperature) || std::isfinite(output.air_quality_index)) {
    return fw::Result::Ok;
  }
  if (ahtResult == fw::Result::IoFailure || ensResult == fw::Result::IoFailure) {
    return fw::Result::IoFailure;
  }
  if (ahtResult == fw::Result::Fail || ahtResult == fw::Result::Timeout) {
    return ahtResult;
  }
  return fw::Result::NotReady;
}

void EnvironmentManager::fillOutput(uint32_t nowMs, fw::EnvData& output) const {
  const bool ahtFresh = isFresh(aht21AcquiredMs_, nowMs);
  const bool ensFresh = isFresh(ens160AcquiredMs_, nowMs);
  output.temperature = ahtFresh ? latest_.temperature : NAN;
  output.humidity = ahtFresh ? latest_.humidity : NAN;
  output.air_quality_index = ensFresh ? latest_.air_quality_index : NAN;
  output.eco2_ppm = ensFresh ? latest_.eco2_ppm : NAN;
  output.tvoc_ppb = ensFresh ? latest_.tvoc_ppb : NAN;
}

fw::Result EnvironmentManager::readAht21(uint32_t nowMs) {
  uint8_t frame[7] = {0};
  if (wire_.requestFrom(SystemConfig::kAht21Address, static_cast<uint8_t>(7)) != 7) {
    conversionPending_ = false;
    return fw::Result::IoFailure;
  }
  for (uint8_t& byte : frame) {
    byte = static_cast<uint8_t>(wire_.read());
  }
  if (frame[0] & kAhtBusyBit) {
    if (nowMs - lastCycleMs_ >= SystemConfig::kEnvironmentPeriodMs) {
      conversionPending_ = false;  // Retry next cycle if the sensor stayed busy.
      return fw::Result::Timeout;
    }
    return fw::Result::NotReady;
  }
  conversionPending_ = false;
  if (aht21Crc(frame, 6) != frame[6]) {
    return fw::Result::Fail;  // Keep the previous reading; it expires if this repeats.
  }

  const uint32_t rawHumidity = (static_cast<uint32_t>(frame[1]) << 12) |
                               (static_cast<uint32_t>(frame[2]) << 4) | (frame[3] >> 4);
  const uint32_t rawTemperature = (static_cast<uint32_t>(frame[3] & 0x0F) << 16) |
                                  (static_cast<uint32_t>(frame[4]) << 8) | frame[5];
  latest_.humidity = rawHumidity * 100.0f / 1048576.0f;
  latest_.temperature = rawTemperature * 200.0f / 1048576.0f - 50.0f;
  aht21AcquiredMs_ = nowMs;

  // ENS160 estimates improve with ambient compensation.
  if (ens160Present_) {
    writeEns160Compensation(latest_.temperature, latest_.humidity);
  }
  return fw::Result::Ok;
}

fw::Result EnvironmentManager::readEns160(uint32_t nowMs) {
  uint8_t status = 0;
  if (readRegister(SystemConfig::kEns160Address, kEnsDataStatus, &status, 1) != fw::Result::Ok) {
    return fw::Result::IoFailure;
  }
  // Validity flag, bits 3:2: 0 normal, 1 warm-up, 2 initial start-up, 3 invalid.
  const uint8_t validityFlag = (status >> 2) & 0x03;
  if (validityFlag != 0) {
    latest_.air_quality_index = NAN;
    latest_.eco2_ppm = NAN;
    latest_.tvoc_ppb = NAN;
    return fw::Result::NotReady;
  }
  if ((status & kEnsNewDataBit) == 0) {
    return fw::Result::NoNewSample;
  }

  uint8_t data[5] = {0};  // AQI, TVOC (LE16), eCO2 (LE16).
  if (readRegister(SystemConfig::kEns160Address, kEnsDataAqi, data, 5) != fw::Result::Ok) {
    return fw::Result::IoFailure;
  }
  latest_.air_quality_index = static_cast<float>(data[0] & 0x07);
  latest_.tvoc_ppb = static_cast<float>(data[1] | (data[2] << 8));
  latest_.eco2_ppm = static_cast<float>(data[3] | (data[4] << 8));
  ens160AcquiredMs_ = nowMs;
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
