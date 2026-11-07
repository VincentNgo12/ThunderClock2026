/**
 * File: EnvironmentManager.h
 * Purpose: Acquisition-owned AHT21 and ENS160 adapter boundary.
 * Contributors:
 * - Suvir Duggal
 */
#pragma once
#include <Wire.h>

#include "FirmwareTypes.h"
#include "ImuManager.h"

// Non-blocking cycle: startConversion() triggers the AHT21, and read() collects it
// at least kAht21ConversionMs later. Both select the environment mux channel first,
// using the IMU manager's mux helper.
class EnvironmentManager {
 public:
  EnvironmentManager(TwoWire& wire, ImuManager& mux) : wire_(wire), mux_(mux) {}
  fw::Result begin();  // Ok when at least one of the two sensors answered.
  fw::Result startConversion();  // NotReady while the previous conversion is pending.
  // Fills output with the latest readings. Missing, stale or warming-up values are
  // NaN (FirmwareTypes has no validity flags). NoNewSample when nothing changed.
  fw::Result read(fw::EnvData& output);
  bool ens160WarmingUp() const { return ens160WarmingUp_; }

 private:
  fw::Result beginAht21();
  fw::Result beginEns160();
  fw::Result readAht21(uint32_t nowMs, bool& updated);
  fw::Result readEns160(uint32_t nowMs, bool& updated);
  fw::Result writeEns160Compensation(float temperatureC, float humidityPercent);
  fw::Result writeRegister(uint8_t address, uint8_t reg, const uint8_t* data, size_t length);
  fw::Result readRegister(uint8_t address, uint8_t reg, uint8_t* data, size_t length);
  void fillOutput(uint32_t nowMs, fw::EnvData& output) const;

  TwoWire& wire_;
  ImuManager& mux_;
  bool aht21Present_ = false;
  bool ens160Present_ = false;
  bool ens160WarmingUp_ = true;
  bool conversionPending_ = false;
  uint32_t conversionStartedMs_ = 0;
  // Latest good readings and when they were acquired (millis, 0 = never).
  float temperatureC_ = 0;
  float humidityPercent_ = 0;
  uint32_t aht21AcquiredMs_ = 0;
  float aqi_ = 0;
  float eco2Ppm_ = 0;
  float tvocPpb_ = 0;
  uint32_t ens160AcquiredMs_ = 0;
};
