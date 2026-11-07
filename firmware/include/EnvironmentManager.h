/**
 * File: EnvironmentManager.h
 * Purpose: Acquisition-owned AHT21 and ENS160 adapter boundary.
 * Contributors:
 * - Suvir Duggal
 */
#pragma once
#include <Wire.h>

#include "FirmwareTypes.h"

// Non-blocking cycle: startConversion() triggers the AHT21, and read() collects it
// at least kAht21ConversionMs later. Both select the environment mux channel first.
class EnvironmentManager {
 public:
  explicit EnvironmentManager(TwoWire& wire) : wire_(wire) {}
  fw::Result begin();  // Ok when at least one of the two sensors answered.
  fw::Result startConversion();  // NotReady when called before the previous read.
  // Copies the latest readings, each with its own stamp. NoNewSample when nothing
  // changed since the last call. ENS160 fields stay Unavailable during warm-up.
  fw::Result read(fw::Environment& output);
  bool ens160WarmingUp() const { return ens160WarmingUp_; }

 private:
  fw::Result beginAht21();
  fw::Result beginEns160();
  fw::Result readAht21(fw::TimeUs now, bool& updated);
  fw::Result readEns160(fw::TimeUs now, bool& updated);
  fw::Result writeEns160Compensation(float temperatureC, float humidityPercent);
  fw::Result writeRegister(uint8_t address, uint8_t reg, const uint8_t* data, size_t length);
  fw::Result readRegister(uint8_t address, uint8_t reg, uint8_t* data, size_t length);

  TwoWire& wire_;
  fw::Environment latest_{};
  bool aht21Present_ = false;
  bool ens160Present_ = false;
  bool ens160WarmingUp_ = true;
  bool conversionPending_ = false;
  fw::TimeUs conversionStartedUs_ = 0;
};
