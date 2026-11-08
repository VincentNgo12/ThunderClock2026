/**
 * File: EnvironmentManager.h
 * Purpose: One temperature, humidity and air-quality reader for AHT21 + ENS160.
 * Contributors:
 * - Suvir Duggal
 */
#pragma once
#include <cmath>
#include <Wire.h>

#include "FirmwareTypes.h"

// Call from one task, holding the shared I2C mutex after setup. read() manages
// conversion timing internally; call it regularly (for example, every 50 ms).
class EnvironmentManager {
 public:
  explicit EnvironmentManager(TwoWire& wire) : wire_(wire) {}
  fw::Result begin();  // Ok only when both AHT21 and ENS160 initialize.
  // Never waits for a conversion. Ok when any fresh reading is available,
  // including cached readings. ENS160 quality flags do not suppress readings.
  // Missing or stale fields are NaN. NotReady means no readings yet.
  fw::Result read(fw::EnvData& output);

 private:
  fw::Result selectChannel();
  fw::Result beginAht21();
  fw::Result beginEns160();
  fw::Result startConversion();
  fw::Result readAht21(uint32_t nowMs);
  fw::Result readEns160(uint32_t nowMs);
  fw::Result writeEns160Compensation(float temperatureC, float humidityPercent);
  fw::Result writeRegister(uint8_t address, uint8_t reg, const uint8_t* data, size_t length);
  fw::Result readRegister(uint8_t address, uint8_t reg, uint8_t* data, size_t length);
  void fillOutput(uint32_t nowMs, fw::EnvData& output) const;

  TwoWire& wire_;
  bool aht21Present_ = false;
  bool ens160Present_ = false;
  bool conversionPending_ = false;
  bool cycleStarted_ = false;
  uint32_t lastCycleMs_ = 0;
  // NaN means no measurement yet. Timestamps use millis(), including wraparound.
  fw::EnvData latest_{NAN, NAN, NAN, NAN, NAN};
  uint32_t aht21AcquiredMs_ = 0;
  uint32_t ens160AcquiredMs_ = 0;
};
