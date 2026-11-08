/**
 * File: HeartRateMonitor.h
 * Purpose: Acquisition-owned MAX30102 driver producing bpm and an SpO2 estimate.
 * Contributors:
 * - Suvir Duggal
 */
#pragma once
#include <Wire.h>

#include "BeatDetector.h"
#include "FirmwareTypes.h"
#include "Spo2Estimator.h"

// Call from one task, holding the shared I2C mutex after setup.
class HeartRateMonitor {
 public:
  explicit HeartRateMonitor(TwoWire& wire) : wire_(wire) {}
  fw::Result begin();  // Checks the part ID and starts 100 Hz red+IR sampling.
  // Drains the sensor FIFO. Call at least every 250 ms or samples are lost.
  // Unavailable values are NaN (FirmwareTypes has no validity flags), so the
  // encoder can omit them. Ok when bpm is valid; NotReady while settling or off-skin.
  fw::Result read(fw::HeartData& output);

 private:
  fw::Result selectChannel();
  void resetSignal();
  fw::Result writeRegister(uint8_t reg, uint8_t value);
  fw::Result readRegisters(uint8_t reg, uint8_t* data, size_t length);

  TwoWire& wire_;
  BeatDetector detector_{100.0f};
  Spo2Estimator spo2_{};
  bool present_ = false;
  bool skinContact_ = false;
};
