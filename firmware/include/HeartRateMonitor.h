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

class HeartRateMonitor {
 public:
  explicit HeartRateMonitor(TwoWire& wire) : wire_(wire) {}
  fw::Result begin();  // Checks the part ID and starts 100 Hz red+IR sampling.
  // Drains the sensor FIFO. Call at least every 250 ms or samples are lost.
  // Unavailable values are NaN (FirmwareTypes has no validity flags), so the
  // encoder can omit them. Ok when bpm is valid; NotReady while settling or off-skin.
  fw::Result poll(fw::HeartData& output);
  bool skinContact() const { return skinContact_; }
  uint32_t lastIr() const { return lastIr_; }
  uint32_t overflows() const { return overflows_; }

 private:
  void resetSignal();
  fw::Result writeRegister(uint8_t reg, uint8_t value);
  fw::Result readRegisters(uint8_t reg, uint8_t* data, size_t length);

  TwoWire& wire_;
  BeatDetector detector_{100.0f};
  Spo2Estimator spo2_{};
  bool present_ = false;
  bool skinContact_ = false;
  uint32_t lastIr_ = 0;
  uint32_t overflows_ = 0;
};
