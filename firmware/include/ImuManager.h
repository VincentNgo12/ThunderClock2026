/**
 * File: ImuManager.h
 * Purpose: Accel and gyro reads from the four mux-connected IMUs, called by acquisition.
 * Contributors:
 * - Unassigned
 */
#pragma once
#include <Adafruit_ICM20948.h>
#include <Adafruit_MPU6050.h>
#include <Arduino.h>
#include <Wire.h>

#include "FirmwareTypes.h"

class ImuManager {
 public:
  static constexpr size_t kSlotCount = 4;

  explicit ImuManager(TwoWire& wire) : wire_(wire) {}
  // Ok when at least one sensor starts. A failed sensor stays unavailable and reads as invalid.
  fw::Result begin();
  // Ok when at least one slot is valid. Each slot's valid flag reports its own read.
  fw::Result read(fw::StripData& output);
  bool available(size_t slot) const;
  // Selects a mux channel. Shared with the environment and heart-rate modules on the same bus.
  fw::Result setChannel(uint8_t channel);

 private:

  TwoWire& wire_;
  uint8_t muxAddress = 0x70;
  // Mux channel i: pelvis, lumbar, thoracic, upper thoracic.
  bool isIcm[kSlotCount] = {true, false, false, true};
  Adafruit_ICM20948 icmSensors[kSlotCount];  // Only slots where isIcm is true.
  Adafruit_MPU6050 mpuSensors[kSlotCount];   // Only slots where isIcm is false.
  bool slotAvailable[kSlotCount] = {};
};
