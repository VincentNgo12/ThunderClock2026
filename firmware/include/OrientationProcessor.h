/**
 * File: OrientationProcessor.h
 * Purpose: Sensor fusion for the IMU strip. Turns each StripData (raw accel/gyro)
 * into a StripDataProcessed (one quaternion per sensor) with x-io Fusion, 6-axis.
 * Contributors:
 * - Unassigned
 */
#pragma once
#include <Fusion.h>

#include "FirmwareTypes.h"

class OrientationProcessor {
 public:
  static constexpr size_t kSlotCount = 4;

  void begin();  // Initialise all four filters. Call once before update().
  // Feed every strip in order. Ok when at least one output quaternion is valid.
  // output.time = input.time; output slots match input slots.
  fw::Result update(const fw::StripData& input, fw::StripDataProcessed& output);

 private:
  void restartSlot(size_t slot);

  FusionAhrs ahrs[kSlotCount]; //gyro orientation filters (turn it into quaternion)
  FusionBias bias[kSlotCount]; // gyro offset estimation
  uint32_t lastSampleTimeMs[kSlotCount] = {};
  bool hasPreviousSample[kSlotCount] = {};
};
