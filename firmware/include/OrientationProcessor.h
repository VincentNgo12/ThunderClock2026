/**
 * File: OrientationProcessor.h
 * Purpose: One IMU's calibration and orientation-processing boundary.
 * Contributors:
 * - Unassigned
 */
#pragma once
#include "FirmwareTypes.h"

class OrientationProcessor {
 public:
  fw::Result reset();  // Future reset of this sensor's calibration and filter state.
  // Raw units are m/s² and rad/s with a boot-relative acquisition timestamp.
  fw::Result update(const fw::ImuData& raw, fw::Orientation& output);
  unsigned calibrationSamples() const;
  bool calibrated() const;
};
