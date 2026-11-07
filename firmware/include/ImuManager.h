/**
 * File: ImuManager.h
 * Purpose: Acquisition-owned mux and four-IMU driver boundary.
 * Contributors:
 * - Unassigned
 */
#pragma once
#include <Wire.h>

#include "FirmwareTypes.h"
#include "SystemConfig.h"

struct ImuBatch {
  fw::ImuData samples[SystemConfig::kFifoBatchSize]{};
  size_t count = 0;
};

class ImuManager {
 public:
  explicit ImuManager(TwoWire& wire) : wire_(wire) {}
  fw::Result begin();  // Future per-sensor configuration and identity checks.
  // Only Ok exposes count usable entries for the requested sensor index.
  fw::Result readBatch(size_t sensor, ImuBatch& output);
  uint32_t losses(size_t sensor) const;
  bool available(size_t sensor) const;

 private:
  TwoWire& wire_;
};
