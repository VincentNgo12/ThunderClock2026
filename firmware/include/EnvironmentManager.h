/**
 * File: EnvironmentManager.h
 * Purpose: Acquisition-owned AHT21 and ENS160 adapter boundary.
 * Contributors:
 * - Unassigned
 */
#pragma once
#include <Wire.h>

#include "FirmwareTypes.h"

class EnvironmentManager {
 public:
  explicit EnvironmentManager(TwoWire& wire) : wire_(wire) {}
  fw::Result begin();  // Future sensor setup and warm-up handling.
  fw::Result startConversion();  // Request due readings without waiting.
  fw::Result read(fw::Environment& output);  // Future individually stamped fields.

 private:
  TwoWire& wire_;
};
