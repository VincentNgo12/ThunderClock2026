/**
 * File: EnvironmentManager.cpp
 * Purpose: Unimplemented temperature, humidity and air-quality acquisition.
 * Contributors:
 * - Unassigned
 */
#include "EnvironmentManager.h"

fw::Result EnvironmentManager::begin() {
  // TODO(FW-ENV): configure confirmed AHT21/ENS160 addresses and handle warm-up.
  return fw::Result::NotImplemented;
}

fw::Result EnvironmentManager::startConversion() {
  // TODO(FW-ENV): start due conversions without blocking IMU acquisition.
  return fw::Result::NotImplemented;
}

fw::Result EnvironmentManager::read(fw::EnvData& output) {
  output = fw::EnvData{};
  // TODO(FW-ENV): collect completed fields with their own acquisition times and validity.
  return fw::Result::NotImplemented;
}
