/**
 * File: SlowSensors.h
 * Purpose: Schedules environment and heart-rate polling for whichever code owns I2C.
 * Contributors:
 * - Suvir Duggal
 */
#pragma once
#include <Wire.h>

#include "FirmwareTypes.h"

// Call once after Wire.begin(). Sensors disabled in SystemConfig are skipped.
void slowSensorsBegin(TwoWire& wire);

// Call every kSlowSensorPollMs from the I2C owner: loop() while the acquisition
// task is disabled, or the acquisition task between IMU reads once it is enabled.
// Never call it from two places at once; the mux is shared.
void slowSensorsPoll();

// Thread-safe copy of the latest readings for telemetry. NaN means unavailable,
// so the encoder should omit that field. Safe to call from any task.
void slowSensorsLatest(fw::EnvData& environment, fw::HeartData& heart);
