/**
 * File: SlowSensors.h
 * Purpose: Heart-rate polling and I2C bus-scan helpers for bring-up.
 * Contributors:
 * - Suvir Duggal
 */
#pragma once
#include <Wire.h>

#include "FirmwareTypes.h"
#include "ImuManager.h"

// Call once after Wire.begin(). Skipped when heart rate is disabled in SystemConfig.
// mux is the IMU manager, whose setChannel() selects the heart-rate channel.
void slowSensorsBegin(TwoWire& wire, ImuManager& mux);

// Prints every responding address on each mux channel. Bring-up only: it blocks.
void slowSensorsScanBus(TwoWire& wire, ImuManager& mux);

// Call every kSlowSensorPollMs from the I2C owner: loop() while the acquisition
// task is disabled, or the acquisition task between IMU reads once it is enabled.
// Never call it from two places at once; the mux is shared.
void slowSensorsPoll();

// Thread-safe copy of the latest readings for telemetry. NaN means unavailable,
// so the encoder should omit that field. Safe to call from any task.
void slowSensorsLatest(fw::HeartData& heart);
