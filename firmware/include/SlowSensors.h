/**
 * File: SlowSensors.h
 * Purpose: Schedules environment and heart-rate polling for whichever code owns I2C.
 * Contributors:
 * - Suvir Duggal
 */
#pragma once
#include <Wire.h>

// Call once after Wire.begin(). Sensors disabled in SystemConfig are skipped.
void slowSensorsBegin(TwoWire& wire);

// Call every kSlowSensorPollMs from the I2C owner: loop() while the acquisition
// task is disabled, or the acquisition task between IMU reads once it is enabled.
// Never call it from two places at once; the mux is shared.
void slowSensorsPoll();
