/**
 * File: SlowSensors.h
 * Purpose: I2C bus-scan helper for sensor bring-up.
 * Contributors:
 * - Suvir Duggal
 */
#pragma once
#include <Wire.h>

#include "ImuManager.h"

// Prints every responding address on each mux channel. Bring-up only: it blocks.
void slowSensorsScanBus(TwoWire& wire, ImuManager& mux);
