#pragma once

#include <Arduino.h>
#include <NimBLEDevice.h>
#include "FirmwareTypes.h"

extern NimBLECharacteristic* pCharIMU;
extern NimBLECharacteristic* pCharSnapshot;

fw::Result bleBegin();

// Initialize the onboard NeoPixel before any startup operation can fail.
void bleStatusBegin();
// Call from Arduino loop(); callbacks only change state, never write the LED.
void bleUpdateStatus();
// Latches red until reboot. Also used for sensor/resource/task startup failures.
void bleReportError();
