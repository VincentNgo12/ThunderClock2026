/**
 * File: BLEManager.h
 * Purpose: BLE startup and complete-packet publication boundary.
 * Contributors:
 * - Unassigned
 */
#pragma once

#include <Arduino.h>
#include <NimBLEDevice.h>
#include "FirmwareTypes.h"

extern NimBLECharacteristic* pCharIMU;
extern NimBLECharacteristic* pCharSnapshot;
extern bool deviceConnected;

fw::Result bleBegin();

