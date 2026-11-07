/**
 * File: BLEManager.h
 * Purpose: BLE startup and complete-packet publication boundary.
 * Contributors:
 * - Unassigned
 */
#pragma once
#include "FirmwareTypes.h"

fw::Result bleBegin();  // Future GATT setup and advertising.
// Telemetry supplies complete encoded packets for the agreed characteristics.
fw::Result blePublish(const uint8_t* raw, size_t rawLength,
                      const uint8_t* orientations, size_t orientationLength);
