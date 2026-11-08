/**
 * File: SlowSensors.cpp
 * Purpose: I2C bus scan through the mux for sensor bring-up.
 * Contributors:
 * - Suvir Duggal
 */
#include "SlowSensors.h"

#include <Arduino.h>
#include "SystemConfig.h"

void slowSensorsScanBus(TwoWire& wire, ImuManager& mux) {
  Serial.println("I2C scan through the mux:");
  for (uint8_t channel = 0; channel < 8; channel++) {
    mux.setChannel(channel);
    Serial.printf("  ch%u:", channel);
    int found = 0;
    for (uint8_t address = 1; address < 127; address++) {
      if (address == SystemConfig::kMuxAddress) {
        continue;
      }
      wire.beginTransmission(address);
      if (wire.endTransmission() == 0) {
        Serial.printf(" 0x%02X", address);
        found++;
      }
    }
    Serial.println(found == 0 ? " (none)" : "");
  }
}
