/**
 * File: I2cMux.cpp
 * Purpose: TCA9548A channel selection and a bring-up bus scan.
 * Contributors:
 * - Suvir Duggal
 */
#include "I2cMux.h"

#include <Arduino.h>

#include "SystemConfig.h"

namespace i2cmux {

fw::Result select(TwoWire& wire, int channel) {
  if (channel < 0 || channel > 7) {
    return fw::Result::InvalidArgument;
  }
  wire.beginTransmission(SystemConfig::kMuxAddress);
  wire.write(static_cast<uint8_t>(1u << channel));
  if (wire.endTransmission() != 0) {
    return fw::Result::IoFailure;
  }
  return fw::Result::Ok;
}

void scanAllChannels(TwoWire& wire) {
  Serial.println("I2C scan through the mux:");
  for (int channel = 0; channel < 8; channel++) {
    if (select(wire, channel) != fw::Result::Ok) {
      Serial.println("  mux did not answer at kMuxAddress; check SDA/SCL/power.");
      return;
    }
    Serial.printf("  ch%d:", channel);
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

}  // namespace i2cmux
