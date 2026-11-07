/**
 * File: I2cMux.h
 * Purpose: TCA9548A channel selection and a bring-up bus scan.
 * Contributors:
 * - Suvir Duggal
 */
#pragma once
#include <Wire.h>

#include "FirmwareTypes.h"

namespace i2cmux {
// Routes the shared bus to one downstream channel (0-7). Call from the I2C owner only.
fw::Result select(TwoWire& wire, int channel);

// Prints every responding address on each mux channel. Bring-up only: it blocks.
void scanAllChannels(TwoWire& wire);
}  // namespace i2cmux
