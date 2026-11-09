#pragma once
#include <Arduino.h>

#include "SystemConfig.h"

// Human-readable diagnostics only.
inline void debugPrintln(const char* message) {
  if (SystemConfig::kDebugEnabled) {
    Serial.println(message);
  }
}
