#pragma once
#include <cstdint>

inline uint32_t testMillis = 0;
inline uint32_t testDelayedMs = 0;

inline uint32_t millis() { return testMillis; }
inline void delay(uint32_t milliseconds) {
  testMillis += milliseconds;
  testDelayedMs += milliseconds;
}

struct TestSerial {
  void println(const char*) {}
};
inline TestSerial Serial;
