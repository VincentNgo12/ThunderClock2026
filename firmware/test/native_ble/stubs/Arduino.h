/**
 * File: Arduino.h
 * Purpose: Clock, GPIO and RGB LED observations for BLE state host checks.
 * Contributors: Unassigned
 */
#pragma once
#include <cstdint>

constexpr int OUTPUT = 1;
constexpr int HIGH = 1;
inline uint32_t testMillis = 0;
inline int poweredPin = -1;
inline int outputPin = -1;
inline int ledPin = -1;
inline uint8_t ledRed = 0, ledGreen = 0, ledBlue = 0;
inline uint32_t millis() { return testMillis; }
inline void pinMode(int pin, int) { outputPin = pin; }
inline void digitalWrite(int pin, int value) { poweredPin = value == HIGH ? pin : -1; }
inline void rgbLedWrite(uint8_t pin, uint8_t red, uint8_t green, uint8_t blue) {
  ledPin = pin;
  ledRed = red;
  ledGreen = green;
  ledBlue = blue;
}
struct TestSerial {
  void println(const char*) {}
  void println(uint16_t) {}
  void print(const char*) {}
};
inline TestSerial Serial;
