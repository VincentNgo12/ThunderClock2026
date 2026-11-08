/**
 * File: SlowSensors.cpp
 * Purpose: Heart-rate polling, latest-value storage and I2C bring-up prints.
 * Contributors:
 * - Suvir Duggal
 */
#include "SlowSensors.h"

#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>

#include <cmath>

#include "HeartRateMonitor.h"
#include "SystemConfig.h"

namespace {
HeartRateMonitor* heartRate = nullptr;
bool heartRateReady = false;
uint32_t lastPrintMs = 0;

// Latest values, guarded by latestMutex. Everything starts unavailable (NaN).
SemaphoreHandle_t latestMutex = nullptr;
fw::HeartData latestHeart{NAN, NAN};

void store(const fw::HeartData& heart) {
  if (latestMutex != nullptr && xSemaphoreTake(latestMutex, pdMS_TO_TICKS(5)) == pdTRUE) {
    latestHeart = heart;
    xSemaphoreGive(latestMutex);
  }
}

void printStatus(const fw::HeartData& heart) {
  if (!SystemConfig::kDebugEnabled || SystemConfig::kPrintHeartRateRaw) {
    return;  // Raw printing owns the serial port for the plotter.
  }
  if (heartRateReady) {
    Serial.printf("HR  bpm=%.0f SpO2=%.1f%% contact=%s ir=%lu overflows=%lu\n", heart.bpm, heart.oxi,
                  heartRate->skinContact() ? "yes" : "no",
                  static_cast<unsigned long>(heartRate->lastIr()),
                  static_cast<unsigned long>(heartRate->overflows()));
  }
}
}  // namespace

void slowSensorsBegin(TwoWire& wire, ImuManager& mux) {
  latestMutex = xSemaphoreCreateMutex();
  if (SystemConfig::kInitHeartRate) {
    heartRate = new HeartRateMonitor(wire, mux);
    heartRateReady = heartRate->begin() == fw::Result::Ok;
  }
}

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

void slowSensorsPoll() {
  const uint32_t nowMs = millis();
  fw::HeartData heart{NAN, NAN};

  if (heartRateReady) {
    heartRate->poll(heart);
  }
  store(heart);

  if (nowMs - lastPrintMs >= 1000) {
    lastPrintMs = nowMs;
    printStatus(heart);
  }
}

void slowSensorsLatest(fw::HeartData& heart) {
  heart = {NAN, NAN};
  if (latestMutex != nullptr && xSemaphoreTake(latestMutex, pdMS_TO_TICKS(5)) == pdTRUE) {
    heart = latestHeart;
    xSemaphoreGive(latestMutex);
  }
}
