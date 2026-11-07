/**
 * File: SlowSensors.cpp
 * Purpose: Environment and heart-rate polling, latest-value storage and bring-up prints.
 * Contributors:
 * - Suvir Duggal
 */
#include "SlowSensors.h"

#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>

#include <cmath>

#include "EnvironmentManager.h"
#include "HeartRateMonitor.h"
#include "SystemConfig.h"

namespace {
EnvironmentManager* environment = nullptr;
HeartRateMonitor* heartRate = nullptr;
bool environmentReady = false;
bool heartRateReady = false;
uint32_t lastConversionMs = 0;
uint32_t lastPrintMs = 0;

// Latest values, guarded by latestMutex. Everything starts unavailable (NaN).
SemaphoreHandle_t latestMutex = nullptr;
fw::EnvData latestEnvironment{NAN, NAN, NAN, NAN, NAN};
fw::HeartData latestHeart{NAN, NAN};

void store(const fw::EnvData& env, const fw::HeartData& heart) {
  if (latestMutex != nullptr && xSemaphoreTake(latestMutex, pdMS_TO_TICKS(5)) == pdTRUE) {
    latestEnvironment = env;
    latestHeart = heart;
    xSemaphoreGive(latestMutex);
  }
}

void printStatus(const fw::EnvData& env, const fw::HeartData& heart) {
  if (!SystemConfig::kDebugEnabled || SystemConfig::kPrintHeartRateRaw) {
    return;  // Raw printing owns the serial port for the plotter.
  }
  if (environmentReady) {
    Serial.printf("ENV T=%.1fC RH=%.1f%% AQI=%.0f eCO2=%.0fppm TVOC=%.0fppb%s\n", env.temperature,
                  env.humidity, env.air_quality_index, env.eco2_ppm, env.tvoc_ppb,
                  environment->ens160WarmingUp() ? " [ENS160 warming up]" : "");
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
  if (SystemConfig::kInitEnvironment) {
    environment = new EnvironmentManager(wire, mux);
    environmentReady = environment->begin() == fw::Result::Ok;
  }
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
  fw::EnvData env{NAN, NAN, NAN, NAN, NAN};
  fw::HeartData heart{NAN, NAN};

  if (environmentReady) {
    // Read first so a finished conversion is collected before the next trigger.
    environment->read(env);
    if (nowMs - lastConversionMs >= SystemConfig::kEnvironmentPeriodMs) {
      if (environment->startConversion() == fw::Result::Ok) {
        lastConversionMs = nowMs;
      }
    }
  }
  if (heartRateReady) {
    heartRate->poll(heart);
  }
  store(env, heart);

  if (nowMs - lastPrintMs >= 1000) {
    lastPrintMs = nowMs;
    printStatus(env, heart);
  }
}

void slowSensorsLatest(fw::EnvData& env, fw::HeartData& heart) {
  env = {NAN, NAN, NAN, NAN, NAN};
  heart = {NAN, NAN};
  if (latestMutex != nullptr && xSemaphoreTake(latestMutex, pdMS_TO_TICKS(5)) == pdTRUE) {
    env = latestEnvironment;
    heart = latestHeart;
    xSemaphoreGive(latestMutex);
  }
}
