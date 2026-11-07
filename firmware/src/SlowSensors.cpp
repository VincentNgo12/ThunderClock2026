/**
 * File: SlowSensors.cpp
 * Purpose: Environment and heart-rate polling, shared snapshot updates and bring-up prints.
 * Contributors:
 * - Suvir Duggal
 */
#include "SlowSensors.h"

#include <Arduino.h>

#include "EnvironmentManager.h"
#include "HeartRateMonitor.h"
#include "SharedData.h"
#include "SystemConfig.h"

namespace {
EnvironmentManager* environment = nullptr;
HeartRateMonitor* heartRate = nullptr;
bool environmentReady = false;
bool heartRateReady = false;
uint32_t lastConversionMs = 0;
uint32_t lastPrintMs = 0;
fw::Environment latestEnvironment{};
fw::HeartRate latestHeartRate{};

const char* validityName(fw::Validity validity) {
  switch (validity) {
    case fw::Validity::Valid:
      return "ok";
    case fw::Validity::Stale:
      return "stale";
    case fw::Validity::Fault:
      return "fault";
    default:
      return "n/a";
  }
}

void printStatus() {
  if (!SystemConfig::kDebugEnabled || SystemConfig::kPrintHeartRateRaw) {
    return;  // Raw IR printing owns the serial port for the plotter.
  }
  if (environmentReady) {
    Serial.printf("ENV T=%.1fC(%s) RH=%.1f%%(%s) AQI=%.0f eCO2=%.0fppm TVOC=%.0fppb(%s)%s\n",
                  latestEnvironment.temperatureC.value,
                  validityName(latestEnvironment.temperatureC.stamp.validity),
                  latestEnvironment.humidityPercent.value,
                  validityName(latestEnvironment.humidityPercent.stamp.validity),
                  latestEnvironment.ens160Aqi.value, latestEnvironment.eco2Ppm.value,
                  latestEnvironment.tvocPpb.value,
                  validityName(latestEnvironment.eco2Ppm.stamp.validity),
                  environment->ens160WarmingUp() ? " [ENS160 warming up]" : "");
  }
  if (heartRateReady) {
    Serial.printf("HR  bpm=%.0f(%s) contact=%s ir=%lu overflows=%lu\n", latestHeartRate.bpm,
                  validityName(latestHeartRate.stamp.validity),
                  heartRate->skinContact() ? "yes" : "no",
                  static_cast<unsigned long>(heartRate->lastIr()),
                  static_cast<unsigned long>(heartRate->overflows()));
  }
}
}  // namespace

void slowSensorsBegin(TwoWire& wire) {
  if (SystemConfig::kInitEnvironment) {
    environment = new EnvironmentManager(wire);
    environmentReady = environment->begin() == fw::Result::Ok;
  }
  if (SystemConfig::kInitHeartRate) {
    heartRate = new HeartRateMonitor(wire);
    heartRateReady = heartRate->begin() == fw::Result::Ok;
  }
}

void slowSensorsPoll() {
  const uint32_t nowMs = millis();
  bool environmentChanged = false;
  bool heartRateChanged = false;

  if (environmentReady) {
    // Read first so a finished conversion is collected before the next trigger.
    if (environment->read(latestEnvironment) == fw::Result::Ok) {
      environmentChanged = true;
    }
    if (nowMs - lastConversionMs >= SystemConfig::kEnvironmentPeriodMs) {
      if (environment->startConversion() == fw::Result::Ok) {
        lastConversionMs = nowMs;
      }
    }
  }
  if (heartRateReady) {
    heartRate->poll(latestHeartRate);
    heartRateChanged = true;  // Validity can drop to unavailable, which must be shared too.
  }

  if ((environmentChanged || heartRateChanged) && snapshotMutex != nullptr &&
      xSemaphoreTake(snapshotMutex, pdMS_TO_TICKS(5)) == pdTRUE) {
    if (environmentChanged) {
      sharedSnapshot.environment = latestEnvironment;
    }
    if (heartRateChanged) {
      sharedSnapshot.heartRate = latestHeartRate;
    }
    xSemaphoreGive(snapshotMutex);
  }

  if (nowMs - lastPrintMs >= 1000) {
    lastPrintMs = nowMs;
    printStatus();
  }
}
