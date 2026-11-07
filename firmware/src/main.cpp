/**
 * File: main.cpp
 * Purpose: Arduino startup for the shared snapshot and optional empty tasks.
 * Contributors:
 * - Unassigned
 */
#include <Arduino.h>
#include <Wire.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "BLEManager.h"
#include "Debug.h"
#include "FirmwareTasks.h"
#include "ImuManager.h"
#include "SharedData.h"
#include "SystemConfig.h"

namespace {
ImuManager imuManager(Wire);
}

void setup() {
  Serial.begin(SystemConfig::kSerialBaud);
  debugPrintln("ThunderClock firmware scaffold. Hardware modules are unimplemented.");

//  snapshotMutex = xSemaphoreCreateMutex();
//   if (snapshotMutex == nullptr) {
//     debugPrintln("Snapshot mutex allocation failed.");
//     return;
//   }

  if (SystemConfig::kInitImuStrip) {
    Wire.begin(SystemConfig::kSdaPin, SystemConfig::kSclPin);
    if (imuManager.begin() != fw::Result::Ok) {
      debugPrintln("IMU initialization is unavailable.");
    }
  }
  if (SystemConfig::kInitBLE) {
    if (bleBegin() != fw::Result::Ok) {
      debugPrintln("BLE initialization is unavailable.");
    }
  }

  if (SystemConfig::kStartAcquisitionTask) {
    if (xTaskCreate(acquisitionTask, "acquisition", SystemConfig::kTaskStackBytes,
                    &imuManager, 1, nullptr) != pdPASS) {
      debugPrintln("Acquisition task creation failed.");
    }
  }
  if (SystemConfig::kStartTelemetryTask) {
    if (xTaskCreate(telemetryTask, "telemetry", SystemConfig::kTaskStackBytes,
                    nullptr, 1, nullptr) != pdPASS) {
      debugPrintln("Telemetry task creation failed.");
    }
  }
  if (SystemConfig::kStartAudioTask) {
    if (xTaskCreate(audioTask, "audio", SystemConfig::kTaskStackBytes,
                    nullptr, 1, nullptr) != pdPASS) {
      debugPrintln("Audio task creation failed.");
    }
  }
}

void loop() {
  delay(100);
}
