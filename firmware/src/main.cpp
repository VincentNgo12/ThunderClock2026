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
#include "SlowSensors.h"
#include "SystemConfig.h"

#include "AudioMonitor.h"

// queues & mutexes
QueueHandle_t imuToTelemetryQueue = nullptr;
QueueHandle_t imuToSnapshotQueue = nullptr;
QueueHandle_t snapshotToTelemetryQueue = nullptr;
SemaphoreHandle_t i2cMutex = nullptr;

// sensor objects
AudioMonitor audioMonitor{
  SystemConfig::kI2sBclkPin,
  SystemConfig::kI2sWsPin,
  SystemConfig::kI2sDataPin
}; 
ImuManager imuManager(Wire);

void setup() {
  Serial.begin(SystemConfig::kSerialBaud);
  while (!Serial) delay(500);
  debugPrintln("ThunderClock firmware scaffold. Hardware modules are unimplemented.");

  // -----------------------------------
  // INITIALIZE MUTEX & QUEUE 
  // -----------------------------------

  i2cMutex = xSemaphoreCreateMutex();
  if (i2cMutex == nullptr) Serial.println("i2c mutex failed");

  imuToTelemetryQueue = xQueueCreate(1, sizeof(fw::StripData));
  if (imuToTelemetryQueue == nullptr) Serial.println("imu to telemetry queue failed");
  imuToSnapshotQueue = xQueueCreate(1, sizeof(fw::StripData));
  if (imuToSnapshotQueue == nullptr) Serial.println("imu to snapshot queue failed");
  snapshotToTelemetryQueue = xQueueCreate(1, sizeof(fw::Snapshot));
  if (snapshotToTelemetryQueue == nullptr) Serial.println("snapshot to telemetry queue failed");

  // ?? \/ ??
  const bool needsI2c = SystemConfig::kInitImuStrip || SystemConfig::kInitEnvironment ||
                        SystemConfig::kInitHeartRate || SystemConfig::kRunI2cScanAtBoot;
  if (needsI2c) {
    Wire.begin(SystemConfig::kSdaPin, SystemConfig::kSclPin, SystemConfig::kI2cClockHz);
    Wire.setTimeOut(SystemConfig::kI2cTimeoutMs);
  }
  if (SystemConfig::kRunI2cScanAtBoot) {
    slowSensorsScanBus(Wire, imuManager);
  }
  slowSensorsBegin(Wire, imuManager);

  // -----------------------------------
  // INITIALIZE SENSORS & PERIPHERALS
  // -----------------------------------
  if (SystemConfig::kInitImuStrip) {
    if (imuManager.begin() != fw::Result::Ok) {
      Serial.println("failed to ini imu strip");
      while (true) delay(1000);
    }
    Serial.println("imu strip ini succesful");
  }

  if (SystemConfig::kInitAudio) {
    if (audioMonitor.begin() != fw::Result::Ok) {
      Serial.println("failed to ini audio audio sensor");
      while (true) delay(1000);
    };
    Serial.println("audio sensor ini succesful");
  }

  if (SystemConfig::kInitBLE) {
    if (bleBegin() != fw::Result::Ok) {
      Serial.println("ble stack failed to ini");
    }
    Serial.println("ble stack ini succesful");
  }

  // -----------------------------------
  // INITIALIZE RTOS TASKS 
  // -----------------------------------

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
  if (SystemConfig::kStartIMUTask) {
    if (xTaskCreate(IMUTask, "IMUTask", SystemConfig::kTaskStackBytes,
                    nullptr, 1, nullptr) != pdPASS) {
      debugPrintln("IMU task creation failed.");
    }
  }
}

void loop() {
  // Until the acquisition task owns I2C, loop() polls the slow sensors.
  // TODO(FW-ACQUISITION): move this call into acquisitionTask when it is enabled.
  if (!SystemConfig::kStartAcquisitionTask) {
    slowSensorsPoll();
  }
  delay(SystemConfig::kSlowSensorPollMs);
}
