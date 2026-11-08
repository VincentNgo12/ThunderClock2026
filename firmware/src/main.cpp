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

#include "FirmwareTasks.h"
#include "SharedData.h"
#include "SystemConfig.h"
#include "Debug.h"

#include "ImuManager.h"
#include "AudioMonitor.h"
#include "EnvironmentManager.h"
#include "HeartRateMonitor.h"
#include "BLEManager.h"

// queues & mutexes
QueueHandle_t imuToSnapshotQueue = nullptr;
SemaphoreHandle_t i2cMutex = nullptr;

// sensor objects
AudioMonitor audioMonitor{
  SystemConfig::kI2sBclkPin,
  SystemConfig::kI2sWsPin,
  SystemConfig::kI2sDataPin
}; 
ImuManager imuManager(Wire);
EnvironmentManager environmentManager(Wire);
HeartRateMonitor heartRateMonitor(Wire);

void setup() {
  Serial.begin(SystemConfig::kSerialBaud);
  while (!Serial) delay(500);

  // -----------------------------------
  // INITIALIZE MUTEX & QUEUE 
  // -----------------------------------

  i2cMutex = xSemaphoreCreateMutex();
  if (i2cMutex == nullptr) Serial.println("i2c mutex FAILED");

  imuToSnapshotQueue = xQueueCreate(1, sizeof(fw::StripData));
  if (imuToSnapshotQueue == nullptr) Serial.println("imu2snapshot queue FAILED");


  // -----------------------------------
  // INITIALIZE SENSORS & PERIPHERALS
  // -----------------------------------

  // setup I2C peripheral
  Wire.begin(SystemConfig::kSdaPin, SystemConfig::kSclPin, SystemConfig::kI2cClockHz);
  Wire.setTimeOut(SystemConfig::kI2cTimeoutMs);

  if (SystemConfig::kInitImuStrip) {
    if (imuManager.begin() != fw::Result::Ok) {
      Serial.println("IMU Strip Init: FAILED");
      while (true) delay(1000);
    }
    Serial.println("IMU Strip Init: SUCCESFUL");
  }

  if (SystemConfig::kInitAudio) {
    if (audioMonitor.begin() != fw::Result::Ok) {
      Serial.println("AUDIO Sensor Init: FAILED");
      while (true) delay(1000);
    };
    Serial.println("AUDIO Sensor Init: SUCCESFUL");
  }

  if (SystemConfig::kInitEnvironment) {
    if (environmentManager.begin() != fw::Result::Ok) {
      Serial.println("ENV Sensor Init: FAILED");
      while (true) delay(1000);
    }
    Serial.println("ENV Sensor Init: SUCCESFUL");
  }

  if (SystemConfig::kInitHeartRate) {
    if (heartRateMonitor.begin() != fw::Result::Ok) {
      Serial.println("HEART Sensor Init: FAILED");
      while (true) delay(1000);
    }
    Serial.println("HEART Sensor Init: SUCCESFUL");
  }

  if (SystemConfig::kInitBLE) {
    if (bleBegin() != fw::Result::Ok) {
      Serial.println("BLE Stack Init: FAILED");
      while (true) delay(1000);
    }
    Serial.println("BLE Stack Init: SUCCESFUL");
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
  if (SystemConfig::kStartIMUTask) {
    if (xTaskCreate(IMUTask, "IMUTask", SystemConfig::kTaskStackBytes,
      nullptr, 1, nullptr) != pdPASS) {
      debugPrintln("IMU task creation failed.");
    }
  }
}

void loop() {
  // support sensor bring-up without the acquisition task. The IMU task may
  // still be running, so mux selection and reads share its I2C mutex.
  if (!SystemConfig::kStartAcquisitionTask) {
    if (xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(10)) == pdTRUE) {
      if (SystemConfig::kInitEnvironment) {
        fw::EnvData environment;
        environmentManager.read(environment);
      }
      if (SystemConfig::kInitHeartRate) {
        fw::HeartData heart;
        heartRateMonitor.read(heart);
      }
      xSemaphoreGive(i2cMutex);
    }
  }
  delay(SystemConfig::kSlowSensorPollMs);
}
