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
#include "EnvironmentManager.h"
#include "FirmwareTasks.h"
#include "HeartRateMonitor.h"
#include "ImuManager.h"
#include "SharedData.h"
#include "SystemConfig.h"

#include "AudioMonitor.h"

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
  debugPrintln("ThunderClock firmware scaffold. Hardware modules are unimplemented.");

  // -----------------------------------
  // INITIALIZE MUTEX & QUEUE 
  // -----------------------------------

  i2cMutex = xSemaphoreCreateMutex();
  if (i2cMutex == nullptr) Serial.println("i2c mutex failed");

  imuToSnapshotQueue = xQueueCreate(1, sizeof(fw::StripData));
  if (imuToSnapshotQueue == nullptr) Serial.println("imu to snapshot queue failed");


  // -----------------------------------
  // INITIALIZE SENSORS & PERIPHERALS
  // -----------------------------------

  // setup I2C peripheral
  Wire.begin(SystemConfig::kSdaPin, SystemConfig::kSclPin, SystemConfig::kI2cClockHz);
  Wire.setTimeOut(SystemConfig::kI2cTimeoutMs);

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

  if (SystemConfig::kInitEnvironment) {
    if (environmentManager.begin() != fw::Result::Ok) {
      Serial.println("env sensor failed to ini");
      while (true) delay(1000);
    }
    Serial.println("env sensor ini succesful");
  }

  if (SystemConfig::kInitHeartRate) {
    if (heartRateMonitor.begin() != fw::Result::Ok) {
      debugPrintln("heart-rate sensor failed to initialize.");
    }
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
  if (SystemConfig::kStartIMUTask) {
    if (xTaskCreate(IMUTask, "IMUTask", SystemConfig::kTaskStackBytes,
      nullptr, 1, nullptr) != pdPASS) {
      debugPrintln("IMU task creation failed.");
    }
  }
}

void loop() {
  // Support sensor bring-up without the acquisition task. The IMU task may
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
