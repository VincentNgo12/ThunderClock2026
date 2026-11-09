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

// Show fatal startup errors even before loop() starts running.
void stopOnError() {
  bleReportError();
  bleUpdateStatus();
  while (true) {
    delay(1000);
  }
}

void setup() {
  delay(1000);
  bleStatusBegin();
  Serial.begin(SystemConfig::kSerialBaud);
  // while (!Serial) delay(500);

  // -----------------------------------
  // INITIALIZE MUTEX & QUEUE 
  // -----------------------------------

  i2cMutex = xSemaphoreCreateMutex();
  if (i2cMutex == nullptr) {
    Serial.println("i2c mutex FAILED");
    stopOnError();
  }

  imuToSnapshotQueue = xQueueCreate(1, sizeof(fw::StripData));
  if (imuToSnapshotQueue == nullptr) {
    Serial.println("imu2snapshot queue FAILED");
    stopOnError();
  }


  // -----------------------------------
  // INITIALIZE SENSORS & PERIPHERALS
  // -----------------------------------

  // setup I2C peripheral
  if (!Wire.begin(SystemConfig::kSdaPin, SystemConfig::kSclPin, SystemConfig::kI2cClockHz)) {
    Serial.println("I2C Init: FAILED");
    stopOnError();
  }
  Wire.setTimeOut(SystemConfig::kI2cTimeoutMs);

  if (SystemConfig::kInitImuStrip) {
    if (imuManager.begin() != fw::Result::Ok) {
      Serial.println("IMU Strip Init: FAILED");
      stopOnError();
    }
    Serial.println("IMU Strip Init: SUCCESFUL");
  }

  if (SystemConfig::kInitAudio) {
    if (audioMonitor.begin() != fw::Result::Ok) {
      Serial.println("AUDIO Sensor Init: FAILED");
      stopOnError();
    };
    Serial.println("AUDIO Sensor Init: SUCCESFUL");
  }

  if (SystemConfig::kInitEnvironment) {
    if (environmentManager.begin() != fw::Result::Ok) {
      Serial.println("ENV Sensor Init: FAILED");
      stopOnError();
    }
    Serial.println("ENV Sensor Init: SUCCESFUL");
  }

  if (SystemConfig::kInitHeartRate) {
    if (heartRateMonitor.begin() != fw::Result::Ok) {
      Serial.println("HEART Sensor Init: FAILED");
      stopOnError();
    }
    Serial.println("HEART Sensor Init: SUCCESFUL");
  }

  if (SystemConfig::kInitBLE) {
    if (bleBegin() != fw::Result::Ok) {
      Serial.println("BLE Stack Init: FAILED");
      stopOnError();
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
      stopOnError();
    }
  }
  if (SystemConfig::kStartIMUTask) {
    if (xTaskCreate(IMUTask, "IMUTask", SystemConfig::kTaskStackBytes,
      nullptr, 1, nullptr) != pdPASS) {
      debugPrintln("IMU task creation failed.");
      stopOnError();
    }
  }
}

void loop() {
  bleUpdateStatus();
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
