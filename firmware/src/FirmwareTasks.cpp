/**
 * File: FirmwareTasks.cpp
 * Purpose: FreeRTOS task bodies. Acquisition reads the IMU strip every 10 ms.
 * Contributors:
 * - Unassigned
 */
#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "FirmwareTasks.h"
#include "ImuManager.h"
#include "SystemConfig.h"

namespace {
const char* const kSlotNames[ImuManager::kSlotCount] = {"pelvis", "lumbar", "thoracic",
                                                        "upper thoracic"};
constexpr uint32_t kPrintEveryStrips = 10;  // ~10 Hz keeps 115200-baud Serial from falling behind.

void printStrip(const fw::StripData& strip) {
  Serial.printf("t=%lu ms\n", static_cast<unsigned long>(strip.time));
  for (size_t slot = 0; slot < ImuManager::kSlotCount; ++slot) {
    const fw::IMUData& sample = strip.data[slot];
    if (!sample.valid) {
      Serial.printf("  %u %-14s INVALID\n", static_cast<unsigned>(slot + 1), kSlotNames[slot]);
      continue;
    }
    Serial.printf("  %u %-14s a=(%7.2f %7.2f %7.2f) m/s2  g=(%6.2f %6.2f %6.2f) rad/s\n",
                  static_cast<unsigned>(slot + 1), kSlotNames[slot], sample.accel.x,
                  sample.accel.y, sample.accel.z, sample.gyro.x, sample.gyro.y, sample.gyro.z);
  }
}
}  // namespace

// argument is the ImuManager started in setup().
void acquisitionTask(void* argument) {
  ImuManager* imuManager = static_cast<ImuManager*>(argument);
  if (imuManager == nullptr) {
    vTaskDelete(nullptr);
    return;
  }

  fw::StripData strip{};
  uint32_t stripCount = 0;
  TickType_t lastWakeTime = xTaskGetTickCount();
  while (true) {
    // vTaskDelayUntil keeps a fixed 10 ms period even when a read takes a few ms.
    vTaskDelayUntil(&lastWakeTime, pdMS_TO_TICKS(SystemConfig::kAcquisitionPeriodMs));
    imuManager->read(strip);
    // TODO(FW-ORIENTATION): pass each slot to its OrientationProcessor.

    stripCount++;
    if (SystemConfig::kDebugEnabled && stripCount % kPrintEveryStrips == 0) {
      printStrip(strip);
    }
  }
}

void telemetryTask(void*) {
  while (true) {
    // TODO(FW-TELEMETRY): copy shared data under the mutex, then encode/send outside it.
    vTaskDelay(pdMS_TO_TICKS(SystemConfig::kTelemetryPeriodMs));
  }
}

void audioTask(void*) {
  while (true) {
    // TODO(FW-AUDIO): acquire bounded I2S windows and publish relative-noise summaries.
    vTaskDelay(pdMS_TO_TICKS(SystemConfig::kAudioIdlePeriodMs));
  }
}
