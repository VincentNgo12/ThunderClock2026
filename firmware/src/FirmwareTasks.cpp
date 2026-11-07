/**
 * File: FirmwareTasks.cpp
 * Purpose: Disabled-by-default FreeRTOS task shells.
 * Contributors:
 * - Unassigned
 */
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "FirmwareTasks.h"
#include "SystemConfig.h"

void acquisitionTask(void*) {
  while (true) {
    // TODO(FW-ACQUISITION): own I2C reads and per-sensor calibration/fusion here.
    vTaskDelay(pdMS_TO_TICKS(SystemConfig::kAcquisitionPeriodMs));
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
