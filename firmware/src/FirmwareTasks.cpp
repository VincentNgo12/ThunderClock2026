/**
 * File: FirmwareTasks.cpp
 * Purpose: FreeRTOS task bodies. Acquisition reads the IMU strip every 10 ms.
 * Contributors:
 * - Unassigned
 */
#include "FirmwareTasks.h"
#include "ImuManager.h"
#include "OrientationProcessor.h"

void printSnapshot(const fw::Snapshot& snapshot);

void acquisitionTask(void*) {
  TickType_t lastWake = xTaskGetTickCount();
  TickType_t lastEnvTime = lastWake;
  TickType_t lastAudioTime = lastWake;
  TickType_t lastHeartTime = lastWake;

  const TickType_t imuPeriod = pdMS_TO_TICKS(20);      // 0.05s
  const TickType_t envPeriod = pdMS_TO_TICKS(1000);    // 1s
  const TickType_t audioPeriod = pdMS_TO_TICKS(100);   // 0.1s
  const TickType_t heartPeriod = pdMS_TO_TICKS(1000);  // 1s

  fw::Snapshot snapshot{};
  fw::StripData strip{};
  OrientationProcessor orientationProcessor;
  orientationProcessor.begin();
  
  while (true) {
    const TickType_t now = xTaskGetTickCount();


    // env read
    if (now - lastEnvTime >= envPeriod) {
      lastEnvTime = now;
      // env read
    } 

    // audio read
    if (now - lastAudioTime >= audioPeriod) {
      lastAudioTime = now;
      audioMonitor.read(snapshot.audioData);
    } 

    // heart read
    if (now - lastHeartTime >= heartPeriod) {
      lastHeartTime = now; 
      // heart read
    } 

    // imu read from queue 20 (hz)
    // Fuse every queued strip, not only the newest: each filter needs all 100 Hz samples.
    orientationProcessor.update(strip, snapshot.stripData);
    // push it through queue

    // temporary print to serial
    printSnapshot(snapshot);

    vTaskDelayUntil(&lastWake, imuPeriod);
  }
}

// argument is the ImuManager started in setup(). Reads all 4 IMUs every 10 ms.
void IMUTask(void* argument) {
  ImuManager* imuManager = static_cast<ImuManager*>(argument);
  if (imuManager == nullptr) {
    vTaskDelete(nullptr);
    return;
  }

  fw::StripData strip{};
  TickType_t lastWake = xTaskGetTickCount();
  while (true) {
    vTaskDelayUntil(&lastWake, pdMS_TO_TICKS(SystemConfig::kAcquisitionPeriodMs));
    if (imuManager->read(strip) == fw::Result::Ok && imuQueue != nullptr) {
      xQueueSend(imuQueue, &strip, 0);  // Drop when full rather than stall the 100 Hz loop.
    }
  }
}

void telemetryTask(void*) {
  while (true) {
    // TODO(FW-TELEMETRY): copy shared data under the mutex, then encode/send outside it.
    vTaskDelay(pdMS_TO_TICKS(SystemConfig::kTelemetryPeriodMs));
  }
}


// temporary helper to print snapshot for debug
void printSnapshot(const fw::Snapshot& snapshot) {
    const auto& strip = snapshot.stripData;
    const auto& env = snapshot.envData;
    const auto& heart = snapshot.heartData;
    const auto& audio = snapshot.audioData;

    Serial.println("\n=== Snapshot ===");

    Serial.printf("IMU timestamp: %lu\n", static_cast<unsigned long>(strip.time));
    for (size_t i = 0; i < sizeof(strip.data) / sizeof(strip.data[0]); ++i) {
        if (!strip.data[i].valid) {
            Serial.printf("  IMU[%u]  INVALID\n", static_cast<unsigned>(i));
            continue;
        }
        const auto& q = strip.data[i].rotation;
        Serial.printf("  IMU[%u]  w=% .4f  x=% .4f  y=% .4f  z=% .4f\n",
                      static_cast<unsigned>(i), q.w, q.x, q.y, q.z);
    }

    Serial.println("Environment");
    Serial.printf("  Temperature : %7.2f C\n", env.temperature);
    Serial.printf("  Humidity    : %7.2f %%\n", env.humidity);
    Serial.printf("  Air quality : %7.0f AQI\n", env.air_quality_index);
    Serial.printf("  eCO2        : %7.0f ppm\n", env.eco2_ppm);
    Serial.printf("  TVOC        : %7.0f ppb\n", env.tvoc_ppb);

    Serial.println("Heart");
    Serial.printf("  Heart rate  : %7.1f BPM\n", heart.bpm);
    Serial.printf("  SpO2        : %7.1f %%\n", heart.oxi);

    Serial.println("Audio");
    Serial.printf("  Loudness    : %7.2f dB relative\n", audio.relativeDb);
}