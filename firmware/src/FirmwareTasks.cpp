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
void printRawStrip(const fw::StripData& stripData);

void makeIMUFieldsAvailable(telemetry_ImuStripData& message); 
void makeSnapshotFieldsAvailable(telemetry_orientation_v1_OrientationPacket& packet);

void fillOrientationPacket(const fw::Snapshot& snapshot, telemetry_orientation_v1_OrientationPacket& packet); 
void fillIMUMessage(telemetry_ImuStripData &message, fw::StripData &stripData);


void acquisitionTask(void*) {
  TickType_t lastWake = xTaskGetTickCount();
  TickType_t lastEnvTime = lastWake;
  TickType_t lastAudioTime = lastWake;
  TickType_t lastHeartTime = lastWake;

  const TickType_t imuPeriod = pdMS_TO_TICKS(20);      // 0.02 s, 50 Hz
  const TickType_t envPeriod = pdMS_TO_TICKS(1000);    // 1s
  const TickType_t audioPeriod = pdMS_TO_TICKS(100);   // 0.1s
  const TickType_t heartPeriod = pdMS_TO_TICKS(1000);  // 1s

  fw::Snapshot snapshot{};
  fw::StripData stripData{};
  OrientationProcessor orientationProcessor;
  orientationProcessor.begin();

  // protobuff settings
  telemetry_orientation_v1_OrientationPacket packet =
    telemetry_orientation_v1_OrientationPacket_init_zero;
  makeSnapshotFieldsAvailable(packet);
  packet.version = 2;
  uint32_t sequence = 0;
  uint8_t buffer[514]; // this is where encoded snapshot is held

  
  while (true) {
    const TickType_t now = xTaskGetTickCount();

    // env + heart read with mutex safety
    if (xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(10)) == pdTRUE) {

      // env read
      if (now - lastEnvTime >= envPeriod && SystemConfig::kInitEnv) {
        lastEnvTime = now;
        // env read
      } 

      // heart read
      if (now - lastHeartTime >= heartPeriod && SystemConfig::kInitHeartRate) {
        lastHeartTime = now; 
        // heart read
      }

      xSemaphoreGive(i2cMutex);
    } 

    // audio read
    if (now - lastAudioTime >= audioPeriod && SystemConfig::kInitAudio) {
      lastAudioTime = now;
      audioMonitor.read(snapshot.audioData);
    } 

    // imu read from queue 20 (hz)
    if (xQueueReceive(imuToSnapshotQueue, &stripData, portMAX_DELAY) == pdTRUE) {
      // sensor fusion to get q
      orientationProcessor.update(stripData, snapshot.stripData);

      // protobuff encoding
      packet.sequence = sequence++;
      fillOrientationPacket(snapshot, packet);
      pb_ostream_t stream = pb_ostream_from_buffer(buffer, sizeof(buffer));
      if (pb_encode(&stream, telemetry_orientation_v1_OrientationPacket_fields, &packet)) Serial.println("succesful snapshot encoding");;
      size_t length = stream.bytes_written;
      Serial.printf("Encoded %zu bytes\n", length); // print # of bytes encoded

      // send data ble
      pCharSnapshot->setValue(buffer, length);
      pCharSnapshot->notify();
    }

    // temporary print to serial
    // printSnapshot(snapshot);

    vTaskDelayUntil(&lastWake, imuPeriod);
  }
}


void IMUTask(void*) {
  TickType_t lastWake = xTaskGetTickCount();
  TickType_t lastImuToSnapshotTime = lastWake;

  const TickType_t imuToTelemetryPeriod = pdMS_TO_TICKS(10); // target: 100Hz 
  const TickType_t imuToSnapshotPeriod = pdMS_TO_TICKS(50);  // target: 20Hz 
  
  fw::StripData stripData{};

  // protobuf settings 
  telemetry_ImuStripData message = telemetry_ImuStripData_init_zero; // message representing imuStrip 
  message.data_count = 4; // set message repeated data field to 4 (4 imus)
  makeIMUFieldsAvailable(message);
  uint8_t buffer[300]; // this is where encoded snapshot is held

  while (true) {
    const TickType_t now = xTaskGetTickCount();

    if (xSemaphoreTake(i2cMutex, portMAX_DELAY) == pdTRUE) {
      imuManager.read(stripData);
      xSemaphoreGive(i2cMutex);
    }
    
    // encode data
    fillIMUMessage(message, stripData);
    pb_ostream_t stream = pb_ostream_from_buffer(buffer, sizeof(buffer));
    if (pb_encode(&stream, telemetry_ImuStripData_fields, &message)) Serial.println("succesful imu encoding");;
    size_t length = stream.bytes_written;
    Serial.printf("Encoded %zu bytes\n", length); // print # of bytes encoded
    // send data ble
    pCharIMU->setValue(buffer, length);
    pCharIMU->notify();

    if (now - lastImuToSnapshotTime >= imuToSnapshotPeriod) {
      xQueueOverwrite(imuToSnapshotQueue, &stripData);    
      lastImuToSnapshotTime = now;
    }

    // printRawStrip(stripData);

    vTaskDelayUntil(&lastWake, imuToTelemetryPeriod);
  }
}

void telemetryTask(void*) {
  while (true) {
    // TODO(FW-TELEMETRY): copy shared data under the mutex, then encode/send outside it.
    vTaskDelay(pdMS_TO_TICKS(SystemConfig::kTelemetryPeriodMs));
  }
}



// set fields to available
void makeIMUFieldsAvailable(telemetry_ImuStripData& message) {
  for (int i = 0; i < 4; ++i) {
    message.data[i].has_accel = true;
    message.data[i].has_gyro = true;
  } // set has fields of protobut to true so that it gets encoded
}

// set fields to available
void makeSnapshotFieldsAvailable(telemetry_orientation_v1_OrientationPacket& packet) {
    packet.has_version = true;
    packet.has_sequence = true;
    packet.has_sample_time_ms = true;
    packet.has_environment = true;
    packet.has_heart_rate_bpm = true;
    packet.has_spo2_percent = true;

    auto& env = packet.environment;
    env.has_temperature_c = true;
    env.has_humidity_percent = true;
    env.has_air_quality_index = true;
    env.has_eco2_ppm = true;
    env.has_tvoc_ppb = true;
    env.has_noise_relative_db = true;

    packet.orientations_count = 4;
    for (auto& orientation : packet.orientations) {
        orientation.has_sensor_id = true;
        orientation.has_valid = true;
        orientation.valid = true;
        orientation.has_rotation = true;
    }
}

// fill IMU message 
void fillIMUMessage(telemetry_ImuStripData &message, fw::StripData &stripData) {
  message.time = stripData.time;
  for (int i = 0; i < 4; ++i) {
    message.data[i].accel.x = stripData.data[i].accel.x;
    message.data[i].accel.y = stripData.data[i].accel.y;
    message.data[i].accel.z = stripData.data[i].accel.z;

    message.data[i].gyro.x = stripData.data[i].gyro.x;
    message.data[i].gyro.y = stripData.data[i].gyro.y;
    message.data[i].gyro.z = stripData.data[i].gyro.z;
  }
}

// fill orientation packet
void fillOrientationPacket(const fw::Snapshot& snapshot,
    telemetry_orientation_v1_OrientationPacket& packet) {

    packet.version = 2;
    packet.sample_time_ms = snapshot.stripData.time;

    for (pb_size_t i = 0; i < packet.orientations_count; ++i) {
        const auto& imu = snapshot.stripData.data[i];
        auto& orientation = packet.orientations[i];

        orientation.sensor_id = i + 1;
        orientation.valid = imu.valid;
        orientation.has_rotation = imu.valid;
        orientation.rotation = {};

        if (imu.valid) {
            const auto& q = imu.rotation;
            orientation.rotation = {q.w, q.x, q.y, q.z};
        }
    }

    const auto& env = snapshot.envData;
    packet.environment.temperature_c = env.temperature;
    packet.environment.humidity_percent = env.humidity;
    packet.environment.air_quality_index =
        static_cast<uint32_t>(env.air_quality_index);
    packet.environment.eco2_ppm = static_cast<uint32_t>(env.eco2_ppm);
    packet.environment.tvoc_ppb = static_cast<uint32_t>(env.tvoc_ppb);
    packet.environment.noise_relative_db = snapshot.audioData.relativeDb;

    packet.heart_rate_bpm = snapshot.heartData.bpm;
    packet.spo2_percent = snapshot.heartData.oxi;
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
void printRawStrip(const fw::StripData& rawStrip) {
    Serial.printf("\n=== Raw IMU data | timestamp: %lu ===\n",
                  static_cast<unsigned long>(rawStrip.time));

    unsigned index = 0;
    for (const auto& imu : rawStrip.data) {
        Serial.printf("  IMU[%u]\n", index++);
        Serial.printf("    Accel  x=% .4f  y=% .4f  z=% .4f\n",
                      imu.accel.x, imu.accel.y, imu.accel.z);
        Serial.printf("    Gyro   x=% .4f  y=% .4f  z=% .4f\n",
                      imu.gyro.x, imu.gyro.y, imu.gyro.z);
    }
}